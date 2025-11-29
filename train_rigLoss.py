import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import numpy as np
import torch
from Functions import transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from model import ED_lv1, \
    ED_lv2, ED_lv3, SpatialTransform_unit, \
    SpatialTransformNearest_unit, smoothloss, multi_resolution_NCC

from Functions import train_data, AverageMeter,val_dataset
from rigid_filed import extract_class_masks
from rigid_dice import *
from IC import IncompressibilityConstraint
from DC import distances_loss

def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    :return a list as the label length
    """

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl3_rigid", type=int,
                    dest="iteration_lvl3_rigid", default=5000,
                    help="number of lvl3 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=1,
                    help="Anti-fold loss: suggested range 0 to 1000") #0.01
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=1,
                    help="Gradient smooth loss: suggested range 0.1 to 10")# 0.1
parser.add_argument("--rigid", type=float,
                    dest="rigid", default=0.01,
                    help="rigid filed loss: suggested range 0.01 to 1")# 0.1
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=2000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='./data/train/',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=2000,
                    help="Number step for freezing the previous level")
parser.add_argument("--val_dir", type=str,
                    dest="val_dir", default='./data/val/',
                    help="fixed image")

parser.add_argument("--pretrained", type=str,
                    dest="pretrained", default='',
                    help="path to a pretrained model to load (optional)")
parser.add_argument("--model_dir", type=str,
                    dest="model_dir", default='./models',
                    help="directory where models/checkpoints are saved")

opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath
freeze_step = opt.freeze_step
rigid = opt.rigid
val_file = opt.val_dir



iteration_lvl3_rigid = opt.iteration_lvl3_rigid

model_name = "RER_"
torch.autograd.set_detect_anomaly(True)


def train_rig():
    print("Training lvl3...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    model_lvl1 = ED_lv1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).to(device)
           
    model_lvl2 = ED_lv2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)


    model = ED_lv3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2).to(device)
    
    # Load pretrained model if provided via --pretrained
    if getattr(opt, 'pretrained', ''):
        print("Loading pretrained model:", opt.pretrained)
        model.load_state_dict(torch.load(opt.pretrained,  map_location='cuda'))
    else:
        print("No pretrained model provided; training from scratch.")

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss

    transform = SpatialTransform_unit().to(device)
    transform_label = SpatialTransformNearest_unit().cuda()
    transform_label.eval()

    IC = IncompressibilityConstraint().cuda()

    rigid_dice_tran = RigidDiceLoss().cuda()


    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    data = train_data(datapath) 
    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = opt.model_dir
    load_model = False
    if load_model is True:
        model_path = os.path.join(opt.model_dir, ".pth")
        print("Loading weight: ", model_path)
        model.load_state_dict(torch.load(model_path, map_location='cuda'))

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    training_generator = DataLoader(data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    DS_val, _ = val_dataset(val_file)
    test_loader = DataLoader(DS_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
      
    epoch_start = 0
    max_epoch = 100

    for epoch in range(epoch_start, max_epoch):

        Loss_IC = []
        Loss_sim = []
        Loss_smo = []
        Loss_dice = []
        for Y, X, Y_seg, X_seg in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()
            X_seg = X_seg.float()
            Y_seg = Y_seg.float()

            F_X_Y, X_Y, Y_4x, _, _, _, _ = model(X, Y, X_seg, Y_seg)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)
            Loss_sim.append(loss_multiNCC.item())
            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            X_seg = X_seg.to(device)

            X_Y_seg = transform_label(X_seg, F_X_Y.permute(0, 2, 3, 4, 1), grid)

            y_source_list = extract_class_masks(X_Y_seg)
            sorce_list = extract_class_masks(X_seg)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            
            rigid_penalty_LOSS = 0
            for cls, _ in sorce_list.items():
                y_source_oh = y_source_list[cls]
                source_oh = sorce_list[cls]
                rigid_penalty = distances_loss(source_oh, F_X_Y_norm.permute(0,4,1,2,3), 20)
                rigid_penalty_LOSS = rigid_penalty_LOSS + rigid_penalty

            Loss_dice.append((rigid_penalty_LOSS/4).item())

            loss_regulation = loss_smooth(F_X_Y_norm.permute(0,4,1,2,3))
            Loss_smo.append(loss_regulation.item())
            loss_IC = IC(X_Y_seg, F_X_Y)
            Loss_IC.append((loss_IC ).item())
            loss = loss_multiNCC  + 1 * loss_regulation  + 0.008 * rigid_penalty_LOSS + loss_IC * 0.2 
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step() 

        # val
        test_dsc = AverageMeter()
        test_raw_dsc = AverageMeter()
        test_loss = AverageMeter()
        dis_Loss = AverageMeter()

        if epoch % 2 == 0:
            with torch.no_grad():
                model.eval()
                print("\nValidation")
                for fixed_img, moving_img, y_seg, x_seg, y_labels in test_loader:
                    fixed_img = fixed_img.to(device).float() 
                    moving_img = moving_img.to(device).float() 
                    x_seg = x_seg.float()  # bone_moving
                    y_seg = y_seg.float() 
                    y_labels =  y_labels.float() 
                    labels = [1.0,2.0,3.0,4.0]
                    F_X_Y, X_Y, Y_4x, _, _, _, _ = model(moving_img, fixed_img, x_seg, y_seg)
                    x_seg = x_seg.to(device)
                    y_labels = y_labels.to(device)

                    x_y_seg = transform_label(x_seg, F_X_Y.permute(0, 2, 3, 4, 1), grid)
                    y_source_list = extract_class_masks(x_y_seg)
                    sorce_list = extract_class_masks(x_seg)

                    F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())
                    dis_loss = 0
                    for cls, _ in sorce_list.items():
                        y_source_oh = y_source_list[cls]
                        source_oh = sorce_list[cls]
                        dis_penalty = distances_loss(source_oh, F_X_Y_norm.permute(0,4,1,2,3), 20)
                        dis_loss = dis_loss + dis_penalty 
                    dis_Loss.update((dis_loss.detach().cpu())/4, Y.size(0))

                    rigid_penalty_LOSS = 0
                    with torch.enable_grad():
                        for cls, _ in sorce_list.items():
                            y_source_oh = y_source_list[cls]
                            source_oh = sorce_list[cls]
                            rigid_penalty = rigid_dice_tran(y_source_oh, source_oh, F_X_Y_norm.permute(0,4,1,2,3))
                            rigid_penalty_LOSS = rigid_penalty_LOSS + rigid_penalty
                    

                    test_loss.update((rigid_penalty_LOSS.detach().cpu())/4, Y.size(0))
                    dsc_trans = np.sum(dice(y_labels.detach().cpu().numpy()[0, 0, :, :, :], x_y_seg.detach().cpu().numpy()[0, 0, :, :, :], labels))/len(labels)
                    dsc_raw = np.sum(dice(y_labels.detach().cpu().numpy()[0, 0, :, :, :], x_seg.detach().cpu().numpy()[0, 0, :, :, :], labels))/len(labels)
                    test_dsc.update(dsc_trans.item(), Y.size(0))
                    test_raw_dsc.update(dsc_raw.item(), Y.size(0))


                print("--------------------------------------------------------------------------------------------")
                print('sim:{:.4f},IC:{:.4f}, smooth:{:.4f}'.format(np.mean(Loss_sim), np.mean(Loss_IC), np.mean(Loss_smo)))
                print('Trans label dsc avg: {:.4f}, Trans label dsc std: {:.4f}'.format(test_dsc.avg, test_dsc.std))
                print('Raw label dsc avg: {:.4f}, Raw label dsc std: {:.4f}'.format(test_raw_dsc.avg, test_raw_dsc.std))
                print('dice loss avg:{:.4f}, dice loss std:{:.4f}'.format(test_loss.avg, test_loss.std))
                print('dis loss avg:{:.4f}, dis loss std:{:.4f}'.format(dis_Loss.avg, dis_Loss.std))
                print('Train dice loss:', np.mean(Loss_dice))


                modelname = model_dir + '/' + model_name + "_stagelvl3_" + str(epoch) + "_" + "{:.4f}".format(test_dsc.avg) + "_" + "{:.4g}".format(test_loss.avg) + "_" + str(dis_Loss.avg) + '.pth'


                torch.save(model.state_dict(), modelname)


if __name__ == "__main__":

    imgshape = (224, 224, 64)
    imgshape_4 = (224/4, 224/4, 64/4)
    imgshape_2 = (224/2, 224/2, 64/2)

    range_flow = 0.4
    train_rig()








