import glob
import os
import sys
from argparse import ArgumentParser
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.utils.data as Data


from Functions import generate_grid, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from model import ED_lv1, \
    ED_lv2, ED_lv3, SpatialTransform_unit, \
    SpatialTransformNearest_unit, smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC, register_model

from Functions import train_data, AverageMeter, val_dataset



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
parser.add_argument("--iteration_lvl1", type=int,
                    dest="iteration_lvl1", default=16001,
                    help="number of lvl1 iterations")
parser.add_argument("--iteration_lvl2", type=int,
                    dest="iteration_lvl2", default=16001,
                    help="number of lvl2 iterations")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=40001,
                    help="number of lvl3 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=0.01,
                    help="Anti-fold loss: suggested range 0 to 1000") #0.01
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=1,
                    help="Gradient smooth loss: suggested range 0.1 to 10")# 0.1
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='./data/train/',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=4000,
                    help="Number step for freezing the previous level")
parser.add_argument("--val_dir", type=str,
                    dest="val_dir", default='./data/val/',
                    help="fixed image")

parser.add_argument("--model_dir", type=str,
                    dest="model_dir", default='./models',
                    help="directory to save/load models")
parser.add_argument("--shutdown", action='store_true',
                    dest="shutdown", default=False,
                    help="if set, shut down machine after training (disabled by default)")

opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath
freeze_step = opt.freeze_step
val_file = opt.val_dir

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3

model_name = "RER_"



def train_lvl1():
    print("Training lvl1...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ED_lv1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                        range_flow=range_flow).to(device)
    



    loss_similarity = NCC(win=3)
    loss_Jdet = neg_Jdet_loss
    loss_smooth = smoothloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # train data
    data = train_data(datapath)

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = os.path.join(opt.model_dir, 'Stage')

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    lossall = np.zeros((4, iteration_lvl1+1))


    training_generator = DataLoader(data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    step = 0
    load_model = False
    if load_model is True:
        model_path = os.path.join(opt.model_dir, 'Stage', 'LDR_stagelvl1_5000.pth')
        print("Loading weight: ", model_path)
        step = 5000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load(os.path.join(opt.model_dir, 'Stage', 'lossLDR_stagelvl1_5000.npy'))
        lossall[:, 0:5000] = temp_lossall[:, 0:5000]


    loss_all = AverageMeter()
    while step <= iteration_lvl1:
        for Y, X, Y_seg ,X_seg in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()
            Y_seg = Y_seg.float()
            X_seg = X_seg.float()

            F_X_Y, X_Y, Y_4x, _, _ = model(X, Y, X_seg, Y_seg)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)
            
            # 将形变场标准化
            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            loss_all.update(loss.item(), Y.numel())


            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl1_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl1_" + str(step) + '.npy', lossall)

            step += 1

            if step > iteration_lvl1:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl1.npy', lossall)


def train_lvl2():
    print("Training lvl2...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model_lvl1 = ED_lv1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                          range_flow=range_flow).to(device)

    model_path = sorted(glob.glob(os.path.join(opt.model_dir, 'Stage', model_name + "stagelvl1_?????.pth")))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)


    model = ED_lv2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    data = train_data(datapath)


    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = os.path.join(opt.model_dir, 'Stage')

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    lossall = np.zeros((4, iteration_lvl2 + 1))

    training_generator = DataLoader(data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    step = 0
    load_model = False
    if load_model is True:
        model_path = os.path.join(opt.model_dir, 'Stage', 'LDR_stagelvl2_5000.pth')
        print("Loading weight: ", model_path)
        step = 5000
        model.load_state_dict(torch.load(model_path))

    loss_all = AverageMeter()
    while step <= iteration_lvl2:
        for Y, X, Y_seg, X_seg in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()
            X_seg = X_seg.float()
            Y_seg = Y_seg.float()


            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y, X_seg, Y_seg)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            loss_all.update(loss.item(), Y.numel())
            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl2_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl2_" + str(step) + '.npy', lossall)

            if step == freeze_step:
                model.unfreeze_modellvl1()

            step += 1
            if step > iteration_lvl2:
                break

        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl2.npy', lossall)


def train_lvl3():
    print("Training lvl3...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    model_lvl1 = ED_lv1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).to(device)
           
    model_lvl2 = ED_lv2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    model_path = sorted(glob.glob(os.path.join(opt.model_dir, 'Stage', model_name + "stagelvl2_?????.pth")))[-1]
    model_lvl2.load_state_dict(torch.load(model_path,  map_location='cuda'))

    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False


    model = ED_lv3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2).to(device)
    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    reg_model = register_model((64, 224, 224), 'nearest')
    reg_model.cuda()

    transform = SpatialTransform_unit().to(device)
    transform_label = SpatialTransformNearest_unit().cuda()
    transform_label.eval()



    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    data = train_data(datapath)
    flag = 0

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = opt.model_dir

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    lossall = np.zeros((4, iteration_lvl3 + 1))

    training_generator = DataLoader(data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    DS_val, _ = val_dataset(val_file)
    test_loader = DataLoader(DS_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
      
    step = 0
    load_model = False
    if load_model is True:
        model_path = os.path.join(opt.model_dir, 'LDR_stagelvl3_1200.pth')
        print("Loading weight: ", model_path)
        step = 1000
        model.load_state_dict(torch.load(model_path, map_location='cuda'))

    best_dsc = 0
    loss_all = AverageMeter()
    while step <= iteration_lvl3:
        for Y, X, Y_seg, X_seg in training_generator:

            X = X.to(device).float()
            Y = Y.to(device).float()
            X_seg = X_seg.float()
            Y_seg = Y_seg.float()

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            F_X_Y, X_Y, Y_4x, _, _, _, _ = model(X, Y, X_seg, Y_seg)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()
            loss_all.update(loss.item(), Y.numel())

            # with lr 1e-3 + with bia
            if step % 100 == 0 :
                test_dsc = AverageMeter()
                test_raw_dsc = AverageMeter()
                    
                with torch.no_grad():
                    model.eval()
                    print("\nValidation")
                    for fixed_img, moving_img, y_seg, x_seg, y_labels in test_loader:
                        fixed_img = fixed_img.to(device).float() 
                        moving_img = moving_img.to(device).float() 
                        x_seg = x_seg.float()  # bone_moving
                        y_seg = y_seg.float() 
                        y_labels =  y_labels.float() 
                        labels = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0]  # bone labels
                        F_X_Y, X_Y, Y_4x, _, _, _, _ = model(moving_img, fixed_img, x_seg, y_seg)
                        x_seg = x_seg.to(device)
                        y_labels = y_labels.to(device)

                        x_y_seg = transform_label(x_seg, F_X_Y.permute(0, 2, 3, 4, 1), grid)

                        dsc_trans = np.sum(dice(y_labels.detach().cpu().numpy()[0, 0, :, :, :], x_y_seg.detach().cpu().numpy()[0, 0, :, :, :], labels))/len(labels)
                        dsc_raw = np.sum(dice(y_labels.detach().cpu().numpy()[0, 0, :, :, :], x_seg.detach().cpu().numpy()[0, 0, :, :, :], labels))/len(labels)
                        test_dsc.update(dsc_trans.item(), Y.size(0))
                        test_raw_dsc.update(dsc_raw.item(), Y.size(0))

                    print("--------------------------------------------------------------------------------------------")
                    print('Trans label dsc avg: {:.4f}, Trans label dsc std: {:.4f}'.format(test_dsc.avg, test_dsc.std))
                    print('Raw label dsc avg: {:.4f}, Raw label dsc std: {:.4f}'.format(test_raw_dsc.avg, test_raw_dsc.std))


                    if flag < test_dsc.avg:
                        modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                        torch.save(model.state_dict(), modelname)
                        flag = test_dsc.avg
                        
            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break

        print("one epoch pass")



imgshape = (224, 224, 64)
imgshape_4 = (224/4, 224/4, 64/4)
imgshape_2 = (224/2, 224/2, 64/2)

range_flow = 0.4
train_lvl1()
train_lvl2()
train_lvl3()
os.system("/usr/bin/shutdown")




