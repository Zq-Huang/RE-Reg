import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import numpy as np
import torch
import nibabel as nib
import time 
import pandas as pd

from Functions import generate_grid_unit , val_dataset, transform_unit_flow_to_flow_cuda
from model import ED_lv1, ED_lv2, ED_lv3,\
     SpatialTransform_unit, register_model, SpatialTransformNearest_unit
from rigid_dice import *
from rigid_filed import  extract_class_masks
from Evaluation_indicators import SDlogDetJac, HausdorffDistanceMetric  
from DC import distances_loss

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='',
                    help="Pre-trained model path (leave empty to skip loading)")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='./Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--test_dir", type=str,
                    dest="test_dir", default='./data/val/',
                    help="directory with validation/test data")
parser.add_argument("--model_dir", type=str,
                    dest="model_dir", default='./models',
                    help="directory for models (used if modelpath is relative)")

opt = parser.parse_args()

savepath = opt.savepath
test_dir = opt.test_dir

if not os.path.isdir(savepath):
    os.mkdir(savepath)

start_channel = opt.start_channel

def BRCS(dice, delta, scale = 1):
    BRCS = dice * np.exp(-delta/scale)*100
    return BRCS


def test():
    imgshape_4 = (224 / 4, 224 / 4, 64 / 4)
    imgshape_2 = (224 / 2, 224 / 2, 64 / 2)


    model_lvl1 = ED_lv1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).cuda()
           
    model_lvl2 = ED_lv2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).cuda()
    

    model = ED_lv3(2, 3, start_channel, is_train=False, imgshape = imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    transform_label = SpatialTransformNearest_unit().cuda()
    transform = SpatialTransform_unit().cuda()

    if opt.modelpath:
        model.load_state_dict(torch.load(opt.modelpath, map_location='cuda'))
    else:
        print("No model path provided; running without loading pretrained weights.")
    model.eval()
    transform.eval()
    transform_label.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    reg_model = register_model((224, 224, 64), 'nearest')
    reg_model.cuda()
    rigid_dice_loss = RigidDiceLoss2(image_size = (224, 224, 64), device='cuda')
    rigid_dice_tran = RigidDiceLoss()
    compute_jacdet = SDlogDetJac()
    compute_haus95_dist = HausdorffDistanceMetric(percentile=95,
                                                  include_background=True)

    DS_val, files = val_dataset(test_dir)
    test_loader = DataLoader(DS_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    labels = [1.0,2.0,3.0,4.0]
    average_meters = initialize_average_meters()     
    with torch.no_grad():
        idx = 0
        Time = []
        for fixed_img, moving_img, y_seg, x_seg, y_labels in test_loader:
            moving_img = moving_img.cuda().float()
            fixed_img = fixed_img.cuda().float()
            x_seg = x_seg.float()
            y_seg = y_seg.float()

            folder = files[idx]
            affine = nib.load(files[idx]).affine
            folder_path = os.path.dirname(folder)
            t = time.time()
            F_X_Y = model(moving_img, fixed_img, x_seg, y_seg)
            times = time.time() - t
            Time.append(times)
            X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
            x_seg = x_seg.cuda()
            y_labels = y_labels.cuda()


            x_y_seg = transform_label(x_seg, F_X_Y.permute(0, 2, 3, 4, 1), grid)
 
            x_def = nib.Nifti1Image(X_Y, affine=affine)
            nib.save(x_def, folder_path+'/result.nii.gz')

            y_source_list = extract_class_masks(x_y_seg)
            sorce_list = extract_class_masks(x_seg)
            target_list = extract_class_masks(y_labels)
            y_source_lables = torch.cat(list(y_source_list.values()) , dim=1)
            target_labels = torch.cat(list(target_list.values()), dim = 1)
            rigid_penalty_LOSS = 0
            rigid_dice = 0
            for cls, _ in sorce_list.items():
                y_source_oh = y_source_list[cls]
                source_oh = sorce_list[cls]
                rigid_penalty = rigid_dice_loss(y_source_oh, source_oh, F_X_Y)
                rigid_penalty_LOSS = rigid_penalty_LOSS + rigid_penalty
            average_meters["test_loss"].update((rigid_penalty_LOSS.detach().cpu())/4, x_seg.size(0))

            # SDlog
            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            sdlog_jacdet, non_pos_jacdet = compute_jacdet(F_X_Y_norm.permute(0,4,1,2,3).detach().cpu().numpy())
            average_meters['jacobi'].update(non_pos_jacdet*100, 1)
            average_meters['SDlog'].update(sdlog_jacdet, 1)

            # h95
            fwd_haus95_dist = compute_haus95_dist(y_source_lables, target_labels)
            average_meters['h95'].update(fwd_haus95_dist.detach().cpu().numpy().mean(), 1)

            # rigid loss
            delta = []
            with torch.enable_grad():
                for cls, _ in sorce_list.items():
                    y_source_oh = y_source_list[cls]
                    source_oh = sorce_list[cls]
                    rigid_dice_penalty = rigid_dice_tran(y_source_oh, source_oh, F_X_Y)
                    delta.append(rigid_dice_penalty.detach().cpu().numpy())
                    rigid_dice += rigid_dice_penalty
            average_meters["test_dice"].update((rigid_dice.detach().cpu())/4, x_seg.size(0))

            # distance loss(DRS)
            DRS = 0
            for cls, _ in sorce_list.items():
                source_oh = sorce_list[cls]
                DC_loss = distances_loss(source_oh, F_X_Y_norm.permute(0,4,1,2,3), 20)
                DRS = DRS + DC_loss
                average_meters["DRS"].update((DRS.detach().cpu())/4, x_seg.size(0))  


            tran = dice(y_labels.detach().cpu().numpy()[0, 0, :, :, :], x_y_seg.detach().cpu().numpy()[0, 0, :, :, :], labels)
            raw = dice(y_labels.detach().cpu().numpy()[0, 0, :, :, :], x_seg.detach().cpu().numpy()[0, 0, :, :, :], labels)
            dsc_trans = np.sum(dice(y_labels.detach().cpu().numpy()[0, 0, :, :, :], x_y_seg.detach().cpu().numpy()[0, 0, :, :, :], labels))/len(labels)
            dsc_raw = np.sum(dice(y_labels.detach().cpu().numpy()[0, 0, :, :, :], x_seg.detach().cpu().numpy()[0, 0, :, :, :], labels))/len(labels)
            average_meters["test_dsc"].update(dsc_trans, x_seg.size(0))
            average_meters["test_raw_dsc"].update(dsc_raw, x_seg.size(0))


            for i in range(4):
                brcs = BRCS(tran[i], delta[i], 1.0)
                average_meters["BRCS"].update(brcs)

            idx += 1

            for a in range(0, 4):
                print('idx:{:.1f}, Trans label dsc : {:.4f}, Raw label dsc : {:.4f}'.format(a, tran[a], raw[a]))
                if a == 0:
                    average_meters["test_raw_dsc1"].update(raw[a].item(), x_seg.size(0))
                    average_meters["test_dsc1"].update(tran[a].item(), x_seg.size(0))
                if a == 1:
                    average_meters["test_raw_dsc2"].update(raw[a].item(), x_seg.size(0))
                    average_meters["test_dsc2"].update(tran[a].item(), x_seg.size(0))
                if a == 2:
                    average_meters["test_raw_dsc3"].update(raw[a].item(), x_seg.size(0))
                    average_meters["test_dsc3"].update(tran[a].item(), x_seg.size(0))
                if a == 3:
                    average_meters["test_raw_dsc4"].update(raw[a].item(), x_seg.size(0))
                    average_meters["test_dsc4"].update(tran[a].item(), x_seg.size(0))

            
                
                    
            print('Trans label avg: {:.4f}, Trans label dsc std: {:.4f}'.format(dsc_trans, dsc_raw))

            

        idx += 1

    print("--------------------------------------------------------------------------------------------")
    print('Trans label dsc avg: {:.4f}, Trans label dsc std: {:.4f}'.format(average_meters['test_dsc'].avg, average_meters['test_dsc'].std))
    print('Raw label dsc avg: {:.4f}, Raw label dsc std: {:.4f}'.format(average_meters['test_raw_dsc'].avg, average_meters['test_raw_dsc'].std)) 
    print("-----------------------------------------------------------------------------------------------")
    print("bone 1")
    print('Trans label dsc avg1: {:.4f}, Trans label dsc std: {:.4f}'.format(average_meters['test_dsc1'].avg, average_meters['test_dsc1'].std))
    print('Raw label dsc avg1: {:.4f}, Raw label dsc std: {:.4f}'.format(average_meters['test_raw_dsc1'].avg, average_meters['test_raw_dsc1'].std))
    print("-----------------------------------------------------------------------------------------------")
    print("bone 2")
    print('Trans label dsc avg2: {:.4f}, Trans label dsc std: {:.4f}'.format(average_meters['test_dsc2'].avg, average_meters['test_dsc2'].std))
    print('Raw label dsc avg2: {:.4f}, Raw label dsc std: {:.4f}'.format(average_meters['test_raw_dsc2'].avg, average_meters['test_raw_dsc2'].std))
    print("-----------------------------------------------------------------------------------------------")
    print("bone 3")
    print('Trans label dsc avg3: {:.4f}, Trans label dsc std: {:.4f}'.format(average_meters['test_dsc3'].avg, average_meters['test_dsc3'].std))
    print('Raw label dsc avg3: {:.4f}, Raw label dsc std: {:.4f}'.format(average_meters['test_raw_dsc3'].avg, average_meters['test_raw_dsc3'].std))
    print("-----------------------------------------------------------------------------------------------")
    print("bone 4")
    print('Trans label dsc avg4: {:.4f}, Trans label dsc std: {:.4f}'.format(average_meters['test_dsc4'].avg, average_meters['test_dsc4'].std))
    print('Raw label dsc avg4: {:.4f}, Raw label dsc std: {:.4f}'.format(average_meters['test_raw_dsc4'].avg, average_meters['test_raw_dsc4'].std))

    print("-----------------------------------------------------------------------------------------------")
    print("                                  Evaluation indicators                                        ")
    print("-----------------------------------------------------------------------------------------------")

    print('dice loss avg:{:.4f}, dice loss std:{:.4f}'.format(average_meters['test_loss'].avg, average_meters['test_loss'].std))
    print('dice loss2 avg:{:.4f}, dice loss2 std:{:.4f}'.format(average_meters['test_dice'].avg, average_meters['test_dice'].std))
    print('jacobi avg:{:.4f}, jacobi std:{:.4f}'.format(average_meters['jacobi'].avg, average_meters['jacobi'].std))
    print('SDlog avg:{:.4f}, SDlog std:{:.4f}'.format(average_meters['SDlog'].avg, average_meters['SDlog'].std))
    print('h95 avg:{:.4f}, h95 std:{:.4f}'. format(average_meters['h95'].avg, average_meters['h95'].std))
    print('DRS avg:{:.4f}, DRS std:{:.4f}'. format(average_meters['DRS'].avg, average_meters['DRS'].std))
    print('BRAS avg:{:.2f}, BRAS std:{:.2f}'. format(average_meters['BRCS'].avg, average_meters['BRCS'].std))
    print('Time avg:{:.4f}'.format(np.average(Time)))
    print("Finished")


    data = {
            "test_dsc1": average_meters['test_dsc1'].vals,
            # "raw_dsc1": average_meters['test_raw_dsc1'].vals,
            "test_dsc2": average_meters['test_dsc2'].vals,
            # "raw_dsc2": average_meters['test_raw_dsc2'].vals,
            "test_dsc3": average_meters['test_dsc3'].vals,
            # "raw_dsc3": average_meters['test_raw_dsc3'].vals,
            "test_dsc4": average_meters['test_dsc4'].vals,
            # "raw_dsc4": average_meters['test_raw_dsc4'].vals,
    }
    df = pd.DataFrame(data)
    df.to_excel("output_rig.xlsx", index=False)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def initialize_average_meters():
    meters = {
        "test_dsc": AverageMeter(),
        "test_raw_dsc": AverageMeter(),
        "test_dsc1": AverageMeter(),
        "test_raw_dsc1": AverageMeter(),
        "test_dsc2": AverageMeter(),
        "test_raw_dsc2": AverageMeter(),
        "test_dsc3": AverageMeter(),
        "test_raw_dsc3": AverageMeter(),
        "test_dsc4": AverageMeter(),
        "test_raw_dsc4": AverageMeter(),
        "test_loss": AverageMeter(),
        "test_dice": AverageMeter(),
        "SDlog": AverageMeter(),
        "jacobi": AverageMeter(),
        "h95": AverageMeter(),
        "DRS": AverageMeter(),
        "BRCS": AverageMeter()          
    }
    return meters

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

if __name__ == '__main__':
    imgshape = (224, 224, 64)
    range_flow = 0.4
    test()



