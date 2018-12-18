import os,time,scipy.io

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from model import SeeInDark
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def pack_raw(img):
    # out = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
    #NORMALIZE out
    out = np.float32(img) / 255
    #out /= 255
    return out


def enhance(rgb_image, model):
    # model = SeeInDark()
    # model.load_state_dict(torch.load("/home/mihir/Downloads/CARLA_0.8.2/PythonClient/models/learning_to_see/checkpoint_sony_e3300.pth" ,map_location={'cuda:1':'cuda:0'}))
    # model = model.to(device)

    # if not os.path.isdir(result_dir):
    #     os.makedirs(result_dir)

    # for test_id in test_ids:
    #     #test the first image in each sequence
    #     in_files = glob.glob(input_dir + '%05d_00*.ARW'%test_id)
    #     for k in range(len(in_files)):

            
    ratio = 300
    input_full = np.expand_dims(pack_raw(rgb_image),axis=0) #*ratio
    input_full = np.minimum(input_full,1.0)

    in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
    out_img = model(in_img)
    output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
    output = np.minimum(np.maximum(output,0),1)

    output = output[0,:,:,:]
    
    # scipy.misc.toimage(origin_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_ori.png'%(test_id,ratio))
    scipy.misc.toimage(output*255,  high=255, low=0, cmin=0, cmax=255).save('%5d_00_%d_out.png'%(1,1))
    # scipy.misc.toimage(scale_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_scale.png'%(test_id,ratio))
    # scipy.misc.toimage(gt_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_gt.png'%(test_id,ratio))

    return output
