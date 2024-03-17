import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
import os
# import datetime
# import imageio
import torch.nn.functional as F
from scipy import io as sio
from torch.utils.data import Dataset, DataLoader
# from udl_vis.Basis.postprocess import showimage8
import matplotlib.pyplot as plt
from dataloader_h5 import Dataset_Pro, MultiExmTest_h5
from PIL import Image
from postprocess import showimage8

root = '/hdd/sungpyo/Pan-Sharpening/Pancollection'
wv3_train = 'training_data/train_wv3.h5'
qb_train = 'training_data/train_qb.h5'
gf2_train = 'training_data/train_gf2.h5'

wv3_fr_test = 'test_data/test_wv3_OrigScale_multiExm1.h5'
qb_fr_test = 'test_data/test_qb_OrigScale_multiExm1.h5'
gf2_fr_test = 'test_data/test_gf2_OrigScale_multiExm1.h5'
wv2_fr_test = 'test_data/test_wv2_OrigScale_multiExm1.h5'

wv3_rr_test = 'test_data/test_wv3_multiExm1.h5'
qb_rr_test = 'test_data/test_qb_multiExm1.h5'
gf2_rr_test = 'test_data/test_gf2_multiExm1.h5'
wv2_rr_test = 'test_data/test_wv2_multiExm1.h5'

def cv2rgb_test(root, test_dataset_name ,dataset = 'wv3'):
    if dataset == 'gf2':
        img_scale = 2**10-1
    else:
        img_scale = 2**11-1
    test_dataset_path = os.path.join(root, test_dataset_name)
    test_dataset = MultiExmTest_h5(test_dataset_path, test_dataset_name, img_scale)
    directory_name = test_dataset_name.split('/')[0]
    # 파일 이름만 분리 (확장자 포함)
    file_name_with_extension = os.path.basename(test_dataset_path)
    # 확장자 제거하고 파일 이름만 추출
    file_name = os.path.splitext(file_name_with_extension)[0]

    for type in ['ms', 'lms', 'gt', 'pan']:
        for i in range(len(test_dataset)):
            if type == 'gt':
                output = test_dataset.gt[i]
            elif type == 'lms':
                output = test_dataset.lms[i]
            elif type == 'ms':
                output = test_dataset.ms[i]
            elif type == 'pan':
                output = test_dataset.pan[i]

            print(output.shape)
            
            output = showimage8(output, unnormlize = img_scale, first_channel=True)
            output =  torch.Tensor(output.copy())   #BGR
            output = output.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            print(output.shape)
            im  = Image.fromarray(output)
            png_path = "/hdd/sungpyo/Pan-Sharpening/Pancollection/pngformat/"
            if 'OrigScale' in test_dataset_name:
                resolution = 'Full'
            else:
                resolution = 'Reduced'
            save_dir = os.path.join(png_path,directory_name,dataset,resolution,type)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir,f'{file_name}_{i}.png')
            im.save(save_path)

def cv2rgb_train(root, train_dataset_name ,dataset = 'wv3'):
    if dataset == 'gf2':
        img_scale = 2**10-1
    else:
        img_scale = 2**11-1
    train_dataset_path = os.path.join(root, train_dataset_name)
    train_dataset = Dataset_Pro(train_dataset_path, img_scale)
    directory_name = train_dataset_name.split('/')[0]
    # 파일 이름만 분리 (확장자 포함)
    file_name_with_extension = os.path.basename(train_dataset_path)
    # 확장자 제거하고 파일 이름만 추출
    file_name = os.path.splitext(file_name_with_extension)[0]

    for type in ['ms', 'lms', 'gt', 'pan']:
        for i in range(len(train_dataset)):
            if type == 'gt':
                output = train_dataset.gt[i]
            elif type == 'lms':
                output = train_dataset.lms[i]
            elif type == 'ms':
                output = train_dataset.ms[i]
            elif type == 'pan':
                output = train_dataset.pan[i]

            print(output.shape)

            output = showimage8(output, unnormlize = img_scale, first_channel=True)
            output =  torch.Tensor(output.copy())   #BGR
            output = output.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            print(output.shape)
            im  = Image.fromarray(output)
            png_path = "/hdd/sungpyo/Pan-Sharpening/Pancollection/pngformat/"
            save_dir = os.path.join(png_path,directory_name,dataset,type)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir,f'{file_name}_{i}.png')
            im.save(save_path)

def rgb_save(output, img_scale, save_dir):
    output = showimage8(output, unnormlize = img_scale, first_channel=True)
    output =  torch.Tensor(output.copy())   #BGR
    output = output.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    im  = Image.fromarray(output)
    save_dir = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,f'test_img.png')
    im.save(save_path)

if __name__ == '__main__':
    from PIL import Image
    file_path = "/hdd/sungpyo/Pan-Sharpening/Pancollection/test_data/test_qb_OrigScale_multiExm1.h5"
    img_scale = 2**11-1
    test_dataset_name = 'test_data/test_gf2_multiExm1.h5'
    test_dataset_path = f'/hdd/sungpyo/Pan-Sharpening/Pancollection/{test_dataset_name}'
    dataset = MultiExmTest_h5(test_dataset_path, test_dataset_name, img_scale)
    output = dataset.gt[15]
    print(output.shape)

    test_file_path = "/hdd/sungpyo/Pan-Sharpening/Pancollection/test_data/test_qb_OrigScale_multiExm1.h5"
    dataset_name = "test_gf2_OrigScale_multiExm1.h5"
    exam_dataset = MultiExmTest_h5(test_file_path, dataset_name, img_scale, suffix='.h5')
    print(exam_dataset)

    root = '/hdd/sungpyo/Pan-Sharpening/Pancollection'
    wv3_train = 'training_data/train_wv3.h5'
    qb_train = 'training_data/train_qb.h5'
    gf2_train = 'training_data/train_gf2.h5'

    wv3_fr_test = 'test_data/test_wv3_OrigScale_multiExm1.h5'
    qb_fr_test = 'test_data/test_qb_OrigScale_multiExm1.h5'
    gf2_fr_test = 'test_data/test_gf2_OrigScale_multiExm1.h5'
    wv2_fr_test = 'test_data/test_wv2_OrigScale_multiExm1.h5'

    wv3_rr_test = 'test_data/test_wv3_multiExm1.h5'
    qb_rr_test = 'test_data/test_qb_multiExm1.h5'
    gf2_rr_test = 'test_data/test_gf2_multiExm1.h5'
    wv2_rr_test = 'test_data/test_wv2_multiExm1.h5'

    # for name in ['wv3','gf2','qb']:
    #     cv2rgb_train(root, f'training_data/train_{name}.h5',dataset = name)
    name = 'qb'
    cv2rgb_train(root, f'training_data/train_{name}.h5',dataset = name)
    for name in ['wv3','wv2','gf2','qb']:
        cv2rgb_test(root, f'test_data/test_{name}_OrigScale_multiExm1.h5',dataset = name)
        cv2rgb_test(root, f'test_data/test_{name}_multiExm1.h5',dataset = name)
        