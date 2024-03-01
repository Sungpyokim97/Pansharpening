# import os
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from transform import Diff_transform
# import random
# import numpy as np

# def get_patch(lr, hr, patch_size, scale):
#     ih, iw, c = lr.shape

#     tp = patch_size
#     ip = patch_size // scale
#     ix = random.randrange(0, iw - ip + 1)
#     iy = random.randrange(0, ih - ip + 1)
#     (tx, ty) = (scale * ix, scale * iy)

#     out_lr = lr[:, iy:iy + ip, ix:ix + ip, :]
#     out_hr = hr[:, ty:ty + tp, tx:tx + tp, :]

#     return out_lr, out_hr

# def augment(seq_lr, seq_hr):
#     # random horizontal flip
#     if random.random() < 0.5:
#         seq_lr = seq_lr[:, :, ::-1, :]
#         seq_hr = seq_hr[:, :, ::-1, :]

#     # random vertical flip
#     if random.random() < 0.5:
#         seq_lr = seq_lr[:, ::-1, :, :]
#         seq_hr = seq_hr[:, ::-1, :, :]
    
#     #random rotate
#     rot = random.randint(0, 3)
#     seq_lr = np.rot90(seq_lr, rot, (1, 2))
#     seq_hr = np.rot90(seq_hr, rot, (1, 2))
    
#     return seq_lr, seq_hr

# class Diff_Train_Dataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.train_dir = os.path.join(data_dir, 'train')
#         self.hr_dir = os.path.join(self.train_dir, 'HR')
#         self.lr_dir = os.path.join(self.train_dir, 'LR')
#         self.transform = transform

#         self.hr_images = sorted(os.listdir(self.hr_dir))
#         self.lr_images = sorted(os.listdir(self.lr_dir))

#     def __len__(self):
#         return len(self.hr_images)

#     def __getitem__(self, idx):
#         hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
#         lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])

#         hr_image = Image.open(hr_image_path).convert('RGB')
#         lr_image = Image.open(lr_image_path).convert('RGB')

#         if self.transform:
#             hr_image = self.transform(hr_image)
#             lr_image = self.transform(lr_image)

#         return hr_image, lr_image
    
# class Diff_Val_Dataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.val_dir = os.path.join(data_dir, 'val')
#         self.hr_dir = os.path.join(self.val_dir, 'HR')
#         self.lr_dir = os.path.join(self.val_dir, 'LR')
#         self.transform = transform

#         self.hr_images = sorted(os.listdir(self.hr_dir))
#         self.lr_images = sorted(os.listdir(self.lr_dir))

#     def __len__(self):
#         return len(self.hr_images)

#     def __getitem__(self, idx):
#         hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
#         lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])

#         hr_image = Image.open(hr_image_path).convert('RGB')
#         lr_image = Image.open(lr_image_path).convert('RGB')

#         if self.transform:
#             hr_image = self.transform(hr_image)
#             lr_image = self.transform(lr_image)

#         return hr_image, lr_image
    
# def Diff_Dataloader(data_dir, transform, batch_size):
    
#     train_dataset = Diff_Train_Dataset(data_dir, transform=Diff_transform())
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_dataset = Diff_Val_Dataset(data_dir, transform=transform())
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
#     return train_dataloader, val_dataloader

#############################################################################################
def random_crop(ms, pan, mspan, crop_size):
    """
    이미지를 랜덤하게 크롭하는 함수

    :param image: PIL 이미지 객체
    :param crop_size: (너비, 높이) 형태의 튜플, 크롭할 이미지의 크기
    :return: 크롭된 이미지 객체
    """
    H, W = pan.shape
    crop_width, crop_height = crop_size

    if crop_width > W or crop_height > H:
        raise ValueError("크롭 크기가 원본 이미지 크기보다 클 수 없습니다.")

    left = random.randint(0, W - crop_width)
    bottom = random.randint(0, H - crop_height)
    right = left + crop_width
    top = bottom + crop_height

    return ms[bottom:top, left:right, :], pan[bottom:top, left:right], mspan[bottom:top, left:right],

# preprocess data in different dataset
def Preprocess(ms, pan, mspan):

    # ms를 pan size로 upscale
    H, W = pan.shape  # (128,128,3)

    up_ms = cv2.resize(ms, (int(H),int(W)), interpolation=cv2.INTER_CUBIC)

    ms, pan, mspan = random_crop(up_ms, pan, mspan, (256,256))
    # if C == 8:
    #     ms = ms[:, :, (4, 2, 1, 6)]
    # lrms = ms.resize((int(ms.shape[0]/4),int(ms.shape[1]/4)), Image.BICUBIC)

    scale = (2**11 - 1) # (2**8-1)
    ms = np.array(ms).transpose(2,0,1) / scale
    pan = np.expand_dims(np.array(pan), axis=0) / scale
    mspan = np.array(mspan).transpose(2,0,1) / scale
    return ms, pan, mspan

def get_files_in_subdirectories_train(root, dataset_path, subdirectory_name):

    img_path = os.path.join(root, dataset_path)
    # img_path = '/media/ksp/SH_HDD/Pan-Sharpening/WorldView3_Original_Train/'
    directories = [os.path.join(img_path, d) for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))]
    subdirectories = [os.path.join(d, subdirectory_name) for d in directories]
    
    files = [os.path.join(subdir, file) for subdir in subdirectories for file in os.listdir(subdir)]
    return files

# ex) region = ['AOI_4_Shanghai_Bldg_Test_public', 'AOI_4_Shanghai_Bldg_Test_public']
def get_files_in_subdirectories_test(root, dataset_path, region_list, subdirectory_name):

    img_path = os.path.join(root, dataset_path)
    # img_path = '/media/ksp/SH_HDD/Pan-Sharpening/WorldView3_Original_Test/'
    # directories = [os.path.join(img_path, d) for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))]
    directories = [os.path.join(img_path, d) for d in region_list]
    subdirectories = [os.path.join(d, subdirectory_name) for d in directories]
    
    files = [os.path.join(subdir, file) for subdir in subdirectories for file in os.listdir(subdir)]
    return files

# dataset
class TrainDataset(Dataset):
    
    def __init__(self, root, dataset_path):
        super(TrainDataset, self).__init__()
        self.root = root
        self.dataset_path = dataset_path
        self.ms_list = get_files_in_subdirectories_train(root, dataset_path, 'MUL')
        self.pan_list = get_files_in_subdirectories_train(root, dataset_path, 'PAN')
        self.mspan_list = get_files_in_subdirectories_train(root, dataset_path, 'MUL-PanSharpen')
        
    def __getitem__(self, index):
        # ms = Image.open(self.root+'/'+self.ms_path+'/'+self.ms_list[index])
        # pan = Image.open(self.root+'/'+self.pan_path+'/'+self.pan_list[index])
        ms = tiff.imread(self.ms_list[index])
        pan = tiff.imread(self.pan_list[index])
        mspan = tiff.imread(self.mspan_list[index])

        ms, pan, mspan = Preprocess(ms, pan, mspan)
        return ms, pan, mspan
    
    def __len__(self):
        return len(self.ms_list)
    
class TestDataset(Dataset):
    
    def __init__(self, root, test_dataset_path, region_list):
        super(TestDataset, self).__init__()
        self.root = root
        self.test_dataset_path = test_dataset_path
        self.region_list = region_list
        self.ms_list = get_files_in_subdirectories_test(root, test_dataset_path, region_list, 'MUL')
        self.pan_list = get_files_in_subdirectories_test(root, test_dataset_path, region_list, 'PAN')
        self.mspan_list = get_files_in_subdirectories_test(root, test_dataset_path, region_list, 'MUL-PanSharpen')
        
    def __getitem__(self, index):
        # ms = Image.open(self.root+'/'+self.ms_path+'/'+self.ms_list[index])
        # pan = Image.open(self.root+'/'+self.pan_path+'/'+self.pan_list[index])
        ms = tiff.imread(self.ms_list[index])
        pan = tiff.imread(self.pan_list[index])
        mspan = tiff.imread(self.mspan_list[index])

        ms, pan, mspan = Preprocess(ms, pan, mspan)
        return ms, pan, mspan
    
    def __len__(self):
        return len(self.ms_list)
    
# 업데이트 중인 부분 argument 추가될 가능성 다분함
def WV3_PS_Dataloader(root, batch_size):
    # train_dataset = DiffDataset(data_dir, mode='train',patch_size=patch_size, transform=Diff_transform)
    # val_dataset = DiffDataset(data_dir, mode='val',patch_size=patch_size, transform=Diff_transform)
    # test_dataset = DiffTestDataset(data_dir, mode='test',data=test_data, transform=None)
    train_dataset_path = 'WorldView3_Original_Train/'
    val_dataset_path = 'WorldView3_Original_Test/'
    test_dataset_path = 'WorldView3_Original_Test/'
    regions = ['AOI_4_Shanghai_Bldg_Test_public', 'AOI_4_Shanghai_Bldg_Test_public']

    train_dataset = TrainDataset(root, train_dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TestDataset(root, val_dataset_path, region_list= regions)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TestDataset(root, test_dataset_path, region_list= regions)
    test_loader = DataLoader(test_dataset_path, batch_size=batch_size, shuffle=True)    
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # return train_loader, val_loader, test_loader
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    import tifffile as tiff

    root = '/media/ksp/SH_HDD/Pan-Sharpening'
    train_dataset_path = 'WorldView3_Original_Train/'
    test_dataset_path = 'WorldView3_Original_Test/'
    regions = ['AOI_4_Shanghai_Bldg_Test_public', 'AOI_4_Shanghai_Bldg_Test_public']
    # ms_directories = get_files_in_subdirectories(root, dataset_path, 'MUL')
    # pan_directories = get_files_in_subdirectories(root, dataset_path, 'PAN')
    # mspan_directories = get_files_in_subdirectories(root, dataset_path, 'MUL-PanSharpen')
    # rgbpan_directories = get_files_in_subdirectories(root, dataset_path, 'RGB-PanSharpen')

    # print(len(ms_directories))
    # ms = tiff.imread(ms_directories[0])
    # print(f'mul size : {ms.shape}')
    # pan = tiff.imread(pan_directories[0])
    # print(f'pan size : {pan.shape}')
    # mspan = tiff.imread(mspan_directories[0])
    # print(f'mspan size : {mspan.shape}')
    # rgbpan = tiff.imread(rgbpan_directories[0])
    # print(f'rgbpan size : {rgbpan.shape}')
    
    # loader = WV3_PS_Dataloader(root,16)
    # for a,b,c in loader[0]:
    #     print(a.shape, b.shape, c.shape)
    train_dataset = TrainDataset(root, train_dataset_path)
    test_dataset = TestDataset(root, test_dataset_path, region_list=regions)
    print(len(train_dataset), len(test_dataset))