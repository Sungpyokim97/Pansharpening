import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import os
from tqdm import tqdm
import configparser
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from DiffIRS1 import DiffIRS1
from DiffIRS2 import DiffIRS2
from psnr import calculate_psnr, calculate_ssim
from torchvision.utils import save_image
from dataloader import Diff_Dataloader
from torch.nn import DataParallel


# 모델, 손실 함수, 옵티마이저 초기화
def initialize_model(types, device, learning_rate, start_epoch, ckpt_dir, weight_dir, ckpt_path=None):
    
    if types == 'DiffIRS1':
        model = DiffIRS1().to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
            model = DataParallel(model)
    elif types == 'DiffIRS2':
        # stage2에 들어오기 전 stage1을 거치기 위함
        pre_model = DiffIRS1().to(device)
        model = DiffIRS2().to(device)

        # stage1에 내가 원하는 weight를 씌워서 CPEN 작동시킬 준비
        if weight_dir:
            checkpoint = torch.load(weight_dir)
            # S1
            pre_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            state_dict = checkpoint['model_state_dict']
            # 'module.G'에 해당하는 키들을 필터링
            new_state_dict = {}
            for (key,val) in state_dict.items():
                if key.startswith('module.G'):
                    key = key.replace('module.G.', '')
                    new_state_dict[key]= val

            model.G.load_state_dict(new_state_dict, strict=False)
            
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
            pre_model = DataParallel(pre_model)
            model = DataParallel(model)

    criterion = torch.nn.L1Loss()
    params = [
        {'params': model.module.diffusion.parameters(), 'lr': learning_rate},  # 일반 학습률 적용
        {'params': model.module.G.parameters(), 'lr': learning_rate*0.01}
    ]
    optimizer = Adam(params)
    if start_epoch > 1:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_95.pth')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'loading checkpoint from {start_epoch-1}')
    else:
        pass

    if types == 'DiffIRS1':
        return model, criterion, optimizer, start_epoch
    elif types == 'DiffIRS2':
        return pre_model, model, criterion, optimizer, start_epoch

def test_model(config_path):
        # Validation loop
    config = Config(config_path)
    
    train_loader, val_loader, test_loader = Diff_Dataloader(config.data_dir, config.batch_size, config.test_data)

    if config.types == 'DiffIRS1':    
        model, criterion, optimizer, start_epoch = initialize_model(config.types, config.device, config.learning_rate, config.start_epoch, config.ckpt_dir, ckpt_path=None)
    elif config.types == 'DiffIRS2':
        pre_model, model, criterion, optimizer, start_epoch = initialize_model(config.types, config.device, config.learning_rate, config.start_epoch, config.ckpt_dir, config.weight_dir,ckpt_path=None)

    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    image_count = 0
    with torch.no_grad():
        for i, (labels, inputs) in enumerate(test_loader):
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            outputs = model(inputs)
            # print(labels.shape, outputs.shape, inputs.shape)
            total_ssim += calculate_ssim(labels, outputs)
            print(calculate_psnr(labels, outputs))
            total_psnr += calculate_psnr(labels, outputs)
            
            # 이미지 저장 로직
            save_path = os.path.join(config.result_dir, f'{config.types}_{config.test_data}_img_{i}.png')
            save_image(outputs.cpu(), save_path)

            image_count += 1

    avg_psnr = total_psnr / image_count
    avg_ssim = total_ssim / image_count
    print(f'PSNR : {avg_psnr}, SSIM : {avg_ssim}')

class Config:
    def __init__(self, config_path):
        parser = configparser.ConfigParser()
        parser.read(config_path)
        
        self.types = parser.get('training','types')
        self.data_dir = parser.get('training', 'data_dir')
        self.ckpt_dir = parser.get('training', 'ckpt_dir')
        self.result_dir = parser.get('training', 'result_dir')
        self.batch_size = int(parser.get('training', 'batch_size'))
        self.image_size = int(parser.get('training', 'image_size'))
        self.learning_rate = float(parser.get('training', 'learning_rate'))
        self.epochs = int(parser.get('training', 'epochs'))
        self.test_data = parser.get('training', 'test_data')
        self.start_epoch = int(parser.get('training', 'start_epoch'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dir = parser.get('training', 'weight_dir')

