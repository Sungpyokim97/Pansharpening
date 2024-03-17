import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import os
from tqdm import tqdm
import configparser
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from PIL import Image
from copy import deepcopy

import numpy as np
from DiffIRS1 import DiffIRS1
from DiffIRS2 import DiffIRS2
from mspanmixer import Simplemixer, PGCU
from psnr import calculate_psnr
from torchvision.utils import save_image
# from dataloader import WV3_PS_Dataloader
from dataloader_h5 import WV3_PS_Dataloader
from evaluate import analysis_accu

# 모델, 손실 함수, 옵티마이저 초기화
def initialize_model(types, device, learning_rate, start_epoch, ckpt_dir, ckpt_path=None):
    
    if types == 'DiffIRS1':
        model = DiffIRS1().to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
            model = DataParallel(model)
    elif types == 'DiffIRS2':
        model = DiffIRS2().to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
            model = DataParallel(model)
    elif types == 'SimpleMixer':
        model = Simplemixer(ms_channels=8,pan_channels=1).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
            model = DataParallel(model)
    elif types == 'PGCU':
        model = PGCU(Channel=8, Vec=128).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
            model = DataParallel(model)

    #loss 추가 해야할수도>???
    criterion = torch.nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if start_epoch > 1:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{start_epoch-1}.pth')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'loading checkpoint from {start_epoch-1}')
    else:
        pass

    return model, criterion, optimizer, start_epoch

# Training 루프
def train_model(config_path, mode):
    
    config = Config(config_path, mode)
    
    # 업데이트 중
    train_loader, val_loader, test_loader = WV3_PS_Dataloader(config.root, config.batch_size)

    model, criterion, optimizer, start_epoch = initialize_model(config.types, config.device, config.learning_rate, config.start_epoch, os.path.join(config.ckpt_dir,config.types) , ckpt_path=None)

    writer = SummaryWriter(os.path.join(config.result_dir, 'tensorboard_logs'))

    if config.types == 'DiffIRS2':
        model_S1 = DiffIRS1().to(config.device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
            model_S1 = DataParallel(model_S1)
        checkpoint = torch.load(config.s1_weight_dir)
        model_S1.load_state_dict(checkpoint['model_state_dict'])
        # S2 G is same as S1 G
        model.module.G = deepcopy(model_S1.module.G)
        # S2 G freeze
        for name, param in model.module.named_parameters():
            if 'G.' in name:
                param.requires_grad = False
    best_psnr = 0.0  # 가장 높은 PSNR을 저장할 변수 초기화
    best_epoch = 0   # 가장 높은 PSNR을 갖는 에포크 번호 초기화

    for epoch in range(start_epoch-1, config.epochs):
        
        # learning schedule setting
        # if (epoch+1) % 30 == 0:
        #     optimizer.param_groups[0]['lr'] *= 0.1
        
        # lr = optimizer.param_groups[0]['lr']  
        T = config.epochs 
        lr_s = 0.1**4
        lr_e = 0.1**7
        lr = optimizer.param_groups[0]['lr'] = ((lr_s-lr_e)/2)*np.cos(np.pi/T*epoch) + (lr_s+lr_e)/2    
        print(f'learning late : {lr:.7f}')
        model.train()
        running_loss = 0.0
        running_diff_loss = 0.0
        running_recon_loss = 0.0
        for data_dict  in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch"):
            # inputs, labels = inputs.to(config.device), labels.to(config.device)
            ms, lms, pan, mspan = data_dict['ms'].float(), data_dict['lms'].float(), data_dict['pan'].float(), data_dict['gt'].float()
            ms, lms, pan, mspan = ms.to(config.device), lms.to(config.device), pan.to(config.device), mspan.to(config.device)
            
            #save_path = os.path.join(config.result_dir, f'{config.types}_result_epoch_{epoch}_input_img_.png')
            # save_path = os.path.join(config.result_dir, f'{config.types}_result_input_img_.png')
            # save_image(inputs[0].cpu(), save_path)
            # writer.add_image('Images/LQ', inputs[0].cpu(), epoch)

            #save_path = os.path.join(config.result_dir, f'{config.types}_result_epoch_{epoch}_label_img_.png')
            # save_path = os.path.join(config.result_dir, f'{config.types}_result_label_img_.png')
            # save_image(labels[0].cpu(), save_path)
            # writer.add_image('Images/GT', labels[0].cpu(), epoch)
            if config.types == 'DiffIRS1':
                outputs, _ = model(lms, pan, mspan)
            elif config.types == 'DiffIRS2':
                _, IPRS1 = model_S1(lms, pan, mspan)
                IPRS1 = IPRS1[0]
                outputs, pred_IPR_list = model(lms, pan, IPRS1=IPRS1)
                IPRS2 = pred_IPR_list[-1]
            else:
                outputs = model(pan, lms)
            # loss = criterion(outputs, labels)
            
            output_for_loss = outputs[0] if isinstance(outputs, tuple) else outputs
            # save_path = os.path.join(config.result_dir, f'{config.types}_result_epoch_{epoch}_out_img_.png')
            # save_path = os.path.join(config.result_dir, f'{config.types}_result_out_img_.png')
            # save_image(output_for_loss[0].cpu(), save_path)
            # writer.add_image('Images/HQ', output_for_loss[0].cpu(), epoch)
            if config.types == 'DiffIRS2':
                diff_loss = criterion(IPRS2, IPRS1)
                recon_loss = criterion(output_for_loss, mspan)
                R = 0.2
                loss = R* diff_loss + (1-R)* recon_loss
            else:
                loss = criterion(output_for_loss, mspan)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if config.types == 'DiffIRS2':
                running_diff_loss += diff_loss.item()
                running_recon_loss += recon_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_diff_loss = running_diff_loss / len(train_loader)
        avg_recon_loss = running_recon_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {avg_train_loss:.4f}, Diff Loss : {avg_diff_loss:.4f}, Rcon Loss : {avg_recon_loss:.4f}")

        # Validation loop
        model.eval()
        total_psnr = 0.0
        total_sam = 0.0
        total_ergas = 0.0
        total_cc = 0.0
        image_count = 0
        with torch.no_grad():
            for i, data_dict in enumerate(test_loader):
                ms, lms, pan, mspan = data_dict['ms'].float(), data_dict['lms'].float(), data_dict['pan'].float(), data_dict['gt'].float()
                ms, lms, pan, mspan = ms.to(config.device), lms.to(config.device), pan.to(config.device), mspan.to(config.device)
                
                if config.types == 'DiffIRS1':
                    outputs, _ = model(lms, pan, mspan)

                elif config.types == 'DiffIRS2':
                    # _, IPRS1 = model_S1(lms, pan, mspan)
                    outputs = model(lms, pan)
                else:
                    outputs = model(pan, lms)

                metric_dict = analysis_accu(mspan, outputs, ratio=4, dim_cut=0, choices = 5)
                total_psnr += metric_dict['PSNR']
                total_sam += metric_dict['SAM']
                total_ergas += metric_dict['ERGAS']
                total_cc += metric_dict['CC']
                
                # 이미지 저장 로직
                # save_path = os.path.join(config.result_dir, f'{config.types}_result_epoch_{epoch}_img_{i}.png')
                
                # save_image(outputs.cpu(), save_path)

                image_count += 1

        avg_psnr = total_psnr / image_count
        avg_sam = total_sam / image_count
        avg_ergas = total_ergas / image_count
        avg_cc = total_cc / image_count
        print(f"Epoch [{epoch+1}/{config.epochs}], Validation PSNR: {avg_psnr:.4f} SAM: {avg_sam:.4f} ERGAS: {avg_ergas:.4f} CC: {avg_cc:.4f}")
        
        writer.add_scalar('Learning_rate/Train', lr, epoch)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('PSNR/Validation', avg_psnr, epoch)
        writer.add_scalar('SAM/Validation', avg_sam, epoch)
        writer.add_scalar('ERGAS/Validation', avg_ergas, epoch)
        writer.add_scalar('CC/Validation', avg_cc, epoch)

        if not os.path.exists(os.path.join(config.ckpt_dir, config.types)):
            os.makedirs(os.path.join(config.ckpt_dir, config.types))

        if (epoch+1) % 10 == 0:
            # CKPT 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config.ckpt_dir, config.types, f'checkpoint_epoch_{epoch+1}.pth'))

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config.ckpt_dir, config.types, 'best_checkpoint.pth'))
            
    print(f"Best PSNR: {best_psnr:.4f} at epoch {best_epoch+1}")
    writer.close()
    
class Config:
    def __init__(self, config_path, mode):
        parser = configparser.ConfigParser()
        parser.read(config_path)
        if mode == 'train_S1':
            stage = 'train_S1'
        elif mode == 'train_S2':
            stage = 'train_S2'
        elif mode == 'test':
            pass
        else:
            print('Wrong mode')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = parser.get('common', 'root')
        self.data_dir = parser.get('common', 'data_dir')

        if mode =='train_S1' or mode =='train_S2':
            self.types = parser.get(stage,'types')
            self.ckpt_dir = parser.get(stage, 'ckpt_dir')
            self.result_dir = parser.get(stage, 'result_dir')
            self.batch_size = int(parser.get(stage, 'batch_size'))
            self.learning_rate = float(parser.get(stage, 'learning_rate'))
            self.epochs = int(parser.get(stage, 'epochs'))
            self.start_epoch = int(parser.get(stage, 'start_epoch'))

        self.s1_weight_dir = parser.get('train_S2', 's1_weight_dir')
        self.test_data = parser.get('test', 'test_data')
