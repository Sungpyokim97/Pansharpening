import torch
from train import train_model
from test import test_model
# from test import test_model

train_S1_settings = '/home/ksp/Pansharpening/graduation/S1_settings.cfg'
train_S2_settings = '/home/ksp/Desktop/EDSR_CS_diff/S2_settings.cfg'
    
def main():
    # mode = 'test'
    mode = 'train_S1'
    # mode = 'train_S2'
    config = train_S1_settings

    if mode == 'train_S1':
        train_model(config, mode)
    elif mode == 'train_S2':
        # config = train_S2_settings
        train_model(config, mode)
    elif mode == 'test':
        # config = train_S2_settings
        test_model(config)
        
if __name__ == "__main__":
    main()