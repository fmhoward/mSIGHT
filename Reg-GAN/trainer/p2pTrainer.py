# import argparse
# import itertools
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import get_datasets_comp, get_datasets, get_val_dataset, tmachannels
from Model.CycleGan import *
from .utils import Resize, Normalize, ToTensor, smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
from glob import glob
import json
import cv2
import os
from collections import OrderedDict

class P2p_Trainer():
    def __init__(self, config):
        self.config = config
        ## def networks
        self.netG_A2B = nn.DataParallel(Generator(config['input_nc'], config['output_nc']).cuda())
        self.netD_B = nn.DataParallel(Discriminator(config['input_nc'] + config['output_nc']).cuda())
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
                  
        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        trans = [Normalize(config['output_nc']), ToTensor()]
        train, val = get_datasets_comp(
                        config['tilesdir'], 
                        config['valdir'],
                        config['input_nc'], config['output_nc'], config['size'], 
                        trans=trans, val_trans=trans,
                        )


        print(f"train {len(train)} samples, val {len(val)} samples")

        self.dataloader = DataLoader(
            train,
            batch_size=config['batchSize'], 
            shuffle=True, 
            num_workers=config['n_cpu'],
            drop_last=True)
        
        self.val_data = DataLoader(
            val,
            batch_size=config['batchSize'], 
            shuffle=False, 
            num_workers=config['n_cpu'],
            drop_last=True)
 
        # Loss plot
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))      
    
    def load_weights(self, epoch=None):
        '''
        Load weights if saved from previous training
        '''
        
        def get_state_d(weight_path):
            state = torch.load(weight_path)
            if torch.cuda.device_count() > 1:
                new_state = OrderedDict()
                for k, v in state.items():
                    new_state['module.' + k ] = v
                return new_state
            else:
                return state
        
        if not epoch:
            epoch = self.config['weight_epoch']
        
        print(f'saved model directory: {self.config["model_save"]}')
            
        weight_path = self.config['model_save'] + f'netG_A2B_epoch{epoch}.pth'
        if os.path.exists(weight_path):
            print(f"loading weights A2B from epoch {epoch}")
            self.netG_A2B.load_state_dict(get_state_d(weight_path))
            self.netG_A2B = nn.DataParallel(self.netG_A2B.cuda())

        weight_path_d = self.config['model_save'] + f'netD_B_epoch{epoch}.pth'
        if os.path.exists(weight_path_d):
            print(f"loading weights discriminator from epoch {epoch}")
            self.netD_B.load_state_dict(get_state_d(weight_path_d))
            self.netD_B = nn.DataParallel(self.netD_B.cuda())
    
    def train(self):        
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
               
                self.optimizer_G.zero_grad()
                fake_B = self.netG_A2B(real_A)
                loss_L1 = self.L1_loss(fake_B, real_B) * self.config['P2P_lamda']
                # gan loss: 
                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = self.netD_B(fake_AB)
                loss_GAN_A2B = self.MSE_loss(pred_fake, self.target_real) * self.config['Adv_lamda']

                # Total loss
                toal_loss = loss_L1 + loss_GAN_A2B
                toal_loss.backward()
                self.optimizer_G.step()
        

                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)
                pred_fake0 = self.netD_B(torch.cat((real_A, fake_B), 1)) * self.config['Adv_lamda']
                pred_real = self.netD_B(torch.cat((real_A, real_B), 1)) * self.config['Adv_lamda']
                loss_D_B = self.MSE_loss(pred_fake0, self.target_fake)+self.MSE_loss(pred_real, self.target_real)


                loss_D_B.backward()
                self.optimizer_D_B.step()
                self.logger.log({'loss_D_B': loss_D_B,'loss_G':toal_loss,},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B,})#,'SR':SysRegist_A2B

                
            # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            
            #############val###############
            with torch.no_grad():
                MAE = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B,real_B)
                    MAE += mae
                    num += 1


                    if i % int(100 // self.config['batchSize']) == 0:
                        if not os.path.exists(self.config["image_save"]):
                            os.makedirs(self.config["image_save"])

                        serial = os.path.basename(self.val_data.dataset.files_A[i])
                        
                        image_FB = 255 * ((fake_B + 1) / 2)
                        size = self.config['size']
                        image_FB = np.transpose(image_FB, (1, 2, 0))
                        image_fname = os.path.join(self.config["image_save"], serial)
                        if self.config['output_nc'] == 1:
                            cv2.imwrite(image_fname, image_FB)
                        else:
                            for i in range(self.config['output_nc']):
                                cv2.imwrite(image_fname.replace('.png', f'_channel{i+1}.png'), image_FB[:,:,i])

                print ('MAE:',MAE/num)
                
    def evaluate(self, input_dir, output_dir, channels=['CD20', 'CD3', 'CD4', 'CD8', 'DAPI', 'cytokeratin']):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        val_trans = [Normalize(self.config['output_nc']), ToTensor(),]
        val_data = get_val_dataset(input_dir, self.config['input_nc'], self.config['size'], trans=val_trans)
        val_loader = DataLoader(
            val_data,
            batch_size=self.config['batchSize'], 
            shuffle=False, 
            num_workers=self.config['n_cpu'],
            drop_last=True)
        
        with torch.no_grad():
            batchsize = self.config['batchSize']
            for i, batch in enumerate(val_loader):
                real_A = Variable(self.input_A.copy_(batch['A']))

                filepaths = batch['filepath']
            
                fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                
                for k in range(batchsize):
                    image_FB = 255 * ((fake_B[k,:,:,:] + 1) / 2)
                    image_FB = image_FB.squeeze()
                    
                    filepath = filepaths[k]
                    subdir = os.path.basename(os.path.dirname(filepath))
                    savedir = os.path.join(output_dir, subdir)
                    if not os.path.exists(savedir):
                        os.makedirs(savedir, exist_ok=True)
                    
                    image_FB = np.transpose(image_FB, (1, 2, 0))
                    for j in range(self.config['output_nc']):
                        image_fname = os.path.join(savedir, f"{channels[j]}_{os.path.basename(filepath).replace('HE', 'MIF')}")
                        cv2.imwrite(image_fname, image_FB[:,:,j])                

          
                
    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                num = 0
                
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    
                    mae = self.MAE(fake_B,real_B)
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = measure.compare_ssim(fake_B,real_B)
                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim 
                    num += 1
                print ('MAE:',MAE/num)
                print ('PSNR:',PSNR/num)
                print ('SSIM:',SSIM/num)
    
    def PSNR(self,fake,real):
       x,y = np.where(real!= -1)
       mse = np.mean(((fake[x][y]+1)/2. - (real[x][y]+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
            
    def MAE(self,fake,real):
        mae = 0.0
        if len(real.shape) == 2:
            x, y = np.where(real!= -1)
            if x.size != 0: 
                mae = np.abs(fake[x,y]-real[x,y]).mean()
        elif len(real.shape) == 3:
            x, y, z = np.where(real != -1)
            if x.size != 0: 
                mae = np.abs(fake[x,y,z] - real[x,y,z]).mean()
        else: 
            b, x, y, z = np.where(real != -1)
            if x.size != 0: 
                mae = np.abs(fake[b, x, y, z] - real[b, x, y, z]).mean()
            
        return mae/2  
            
            
            

