
import itertools
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import get_datasets_comp, get_datasets, get_val_dataset
# from .datasets import ImageDataset,ValDataset
from Model.CycleGan import *
from .utils import Resize, ToTensor, Normalize, smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import ToPILImage, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
from glob import glob
import json
import cv2
from collections import OrderedDict
import datetime

class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc'])
        self.netD_B = Discriminator(config['output_nc'])
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['output_nc'],config['input_nc'])
            self.spatial_transform = Transformer_2D()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['output_nc'], config['input_nc'])
            self.netD_A = Discriminator(config['input_nc'])
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            
        # support multiple GPUs
        if torch.cuda.device_count() > 1:
            self.netG_A2B = nn.DataParallel(self.netG_A2B).cuda()
            self.netD_B = nn.DataParallel(self.netD_B).cuda()
            if config['regist']:
                self.R_A = nn.DataParallel(self.R_A).cuda()
                self.spatial_transform = nn.DataParallel(self.spatial_transform).cuda()
            if config['bidirect']:
                self.netG_B2A = nn.DataParallel(self.netG_B2A).cuda()
                self.netD_A = nn.DataParallel(self.netD_A).cuda()
        else:
            self.netG_A2B = self.netG_A2B.cuda()
            self.netD_B = self.netD_B.cuda()
            if config['regist']:
                self.R_A = self.R_A.cuda()
                self.spatial_transform = self.spatial_transform.cuda()
            if config['bidirect']:
                self.netG_B2A = self.netG_B2A.cuda()
                self.netD_A = self.netD_A.cuda()
        
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
        if 'source_root' in config:
            train, val = get_datasets(
                        config['source_root'], config['target_root'], 
                        config['input_nc'], config['output_nc'], config['size']
                        )
        else:
            level = 1
            trans = [
                        ToPILImage(),
                        RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],fill=-1),
                        RandomHorizontalFlip(p=0.5),
                        RandomVerticalFlip(p=0.5),
                        Normalize(config['output_nc']),
                        ToTensor(),
                    ]
            val_trans = [
                        Normalize(config['output_nc']),
                        ToTensor(),
                    ]
            train, val = get_datasets_comp(
                        config['tilesdir'], 
                        config['valdir'],
                        config['input_nc'], config['output_nc'], config['size'], 
                        trans=trans,
                        val_trans=val_trans,
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

        weight_path_reg = self.config['model_save'] + f'Regist_epoch{epoch}.pth'
        if self.config['regist'] and os.path.exists(weight_path_reg):
            print(f"loading weights registration from epoch {epoch}")
            self.R_A.load_state_dict(get_state_d(weight_path_reg))
            self.R_A = nn.DataParallel(self.R_A.cuda())
        
    def save_weights(self, epoch):
        if (epoch != self.config['n_epochs'] - 1) and ((epoch + 1) % 5 != 0): return
        if not os.path.exists(self.config['model_save']):
            os.makedirs(self.config['model_save'])
        if isinstance(self.netG_A2B, nn.DataParallel):
            torch.save(self.netG_A2B.module.state_dict(), self.config['model_save'] + f'netG_A2B_epoch{epoch+1}.pth')
            torch.save(self.netD_B.module.state_dict(), self.config['model_save'] + f'netD_B_epoch{epoch+1}.pth')
            if self.config['regist']:
                torch.save(self.R_A.module.state_dict(), self.config['model_save'] + f'Regist_epoch{epoch+1}.pth') 
            if self.config['bidirect']:
                torch.save(self.netG_B2A.module.state_dict(), self.config['model_save'] + f'netG_B2A_epoch{epoch+1}.pth')
                torch.save(self.netD_A.module.state_dict(), self.config['model_save'] + f'netD_A_epoch{epoch+1}.pth')
        else:
            torch.save(self.netG_A2B.state_dict(), self.config['model_save'] + f'netG_A2B_epoch{epoch+1}.pth')
            torch.save(self.netD_B.state_dict(), self.config['model_save'] + f'netD_B_epoch{epoch+1}.pth')
            if self.config['regist']:
                torch.save(self.R_A.state_dict(), self.config['model_save'] + f'Regist_epoch{epoch+1}.pth')
            if self.config['bidirect']:
                torch.save(self.netG_B2A.state_dict(), self.config['model_save'] + f'netG_B2A_epoch{epoch+1}.pth')
                torch.save(self.netD_A.state_dict(), self.config['model_save'] + f'netD_A_epoch{epoch+1}.pth')
      
    def train(self):
        self.load_weights(self.config['epoch'])
        if not os.path.exists(self.config['save_root']):
            os.makedirs(self.config['save_root'], exist_ok=True)
        
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for batch_idx, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                if self.config['bidirect']:   # C dir
                    if self.config['regist']:    #C + R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        
                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        
                        Trans = self.R_A(fake_B,real_B) 
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        
                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss +SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()
                        
                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ################################### 
                    
                    else: #only  dir:  C
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda']*self.MSE_loss(pred_fake, self.target_real)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                        loss_Total.backward()
                        self.optimizer_G.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################
                        
                        
                        
                else:                  # s dir :NC
                    if self.config['regist']:    # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B = self.netG_A2B(real_A)
                        Trans = self.R_A(fake_B,real_B) 
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        toal_loss = SM_loss+adv_loss+SR_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)


                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        
                        
                        
                    else:        # only NC
                        self.optimizer_G.zero_grad()
                        fake_B = self.netG_A2B(real_A)
                        #### GAN aligin loss
                        pred_fake = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        adv_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        ###################################


                self.logger.log({'loss_D_B': loss_D_B,},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})#,'SR':SysRegist_A2B

            
            self.save_weights(epoch)
            self.val(epoch)
                         
                    
    def val(self, epoch):
        '''
        Evaluate validation set.
        '''
        with torch.no_grad():
            MAE = 0
            num = 0
            for i, batch in enumerate(self.val_data):
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                mae = self.MAE(fake_B,real_B.detach().cpu().numpy().squeeze())
                MAE += mae
                num += 1

                # if epoch >= self.config['n_epochs'] - 1:
                if (epoch + 1) % 5 == 0 or epoch == self.config['n_epochs'] - 1:
                    savedir = self.config["image_save"] + f'img_{epoch+1}'
                    if not os.path.exists(savedir):
                        os.makedirs(savedir, exist_ok=True)
      
                    batchsize = self.config['batchSize']
                    for k in range(batchsize):
                        image_FB = 255 * ((fake_B[k,:,:,:] + 1) / 2)
                        image_FB = image_FB.squeeze()

                        if self.config['output_nc'] == 1:
                            image_FB = np.transpose(image_FB, (1, 0))
                            serial = os.path.basename(self.val_data.dataset.files_B[i*batchsize+k])
                            image_fname = os.path.join(savedir, serial)
                            cv2.imwrite(image_fname, image_FB)
                        else:
                            image_FB = np.transpose(image_FB, (1, 2, 0))
                            for j in range(self.config['output_nc']):
                                serial = os.path.basename(self.val_data.dataset.files_B[j, i*batchsize+k])
                                image_fname = os.path.join(savedir, serial)
                                cv2.imwrite(image_fname, image_FB[:,:,j])

            print ('Val MAE:',MAE/num)
            print(datetime.datetime.now())

    def evaluate(self, input_dir, output_dir, channels):
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
    
    
    def evaluate_discriminator(self, d, savedir, channels):
        dpaths = []
        if '*' in d:
            allfiles = sorted(glob(d))
            for channel in channels:
                lst = [f for f in allfiles if channel in f]
                dpaths.append(lst)
        else:
            allfiles = sorted(os.listdir(d))
            for channel in channels:
                lst = [os.path.join(d,f) for f in allfiles if channel in f]
                dpaths.append(lst)
        
        assert all(len(lst) == len(dpaths[0]) for lst in dpaths)

        preds = []
        with torch.no_grad():
            for paths in zip(*dpaths):
                images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
                fake = np.stack(images, axis=0)
                fake = cv2.normalize(fake, None, -1, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                fake = torch.from_numpy(np.expand_dims(fake, 0)).cuda()
                pred_fake0 = self.netD_B(fake)
                preds.append(pred_fake0.item())
        with open(savedir, 'w') as f:
            json.dump(preds, f)
                 
    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        #self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'Regist.pth'))
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    
                    fake_B = self.netG_A2B(real_A)
                    fake_B = fake_B.detach().cpu().numpy().squeeze()                                                 
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
       x,y = np.where(real!= -1)# Exclude background
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
        else: # first dim batchsize
            b, x, y, z = np.where(real != -1)
            if x.size != 0: 
                mae = np.abs(fake[b, x, y, z] - real[b, x, y, z]).mean()
            
        return mae/2 
            

    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 
