import os
import shutil
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn as nn
from diffusers.models import AutoencoderKL
import traceback
from pytorch_msssim import ssim
from torchvision.utils import make_grid
import kornia
from kornia.augmentation.container import AugmentationSequential
# 检查CUDA是否可用，并据此设置device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from diffusers.models.vae import Decoder
from model.EEGVisModels import FreqEncoder,TimeFreqEncoder,TimeEncoder
from args import args
import argparse
parser = argparse.ArgumentParser(description="Template")
parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
# Parse arguments
opt = parser.parse_args()
args.data_shape = (440,128)
timeE = TimeEncoder(args).to("cuda")
freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value)
                        for (key, value) in [x.split("=") for x in opt.model_params]}
freq_model = FreqEncoder(**freq_model_options)

timefreq_model = TimeFreqEncoder(timeE, freq_model, args)
timefreq_model = timefreq_model.to("cuda")

freqtime_state_dict = torch.load('pretrained_model/timefreqmodel_1.pkl', map_location="cuda")
timefreq_model.load_state_dict(freqtime_state_dict)

timefreq_model.requires_grad_(False)
timefreq_model.eval()

class eeg2StableDiffusionModel(torch.nn.Module):
    def __init__(self, in_dim=1152, h=4096, n_blocks=4, use_cont=False, ups_mode='8x'):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])
        self.ups_mode = ups_mode
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  # prev best
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()
        
        if ups_mode=='16x':
            self.lin1 = nn.Linear(h, 8192, bias=False)
            self.norm = nn.GroupNorm(1, 512)
            
            self.upsampler = Decoder(
                in_channels=512,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256, 512],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()

    def forward(self, x, return_transformer_feats=False):
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        if self.ups_mode == '16x':
            side = 4
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)

lr_scheduler = 'cycle'
batch_size = 1
max_lr = 5e-4
num_epochs = 120
num_devices = 1
mixup_pct = -1
ckpt_saving = True
ckpt_interval = 10
save_at_end = False
cont_model = 'cnx'
ups_mode = '8x'
use_cont = False    
eeg2sd = eeg2StableDiffusionModel(use_cont=use_cont,ups_mode=ups_mode)
eeg2sd = eeg2sd.to(device)
eeg2sd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(eeg2sd)
mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)

outdir = 'exp/try1'
def save_ckpt(tag):
    ckpt_path = os.path.join(outdir, f'{tag}.pth')
    if tag == "last":
        if os.path.exists(ckpt_path):
            shutil.copyfile(ckpt_path, os.path.join(outdir, f'{tag}_old.pth'))
    print(f'saving {ckpt_path}')
    state_dict = eeg2sd.state_dict()
    for key in list(state_dict.keys()):
        if 'module.' in key:
            state_dict[key.replace('module.', '')] = state_dict[key]
            del state_dict[key]
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': losses,
            'val_losses': val_losses,
            'lrs': lrs,
            }, ckpt_path)
    except:
        print('Failed to save weights')
        print(traceback.format_exc())
    if tag == "last":
        if os.path.exists(os.path.join(outdir, f'{tag}_old.pth')):
            os.remove(os.path.join(outdir, f'{tag}_old.pth'))
    
f = open("data/EEG_divided/Subj_01_egg_label_fold_image/Train_eeg_image1.pkl", "rb")
train = torch.load(f,map_location='cpu')
f.close() 
f = open("data/EEG_divided/Subj_01_egg_label_fold_image/Test_eeg_image1.pkl", "rb")
test = torch.load(f,map_location='cpu')
f.close()

train_data = train
test_data = test

num_train = len("numbers of train data:",train_data[1])
num_val = len("numbers of test data:",test_data[1])

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in eeg2sd.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in eeg2sd.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 
                                            total_steps=num_epochs*((num_train//batch_size)//num_devices), 
                                            final_div_factor=1000,
                                            last_epoch=-1, pct_start=2/num_epochs)
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

losses = []
val_losses = []
lrs = []
best_val_loss = 1e10
best_ssim = 0

autoenc = AutoencoderKL(
    down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    sample_size=256
)
autoenc = autoenc.to('cuda')
autoenc.load_state_dict(torch.load('pretrained_model/sd_image_var_autoenc.pth'))
autoenc.requires_grad_(False)
autoenc.eval()

class eegImageDataset(Dataset):
    def __init__(self, eeg_data, labels, image_names, image_dir,transform = None):
        self.eeg_data = eeg_data
        self.labels = labels
        self.image_names = image_names
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()  
        ])
    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        label = self.labels[idx]
        image_name = self.image_names[idx]
        image_name += '.JPEG'
        image_path = os.path.join(self.image_dir.replace("\\", os.sep), label.replace("\\", os.sep), image_name)
        image = Image.open(image_path).convert('RGB') 

        if self.transform:
            image = self.transform(image)  
        return eeg, image

transform = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor(),
])

train_dataset = eegImageDataset(
    eeg_data=train_data[0],
    labels=train_data[2],
    image_names=train_data[3],
    image_dir='data\image',
    transform=transform  
)
test_dataset = eegImageDataset(
    eeg_data=test_data[0],
    labels=test_data[2],
    image_names=test_data[3],
    image_dir='data\image',
    transform=transform  
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

progress_bar = tqdm(range(num_epochs), ncols=150)

for epoch in progress_bar:
    eeg2sd.train()
    loss_mse_sum = 0
    val_loss_mse_sum = 0
    loss_reconst_sum = 0
    val_loss_reconst_sum = 0
    val_ssim_score_sum = 0
    reconst_fails = []

    for train_i, (eeg, image) in enumerate(train_loader):

        optimizer.zero_grad()

        image = image.to(device).float()
        eeg = eeg.to(device).float()

        with torch.cuda.amp.autocast(enabled = False):  
            autoenc_image = kornia.filters.median_blur(image, (7, 7))
            image_enc = autoenc.encode(2*autoenc_image - 1).latent_dist.mode() * 0.18215
            
            _,encoded,_ = timefreq_model(eeg)
            image_enc_pred = eeg2sd(encoded)
            
            mse_loss = F.l1_loss(image_enc_pred, image_enc)
                
            del eeg,image
            
            reconst = autoenc.decode(image_enc_pred/0.18215).sample
            reconst_loss = F.l1_loss(reconst, 2*autoenc_image - 1)
            loss = mse_loss/0.18215 + reconst_loss*2
            del autoenc_image

            check_loss(loss)
            loss_mse_sum += mse_loss.item()
            loss_reconst_sum += reconst_loss.item()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            
            logs = OrderedDict(
                train_loss=np.mean(losses[-(train_i+1):]),
                val_loss=np.nan,
                lr=lrs[-1],
            )
            progress_bar.set_postfix(**logs)
        
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

 
    eeg2sd.eval()
    for val_i, (eeg, image) in enumerate(test_loader): 
        with torch.inference_mode():
            image = image.to(device).float()     
            eeg = eeg.to(device).float()
            
            with torch.cuda.amp.autocast(enabled=False):
                image = kornia.filters.median_blur(image, (7, 7))
                image_enc = autoenc.encode(2*image-1).latent_dist.mode()*0.18215
                
                _,encoded,_ = timefreq_model(eeg)
                image_enc_pred = eeg2sd(encoded)

                mse_loss = F.l1_loss(image_enc_pred, image_enc)
                
                del eeg 
                
                reconst = autoenc.decode(image_enc_pred/0.18215).sample
                reconst_loss = F.l1_loss(reconst, 2*image-1)
                
                ssim_score = ssim((reconst/2 + 0.5).clamp(0,1), image, data_range=1, size_average=True, nonnegative_ssim=True)
                 
                val_loss_mse_sum += mse_loss.item()
                val_loss_reconst_sum += reconst_loss.item()
                val_ssim_score_sum += ssim_score.item()
  
                val_losses.append(mse_loss.item()/0.18215 + reconst_loss.item()*2)  

        logs = OrderedDict(
            train_loss=np.mean(losses[-(train_i+1):]),
            val_loss=np.mean(val_losses[-(val_i+1):]),
            lr=lrs[-1],
        )
        progress_bar.set_postfix(**logs)

    if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
        # save best model
        val_loss = np.mean(val_losses[-(val_i+1):])
        val_ssim = val_ssim_score_sum / (val_i + 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt('best')
        else:
            print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            save_ckpt('best_ssim')
        else:
            print(f'not best - val_ssim: {val_ssim:.3f}, best_ssim: {best_ssim:.3f}')

        if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
            save_ckpt(f'epoch{(epoch+1):03d}')
        try:
            orig = image
            if reconst is None:
                reconst = autoenc.decode(image_enc_pred.detach()/0.18215).sample
                orig = image
            pred_grid = make_grid(((reconst/2 + 0.5).clamp(0,1)*255).byte(), nrow=int(len(reconst)**0.5)).permute(1,2,0).cpu().numpy()
            orig_grid = make_grid((orig*255).byte(), nrow=int(len(orig)**0.5)).permute(1,2,0).cpu().numpy()
            comb_grid = np.concatenate([orig_grid, pred_grid], axis=1)
            del pred_grid, orig_grid
            Image.fromarray(comb_grid).save(f'{outdir}/reconst_epoch{(epoch+1):03d}.png')
        except:
            print("Failed to save reconst image")
            print(traceback.format_exc())

    logs = {
        "train/loss": np.mean(losses[-(train_i+1):]),
        "val/loss": np.mean(val_losses[-(val_i+1):]),
        "train/l1_loss": loss_mse_sum / (train_i + 1),
        "train/loss_reconst": loss_reconst_sum / (train_i + 1),
        "val/l1_mse": val_loss_mse_sum / (val_i + 1),
        "val/loss_reconst": val_loss_reconst_sum / (val_i + 1),
        "val/ssim": val_ssim_score_sum / (val_i + 1),
    }
    print(logs)
    if len(reconst_fails) > 0 :
        print(f'Reconst fails {len(reconst_fails)}/{train_i}: {reconst_fails}')
