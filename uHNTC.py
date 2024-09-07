import torch
import numpy as np
from config import args
from torch import nn
from utils import *
from Model.spa_down import SpaDown, initialize_SpaDNet
from Model.spe_down import SpeDown, initialize_SpeDNet
import torch.nn.functional as F
from Model.UA import MultiLevelFsusionTransformer
import imgvision as iv
import time



saver = saveQUI(args)
setup_seed(1)


idx=2
# Generate HSI and MSI
Spatialdown = SpaDown(args.sf,predefine=None)
HSI,MSI,GT = getInputImgs(args,idx,Spatialdown)
GT_T = array2tensor(GT).cuda()
HSI = HSI.cuda()
MSI= MSI.cuda()

'''
   =============Procedure 1==============
   |                                    |
   | Initialize the SpaDNet and SpeDNet |
   |                                    |
   ======================================
'''

span = [list(range(20, 31)), list(range(10, 23)), list(range(12))]
# Degradation Net Initialization
SpaDNet = SpaDown(sf=args.sf,predefine=None,iscal=True)
SpeDNet = SpeDown(span=span,predefine=None,iscal=True)
SpaDNet = SpaDNet.cuda()
SpeDNet= SpeDNet.cuda()
initialize_SpeDNet(module=SpeDNet,msi=MSI,hsi=HSI,sf=args.sf)
initialize_SpaDNet(module=SpaDNet,msi=MSI,msi2=SpeDNet(HSI))
Couple_init(SpaDNet,SpeDNet,MSI,HSI,GT_T)

'''
   =============Procedure 2==============
   |                                    |
   |        Blind HSI-MSI fusion        |
   |                                    |
   ======================================
'''

# construct network
fusionformer = MultiLevelFsusionTransformer(args)
trainer = torch.optim.AdamW(params=fusionformer.parameters(),lr=8e-3)
sched = torch.optim.lr_scheduler.StepLR(trainer,step_size=100,gamma=0.99)
trainer_spa = torch.optim.AdamW(params=SpaDNet.parameters(),lr=1e-4)
trainer_spe = torch.optim.AdamW(params=SpeDNet.parameters(),lr=1e-4)

# Define Loss
loss = nn.L1Loss()
p_loss=PerceptualLoss(MSI)
phi, c = tensor_svds(HSI,K=args.K)

# Training
Batch, Channel, Height, Weight = HSI.shape
X_coarse = F.interpolate(HSI, scale_factor=args.sf, mode='bicubic')
for i in range(args.pre_epoch):
    # Pre-train
    if i <args.pre_epoch//3:
        trainer.zero_grad()
        pre = fusionformer(X_coarse ,MSI,True)
        recon_HSI = SpaDNet(pre)
        recon_MSI = SpeDNet(pre)
        l = loss(recon_HSI,HSI)+loss(recon_MSI,MSI)+0.5*p_loss.go(SpeDNet(pre))+ 0.1*sp_tv(pre)
        l.backward()
        trainer.step()
        sched.step()
    else:
        trainer.zero_grad()
        trainer_spa.zero_grad()
        trainer_spe.zero_grad()
        pre = fusionformer( X_coarse,MSI,True)
        recon_HSI = SpaDNet(pre)
        recon_MSI = SpeDNet(pre)
        pre_phi = SpeDNet(phi)
        l1 = loss(recon_HSI,HSI)+loss(recon_MSI,MSI)+0.1*p_loss.go(SpeDNet(pre))+loss(SpaDNet(MSI),SpeDNet(HSI))  +loss(SpaDNet(MSI), torch.bmm(pre_phi[:, :, :, 0], c.view(1, args.K, -1)).view(1, 3, Height, Weight))
        l1.backward()
        trainer.step()
        trainer_spa.step()
        trainer_spe.step()
        sched.step()

# Training 2
for i in range(args.max_epoch):
    trainer.zero_grad()
    trainer_spa.zero_grad()
    trainer_spe.zero_grad()
    pre = fusionformer(X_coarse, MSI,True)
    recon_HSI = SpaDNet(pre)
    recon_MSI = SpeDNet(pre)
    l = loss(recon_HSI, HSI) + 0.8*loss(recon_MSI, MSI) + 0.1 * p_loss.go(SpeDNet(pre)) + loss(SpaDNet(MSI),SpeDNet(HSI))+ 0.1*sp_tv(pre)
    l.backward()
    trainer.step()
    trainer_spa.step()
    trainer_spe.step()
    sched.step()

Xre = tensor2array(pre.detach().cpu())
iv.spectra_metric(Xre,GT,scale=args.sf).Evaluation()
np.save(str(idx), Xre)
saver.step(GT, Xre)
saver.save()
saver.print()
