import numpy as np
from glob import glob
import torch
from torch import nn
import cv2
import scipy.io as sio
from torch.nn import functional as F
import imgvision as iv
import torch.utils.data as data
import torchvision
from torch.nn import init

def setup_seed(seed):
    import random
    import os
    #  下面两个常规设置了，用来np和random的话要设置
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU训练需要设置这个
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True,warn_only=True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        '''
         use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
        '''
        self.feature_list = [2, 7, 14]
        vgg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)

        self.model = torch.nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1]+1])


    def forward(self, x):
        x = (x-0.5)/0.5
        features = []
        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)
        return features

def VGGPerceptualLoss(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''

    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss

class PerceptualLoss():
    def __init__(self,real):
        self.weights = [1, 0.2, 0.04]
        self.vgg=VGG19().cuda()
        features_real = self.vgg(real)
        self.features_real_no_grad = [f_real.detach() for f_real in features_real]
        self.loss = nn.MSELoss(reduction='mean')

    def go(self,fake):
        features_fake = self.vgg(fake)
        loss = 0
        for i in range(len( self.features_real_no_grad )):
            loss_i = self.loss(features_fake[i],self.features_real_no_grad[i])
            loss = loss + loss_i * self.weights[i]

        return loss

def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def add_noise(img,SNR):
    '''

    :param img:待添加噪音图像
    :param SNR: 噪音信噪比
    :return: 加噪音后的图像
    最后更新 2022年6月16日 曹栩珩
    '''

    M,N,B = img.shape
    gamma = torch.sqrt(pow(img, 2).sum() / (10 ** (SNR / 10)) / (img.numel()))
    img += gamma*torch.randn(M,N,B)
    return img

def tensor2array(data):
    return data[0].detach().numpy().T

def array2tensor(data):
    return torch.tensor(data.T).float().unsqueeze(0)

class Dataloader():
    def __init__(self,args):
        self.Filelist = glob(args.path+'*.mat')
        self.args = args
    def load(self,index):
        self.Filename = self.Filelist[index].split('\\')[-1]
        # base = sio.loadmat(self.Filelist[index])
        base = sio.loadmat(f'{self.args.path}{index}.mat')
        key = list(base.keys())[-1]
        # key = 'ref'
        Ground_Truth = base[key]
        # Ground_Truth = Ground_Truth/Ground_Truth.max()
        # Ground_Truth = Ground_Truth[:1024,:1024]
        # print('\nLoad {} Successfully\n'.format(self.Filelist[index].split('\\')[-1] ))
        return Ground_Truth
    def save(self,args,img):
        np.save(args.save_path+self.Filename.replace('.mat',''),img)

def generate_lrhsi(gt,net):
    return net(torch.tensor(gt.T).unsqueeze(0).float())

def generate_hrrgb(gt,srf):
    return gt @ srf

def get_pairs(gt,net,srf,noise=None,NSR_HSI=30,NSR_RGB=40):
    lrhsi = generate_lrhsi(gt,net)[0]
    hrrgb = torch.tensor(generate_hrrgb(gt,srf).T).float()
    if noise:
        lrhsi = add_noise(lrhsi,NSR_HSI)
        hrrgb = add_noise(hrrgb,NSR_RGB)

    print("\033[1;36m Image Settings \033[0m".center(61, '-'))
    print('\t\t\033[1;33m LR-HSI\t\t\tHR-RGB \033[0m'.ljust(50))
    print(f'size\t{lrhsi.shape}\t{hrrgb.shape}'.ljust(50))
    print(f'noise\t{noise}'.ljust(50))
    print(''.center(50,'-'))
    return lrhsi,hrrgb

def getInputImgs(args,index,spadown,noise=False):
    dataset = Dataloader(args)
    GT = dataset.load(index)[:128,:128]

    SRF = np.load(args.srfpath)
    lr_hsi, hr_rgb = get_pairs(GT, spadown, SRF, noise=noise)
    return lr_hsi.unsqueeze(0),hr_rgb.unsqueeze(0),GT

class Dataset(data.Dataset):
    def __init__(self,img1,img2,args):
        self.im1 = img1
        self.args = args
        self.im2 = img2
    def __getitem__(self, item):
        im1 = F.avg_pool2d( F.pad( self.im1.unsqueeze(0),pad=[1]*4,mode='circular' )   ,3, 1 )
        im2 = F.avg_pool2d(  F.pad( self.im2.unsqueeze(0),pad=[self.args.sf//2]*4,mode='circular' ) ,2*self.args.sf, self.args.sf)

        return im1[0], im2[0]
    def __len__(self):
        return 1

def PSNR_GPU(im_true, im_fake):
    data_range = 1
    _,C,H,W = im_true.size()
    err = torch.pow(im_true.clone()-im_fake.clone(),2).mean(dim=(-1,-2), keepdim=True)
    psnr = 10. * torch.log10((data_range**2)/err)
    return torch.mean(psnr)

class SAM_Loss(nn.Module):
    def __init__(self):
        super(SAM_Loss, self).__init__()

    def forward(self, output, label):
        ratio = (torch.sum((output + 1e-8).multiply(label + 1e-8), axis=1)) / (torch.sqrt(
            torch.sum((output + 1e-8).multiply(output + 1e-8), axis=1) * torch.sum(
                (label + 1e-8).multiply(label + 1e-8), axis=1)))
        angle = torch.acos(ratio.clip(-1, 1))

        return torch.mean(angle)

def TV_Loss(data):
    return torch.abs(data[:,:,1:]-data[:,:,:-1]).mean()+torch.abs(data[:,:,:,1:]-data[:,:,:,:-1]).mean()

def tensor_svds(data, K):
    b, c, h, w = data.shape
    U, S, vh = torch.linalg.svd(data.flatten(2)[0], full_matrices=False)
    phi = U[:, :K] @ torch.diag(S[:K])
    C = vh[:K]
    return phi.view([1, c, K, 1]), C.view([1, K, h, w])

def Couple_init(spa,spe, msi, hsi, GT_T,k=12):
    print('Start PSNR:{:.2f}'.format(PSNR_GPU(msi, spe(GT_T))))
    Batch, Channel, Height, Weight = hsi.shape
    phi, c = tensor_svds(hsi, k)


    trainer_SPE = torch.optim.AdamW(params=spe.parameters(), lr=5e-3, weight_decay=0.0001)
    lrsched_SPE = torch.optim.lr_scheduler.StepLR(trainer_SPE, 50, 0.8)
    trainer_SPA = torch.optim.AdamW(params=spa.parameters(), lr=5e-3, weight_decay=0.0001)
    lrsched_SPA = torch.optim.lr_scheduler.StepLR(trainer_SPA, 50, 0.8)
    max_epochs = 1000
    L1 = nn.L1Loss()
    L = []
    for epoch in range(max_epochs):
        trainer_SPA.zero_grad()
        trainer_SPE.zero_grad()
        pre_phi = spe(phi)
        lrmsi = spa(msi)
        loss =L1(lrmsi, torch.bmm(pre_phi[:, :, :, 0], c.view(1, k, -1)).view(1, 3, Height, Weight)) + L1(lrmsi,spe(hsi))
        loss.backward()
        trainer_SPE.step()
        lrsched_SPE.step()
        trainer_SPA.step()
        lrsched_SPA.step()
        L.append(loss.detach().cpu().numpy())
    print('End PSNR:{:.2f}'.format(PSNR_GPU(msi, spe(GT_T))))

def sp_tv(data):
    return torch.abs(data[:,1:]-data[:,:-1]).mean()
class wTV_Loss():
    def __init__(self,rgb):

        rgb1 = torch.nn.functional.pad(rgb,[0,0,0,1],mode='replicate')

        rgb2 = torch.nn.functional.pad(rgb,[0,1,0,0],mode='replicate')
        gray1 = rgb1.mean(1)
        gray2 = rgb2.mean(1)

        N_0 = gray1[:,:-1,:]-gray1[:,1:,:]

        N_1 = gray2[:,:,:-1]-gray2[:,:,1:]

        n = 0.01
        W = n / (torch.sqrt(pow(abs(N_0) + abs(N_1), 2) + pow(n, 2)))
        self.W = W.repeat(31,1,1).unsqueeze(0)

    def loss(self,data):
        return TV_Loss(self.W*data)

def SSIM_GPU(r_img,f_img,k1=0.01, k2=0.03):
    l = 1
    x1_ = r_img.view(r_img.size(1),-1)
    x2_ = f_img.view(f_img.size(1),-1)
    u1 = x1_.mean(dim=-1,keepdim=True)
    u2 = x1_.mean(dim=-1,keepdim=True)
    Sig1 = torch.std(x1_, dim=-1,keepdim=True)
    Sig2 = torch.std(x2_, dim=-1,keepdim=True)
    sig12 = torch.sum((x1_ - u1) * (x2_ - u2), dim=-1) / (x1_.size(-1) - 1)
    c1, c2 = pow(k1 * l, 2), pow(k2 * l, 2)
    SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))
    return SSIM.mean()


def printQUI(GT,imgs,sf):
    name = ['X_in','X_act','X_re']
    print(''.ljust(50,'_'))
    print('Name\tPSNR\tSAM\t\tERGAS\tSSIM\tRMSE'.ljust(50))
    for idx,i in enumerate(imgs):
        P,SAM,E,SSIM,R = spectra_metric(GT,i,scale=sf).get_Evaluation()
        print(name[idx]+'\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(P,SAM,E,SSIM,np.sqrt(R)*255))
    print(''.ljust(50, '-'))

def saveQUI_txt(path,metrics):
    f_list = open(path,'a')
    f_list.write(metrics)
    f_list.close()

def head_generator(path):
    f_list = open(path, 'a')
    f_list.write('PSNR\tSAM\t\tERGAS\tSSIM\tRMSE\n')
    f_list.close()

class saveQUI():
    def __init__(self,args):
        self.path = args.save_path+str(args.sf)+'_Quantitative_Results.txt'
        self.sf = args.sf
        head_generator(self.path)
    def step(self,GT,recon):
        P, SAM, E, SSIM, R = iv.spectra_metric(GT, recon, scale=self.sf).get_Evaluation()
        self.metrics = '{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(P,SAM,E,SSIM,np.sqrt(R)*255)

    def save(self):
        saveQUI_txt(self.path,self.metrics)

    def print(self):

        print('PSNR\tSAM\tERGAS\tSSIM\tRMSE\n')
        print(self.metrics)
        print(''.center(50,'—'))
        print('\n\n')