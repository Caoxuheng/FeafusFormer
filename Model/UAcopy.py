# -*- coding: utf-8 -*-
"""
SpaDnet
"""
import torch
import torch.nn.functional as F
from .common import *
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import torchsummary as ts

from utils import PSNR_GPU
vgg16 = vgg16_bn
VGG16_Weights = VGG16_BN_Weights
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
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize PSF network with %s' % init_type)
    net.apply(init_func)



def skip(num_input_channels=2, num_output_channels=3,
        num_channels_down=[40], num_channels_up=[40],
        num_channels_skip=[1],n_scales=2,
        filter_size_down=5, filter_size_up=5, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    num_channels_down *= n_scales
    num_channels_up *= n_scales
    num_channels_skip *= n_scales
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales
    last_scale = n_scales - 1
    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip = nn.Sequential()
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        # deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 1, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # self.feature_list = [7,14,21]
        self.feature_list = [4, 11, 21]
        vgg16_ = vgg16(weights=VGG16_Weights.DEFAULT)

        self.model = torch.nn.Sequential(*list(vgg16_.features.children())[:self.feature_list[-1] + 1])

    def forward(self, x):
        x = (x - 0.5) / 0.5
        features = []
        with torch.no_grad():
            for i, layer in enumerate(list(self.model)):
                x = layer(x)
                if i in self.feature_list:
                    features.append(x.detach())
        return features


def SSIM(r_img,f_img,k1=0.01, k2=0.03):
    l = 1
    x1_ = r_img.view(r_img.size(1),-1)
    x2_ = f_img.view(f_img.size(1),-1)
    u1 = x1_.mean(dim=-1,keepdim=True)
    u2 = x2_.mean(dim=-1,keepdim=True)
    Sig1 = torch.std(x1_, dim=-1,keepdim=True)
    Sig2 = torch.std(x2_, dim=-1,keepdim=True)
    sig12 = torch.sum((x1_ - u1) * (x2_ - u2), dim=-1) / (x1_.size(-1) - 1)
    c1, c2 = pow(k1 * l, 2), pow(k2 * l, 2)
    SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))
    return SSIM

def SSIM_multi(img,k1=0.01, k2=0.03):
    l = 1
    x1_ = img.view(img.size(1),-1)
    u1 = x1_.mean(dim=-1,keepdim=True)

    Sig1 = torch.std(x1_, dim=-1,keepdim=True)
    SSIM = torch.empty([img.size(1),img.size(1)],dtype=torch.float32)
    for i in range(img.size(1)):
        sig12 = torch.sum( (x1_ - u1) *(x1_ - u1)[i], dim=-1,keepdim=True) / (x1_.size(-1) - 1)
        c1, c2 = pow(k1 * l, 2), pow(k2 * l, 2)
        SSIM[i]  =( (2 * u1 * u1[i] + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u1[i] ** 2 + c1) * (Sig1 ** 2 + Sig1[i] ** 2 + c2))).ravel()
    return SSIM

def get_fakeRGBs(HSI):
    blue = [2-1,4-1,12-1]
    green = [14-1,18-1,21-1]
    red = [23-1,28-1,31-1]
    fake_rgbs = []
    fake_rgbs.append(  torch.concat( (HSI[:,red[i]],  HSI[:,green[i]]  ,  HSI[:,blue[i]]),dim=0 ).unsqueeze(0) for i in range(3) )
    return fake_rgbs[0]

def get_query(HSI,Feature_Extractor,level):

    FakeRgbs = get_fakeRGBs(HSI)
    Queries = []
    for i , sub_fakergb in enumerate(FakeRgbs):
        Query = Feature_Extractor(sub_fakergb)
        Queries.append(Query)
    return  Queries

def get_KeyValue(RGB,Feature_Extractor,sf,level):
    sf_num = [sf//4,sf//8,sf]
    if sf_num[level]!=0:
        RGBDU = F.interpolate(F.interpolate(RGB,scale_factor=1/sf_num[level],mode='bilinear'),scale_factor=sf_num[level],mode='bilinear')
    else:
        RGBDU = RGB
    Keys = Feature_Extractor(RGBDU)
    Values = Feature_Extractor(RGB)
    return Keys,Values

def get_QKVs(RGB,HSI,Extractor,sf,level):
    Qs = get_query(HSI,Extractor,level)
    Ks,Vs = get_KeyValue(RGB,Extractor,sf,level)
    return Qs,Ks,Vs

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super(ScaledDotProductAttention,self).__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  = torch.matmul(attn, v)
        return output


class ScaledDotProductAttentionOnly(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self):
        super().__init__()


    def forward(self, v, k, q, mask=None):
        b, c, h, w      = q.size(0), q.size(1), q.size(2), q.size(3)

        # Reshaping K,Q, and Vs...
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)


        # Compute attention
        attn = torch.matmul(q / (h*w)**0.5, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  =v+ torch.matmul(attn, v)

        #Reshape output to original format
        output  = output.view(b, c, h, w)
        return output


class FusionTransformerBlock(nn.Module):
    def __init__(self,level):
        super(FusionTransformerBlock, self).__init__()
        self.level = level
        self.multiatt1 = ScaledDotProductAttentionOnly()
        self.multiatt2 = ScaledDotProductAttentionOnly()
        self.multiatt3 = ScaledDotProductAttentionOnly()
    def forward(self,K,V,Q):
        K,V = K[self.level],V[self.level]
        Q1=Q[0][self.level]
        Q2=Q[1][self.level]
        Q3=Q[2][self.level]
        atten1 = self.multiatt1(V,K,Q1)
        atten2 = self.multiatt2(V, K,Q2 )
        atten3 = self.multiatt3(V, K,Q3 )
        x = torch.concat([atten1,atten2,atten3],dim=1)
        return x


class My_Bn(nn.Module):
    def __init__(self):
        super(My_Bn,self).__init__()
    def forward(self,x):
        return x - nn.AdaptiveAvgPool2d(1)(x)

class Multilevel_Fusion_Transformer(nn.Module):
    def __init__(self,args,Extractor):
        super(Multilevel_Fusion_Transformer, self).__init__()
        self.FTB_h=FusionTransformerBlock(2)
        self.up_16 = nn.Sequential(nn.Conv2d(768,31*16,1),nn.PixelShuffle(4))
        self.FTB_m=FusionTransformerBlock(1)
        self.up_4 =nn.Sequential( nn.Conv2d(384,31*4,1),nn.PixelShuffle(2))
        self.FTB_l=FusionTransformerBlock(0)
        self.skiph = nn.Sequential(nn.Conv2d(62,31,3,padding=1),nn.BatchNorm2d(31), skip(31,31,[40],[40],[2],2,act_fun='ELU'))
        self.skipm = nn.Sequential(nn.Conv2d(62,31,3,padding=1),nn.BatchNorm2d(31),skip(31,31,[40],[40],[2],2,act_fun='ELU'))
        self.skipl = nn.Sequential(nn.Conv2d(223,31,1),nn.BatchNorm2d(31),skip(31,31,[40],[40],[2],2,act_fun='ELU'))
        self.FE = Extractor
        self.sf = args.sf
        self.bn = My_Bn()



    def forward(self,x,rgb):

        # High-Level Fusion
        q, k, v = get_QKVs(rgb, x, self.FE, self.sf, 2)
        x1_ = self.up_16(self.FTB_h(k,v,q))
        x2 =self.bn( self.skiph(torch.concat([x1_, x], dim=1))) + x

        # Middel-Level Fusion
        q, k, v = get_QKVs(rgb, x, self.FE, self.sf, 1)
        x1_ = self.up_4(self.FTB_m(k,v,q))
        x2 = self.bn(self.skipm(torch.concat([x1_, x2], dim=1))) + x2
        # Low-Level Fusion
        q, k, v = get_QKVs(rgb, x, self.FE, self.sf, 0)
        x1_ = self.FTB_l(k,v,q)
        x2 = self.skipl(torch.concat([x1_, x2], dim=1))
        return x2



def MultiLevelFsusionTransformer(args):
    FE = VGG16()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Multilevel_Fusion_Transformer(args,Extractor=FE)
    model = model.to(device)
    return model


if __name__ =='__main__':
    from utils import *
    from config import args
    import imgvision as iv
    torch.manual_seed(10)
    kernels = sio.loadmat('D:/Python_Projrct/UMGAL/Kernel/motion_kernels.mat')['f_set']
    HSI = Dataloader(args).load(1)
    RGB = iv.spectra().space(HSI,'nkd700')
    gt =trans2( HSI )
    rgb = trans2(RGB)
    FE = VGG16()

    ts.summary(FE,(3,512,512))