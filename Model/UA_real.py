# -*- coding: utf-8 -*-
from .common import *
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import torch.nn.functional as F
from torch.nn import init

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
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='reflect', upsample_mode='bicubic', downsample_mode='stride', act_fun='LeakyReLU',
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
        self.feature_list = [5, 12, 21]
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

class FE_VGG16(nn.Module):
    def __init__(self,FE,band):
        super(FE_VGG16, self).__init__()
        self.feature_list = [5-1, 12-1, 21-1]
        self.conv = nn.Conv2d(in_channels=band,out_channels=64,kernel_size=3,padding=1)
        self.model =nn.ModuleList(list(list(FE.children())[0][1:]))
        # self.model.requires_grad_(False)
    def forward(self,x):
        x = (x - 0.5) / 0.5
        features = []
        x = self.conv(x)

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)
        return features

def GetSubspace(data,d_type='PCA',n=15):
    infolen = len(data.shape)

    if infolen>3:
        b,c,h,w = data.shape
        data = data.reshape([b,c,-1])

    d_type=d_type.lower()

    if d_type=='pca':
        pca = torch.pca_lowrank
        phi,s,v = pca(data,n)
        C = v @ torch.diag_embed(s)
        C = C.transpose(1,2)
    return phi,C.reshape([b,n,h,w])

def subspace_combine(phi,C):
    _B,_C,_H,_W = C.shape
    _C_ = C.reshape([_B,_C,-1])
    return (phi @ _C_).reshape([_B,-1,_H,_W])

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

def get_query(HSI,Feature_Extractor_HSI):
    Query = Feature_Extractor_HSI(HSI)
    return  Query

def get_KeyValue(RGB,Feature_Extractor,sf):

    # if sf_num[level]!=0:
    RGBDU = F.interpolate(F.interpolate(RGB,scale_factor=1/sf,mode='bicubic'),scale_factor=sf,mode='bicubic')
    # else:
    #     RGBDU = RGB
    Keys = Feature_Extractor(RGBDU)
    Values = Feature_Extractor(RGB)
    return Keys,Values

def get_QKVs(RGB,HSI,Extractor,sf,band=31):
    Qs = get_query(HSI,FE_VGG16(Extractor,band).cuda())
    Ks,Vs = get_KeyValue(RGB,Extractor,sf)
    return Qs,Ks,Vs

class ScaledDotProductAttentionOnly(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self):
        super().__init__()


    def forward(self, v, k, q, mask=None):
        b, c, h, w      = q.size(0), q.size(1), q.size(2), q.size(3)

        # Reshaping K,Q, and Vs...
        q = q.reshape(b, c, h*w)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)


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

class MultiHeadAttention(nn.Module):


    def __init__(self, n_head, in_pixels, linear_dim, num_features):
        super().__init__()
        # Parameters
        self.n_head = n_head  # No of heads
        self.in_pixels = in_pixels  # No of pixels in the input image
        self.linear_dim = linear_dim  # Dim of linear-layer (outputs)

        # Linear layers

        self.w_qs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for queries
        self.w_ks = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for keys
        self.w_vs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for values
        self.fc = nn.Linear(n_head * linear_dim, in_pixels, bias=False)  # Final fully connected layer

        # Scaled dot product attention
        self.attention = ScaledDotProductAttentionOnly()

        # Batch normalization layer
        self.OutBN = nn.BatchNorm2d(num_features=num_features)

    def forward(self, v, k, q, mask=None):

        # Reshaping matrixes to 2D
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)
        n_head = self.n_head
        linear_dim = self.linear_dim

        # Reshaping K, Q, and Vs...
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Save V
        output = v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(b, c, n_head, linear_dim)
        k = self.w_ks(k).view(b, c, n_head, linear_dim)
        v = self.w_vs(v).view(b, c, n_head, linear_dim)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        # Computing ScaledDotProduct attention for each head
        v_attn = self.attention(v, k, q, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v_attn = v_attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
        v_attn = self.fc(v_attn)

        output = output + v_attn
        # output  = v_attn

        # Reshape output to original image format
        output = output.view(b, c, h, w)

        # We can consider batch-normalization here,,,
        # Will complete it later
        output = self.OutBN(output)
        return output

class FusionTransformerBlock(nn.Module):
    def __init__(self,level,init=True):
        super(FusionTransformerBlock, self).__init__()
        self.level = level
        if init is True:
            self.multiatt1 = ScaledDotProductAttentionOnly()
        else:
            n_f = [64,128,256]
            i_pi = [512*512//4,256*256//4,128*128//4]
            head = 5
            l_dim=10
            self.multiatt1 = MultiHeadAttention(n_head=head,in_pixels=i_pi[level],linear_dim=l_dim,num_features=n_f[level])
    def forward(self,K,V,Q):
        K_,V_ = K[self.level],V[self.level]

        Q_ = Q[self.level]

        x = self.multiatt1(V_,K_,Q_)

        return x

class My_Bn(nn.Module):
    def __init__(self):
        super(My_Bn,self).__init__()
    def forward(self,x):
        return x - nn.AdaptiveAvgPool2d(1)(x)

class Multilevel_Fusion_Transformer(nn.Module):
    def __init__(self,args,Extractor):
        super(Multilevel_Fusion_Transformer, self).__init__()
        HSband = 31
        self.FTB_h2=FusionTransformerBlock(2,init=False)
        self.up_16 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, groups=256,bias=None), nn.ELU(), nn.Conv2d(256, 64 * 4, 1,bias=None),nn.PixelShuffle(2), nn.ELU(),
                                   nn.Conv2d(64, 64, 3, padding=1, groups=64,bias=None), nn.ELU(), nn.Conv2d(64, HSband * 4, 1,bias=None),nn.PixelShuffle(2),
                                    nn.ELU(),nn.Conv2d(HSband,HSband,3,padding=1,groups=HSband,bias=None),nn.Conv2d(HSband,HSband,1,bias=None),nn.ELU())

        self.FTB_m2=FusionTransformerBlock(1,init=False)
        self.up_4 =nn.Sequential( nn.Conv2d(128,128,3,padding=1,groups=128,bias=None),nn.ELU(),nn.Conv2d(128,4*HSband,1,bias=None),nn.PixelShuffle(2),nn.ELU(),nn.Conv2d(HSband,HSband,3,padding=1,groups=HSband,bias=None),nn.Conv2d(HSband,HSband,1,bias=None),nn.ELU())
        self.FTB_l2=FusionTransformerBlock(0,init=False)

        self.FTB_h = FusionTransformerBlock(2, init=True)
        self.FTB_m = FusionTransformerBlock(1, init=True)
        self.FTB_l = FusionTransformerBlock(0, init=True)

        self.skiph = nn.Sequential(nn.Conv2d(HSband+HSband,HSband,3,padding=1,bias=None),nn.InstanceNorm2d(HSband),nn.ELU(), skip(HSband,HSband,[HSband],[HSband],[1],3,act_fun='ELU'))
        self.skipm = nn.Sequential(nn.Conv2d(HSband+HSband,HSband,3,padding=1,bias=None),nn.InstanceNorm2d(HSband),nn.ELU(),skip(HSband,HSband,[HSband],[HSband],[1],3,act_fun='ELU'))
        self.skipl = nn.Sequential(nn.Conv2d(HSband+64,HSband,1,bias=None),nn.InstanceNorm2d(HSband),nn.ELU(),skip(HSband,HSband,[HSband],[HSband],[5],5,act_fun='ELU'))
        self.FE = Extractor
        self.sf = args.sf
        self.bn = My_Bn()

        #===================Subspace



    def forward(self,x,rgb,init=True,band=31):

        # High-Level Fusion
        if init is True:
            q, k, v = get_QKVs(rgb, x, self.FE, self.sf,band=band)
            x1_ = self.up_16(self.FTB_h(k,v,q))
            # print(x1_.shjape,x.shape)
            # print('H:size',torch.concat([x1_, x]).shape)
            x2 =self.bn( self.skiph(torch.concat([x1_, x], dim=1))) + x
            # Middel-Level Fusion
            x1_ = self.up_4(self.FTB_m(k,v,q))
            x2 = self.bn(self.skipm(torch.concat([x1_, x2], dim=1))) + x2
            # Low-Level Fusion
            x1_ = self.FTB_l(k,v,q)
            x2 = self.skipl(torch.concat([x1_, x2], dim=1))
        else:
            q, k, v = get_QKVs(rgb, x, self.FE, self.sf,band=band)
            x1_ = self.up_16(self.FTB_h2(k, v, q))
            x2 = self.bn(self.skiph(torch.concat([x1_, x], dim=1))) + x
            # Middel-Level Fusion
            x1_ = self.up_4(self.FTB_m2(k, v, q))
            x2 = self.bn(self.skipm(torch.concat([x1_, x2], dim=1))) + x2
            # Low-Level Fusion
            x1_ = self.FTB_l2(k, v, q)
            x2 = self.skipl(torch.concat([x1_, x2], dim=1))
        return x2

def MultiLevelFsusionTransformer(args):
    FE = VGG16()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Multilevel_Fusion_Transformer(args,Extractor=FE)
    model = model.to(device)
    return model

