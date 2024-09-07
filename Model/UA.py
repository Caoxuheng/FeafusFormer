
from .common import *
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import torch.nn.functional as F
from torch.nn import init

vgg16 = vgg16_bn
VGG16_Weights = VGG16_BN_Weights


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
    def __init__(self,level,init=True,sz = None):
        super(FusionTransformerBlock, self).__init__()
        self.level = level
        if init is True:
            self.multiatt1 = ScaledDotProductAttentionOnly()
        else:
            i_pi = [pow(sz[0]//i,2) for i in [1,2,4]]
            n_f = [64,128,256]
   
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

        self.up_16 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=None), nn.ELU(),
                                   nn.Conv2d(256, int(2.06*args.hsi_size[-1]) * 4, 1, bias=None), nn.PixelShuffle(2), nn.ELU(),
                                   nn.Conv2d(int(2.06*args.hsi_size[-1]), int(2.06*args.hsi_size[-1]), 3, padding=1, groups=int(2.06*args.hsi_size[-1]), bias=None), nn.ELU(),
                                   nn.Conv2d(int(2.06*args.hsi_size[-1]), args.hsi_size[-1] * 4, 1, bias=None), nn.PixelShuffle(2),
                                   nn.ELU(), nn.Conv2d(args.hsi_size[-1], args.hsi_size[-1], 3, padding=1, groups= args.hsi_size[-1], bias=None),
                                   nn.Conv2d(args.hsi_size[-1], args.hsi_size[-1], 1, bias=None), nn.ELU())

        self.up_4 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=None), nn.ELU(),
                                  nn.Conv2d(128, 4 * args.hsi_size[-1], 1, bias=None), nn.PixelShuffle(2), nn.ELU(),
                                  nn.Conv2d(args.hsi_size[-1], args.hsi_size[-1], 3, padding=1, groups=args.hsi_size[-1], bias=None),
                                  nn.Conv2d(args.hsi_size[-1], args.hsi_size[-1], 1, bias=None), nn.ELU())

        self.FTB_h2 = FusionTransformerBlock(2, init=False,sz = args.hsi_size)
        self.FTB_m2=FusionTransformerBlock(1,init=False,sz = args.hsi_size)
        self.FTB_l2=FusionTransformerBlock(0,init=False,sz = args.hsi_size)

        self.FTB_h = FusionTransformerBlock(2, init=True)
        self.FTB_m = FusionTransformerBlock(1, init=True)
        self.FTB_l = FusionTransformerBlock(0, init=True)

        self.skiph = nn.Sequential(nn.Conv2d(2*args.hsi_size[-1],args.hsi_size[-1],3,padding=1,bias=None),nn.InstanceNorm2d(args.hsi_size[-1]),nn.ELU(), skip(args.hsi_size[-1],args.hsi_size[-1],[args.hsi_size[-1]],[args.hsi_size[-1]],[1],3,act_fun='ELU'))
        self.skipm = nn.Sequential(nn.Conv2d(2*args.hsi_size[-1],args.hsi_size[-1],3,padding=1,bias=None),nn.InstanceNorm2d(args.hsi_size[-1]),nn.ELU(),skip(args.hsi_size[-1],args.hsi_size[-1],[args.hsi_size[-1]],[args.hsi_size[-1]],[1],3,act_fun='ELU'))
        self.skipl = nn.Sequential(nn.Conv2d(95,args.hsi_size[-1],1,bias=None),nn.InstanceNorm2d(args.hsi_size[-1]),nn.ELU(),skip(args.hsi_size[-1],args.hsi_size[-1],[args.hsi_size[-1]],[args.hsi_size[-1]],[5],5,act_fun='ELU'))
        self.FE = Extractor
        self.sf = args.sf
        self.bn = My_Bn()
        self.band = args.hsi_size[-1]

    def forward(self,x,rgb,init=True):


        if init is True:
            # High-Level Fusion
            q, k, v = get_QKVs(rgb, x, self.FE, self.sf,band=self.band)
            x1_ = self.up_16(self.FTB_h(k,v,q))
            x2 =self.bn( self.skiph(torch.concat([x1_, x], dim=1))) + x
            # Middel-Level Fusion
            x1_ = self.up_4(self.FTB_m(k,v,q))
            x2 = self.bn(self.skipm(torch.concat([x1_, x2], dim=1))) + x2
            # Low-Level Fusion
            x1_ = self.FTB_l(k,v,q)
            x2 = self.skipl(torch.concat([x1_, x2], dim=1))
        else:

            # High-Level Fusion
            q, k, v = get_QKVs(rgb, x, self.FE, self.sf,band=self.band)
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

