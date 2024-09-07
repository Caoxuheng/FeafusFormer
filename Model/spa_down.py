import torch
from torch import nn
from utils import  PSNR_GPU

class My_Bn(nn.Module):
    def __init__(self):
        super(My_Bn,self).__init__()
    def forward(self,x):
        return x - nn.AdaptiveAvgPool2d(1)(x)


def getGaussiankernel(ksize,sigma):
    x = torch.arange( -(ksize-1)//2,(ksize-1)//2+1)
    kernel = torch.exp(-1/2*torch.pow(x/sigma,2))
    kernel/=kernel.sum()
    return kernel.view(-1,1).float()

class Spatial_Degradation(nn.Module):
    def __init__(self,sf,predefine=None,batch=1):
        super(Spatial_Degradation, self).__init__()

        if predefine is not None:
            kernel = torch.tensor(predefine).float()
            stride = sf
            sf = kernel.shape[0]
        else:
            kernel = getGaussiankernel(sf,sf//1.5)
            kernel @=kernel.T
            stride = sf
        if batch ==1 :

            self.down = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=sf,stride=stride,bias=None)
            self.down.weight.data[0,0] = kernel
        self.down.requires_grad_(False)

    def forward(self,x):
        return self.down(x.transpose(1,0)).transpose(1,0)



class SpaDown_(nn.Module):
    def __init__(self,sf,predefine=None,batch=1):
        super(SpaDown_, self).__init__()
        if predefine is not  None:
            kernel =torch.tensor( predefine).float()


        else:
            kernel = getGaussiankernel(sf-1,sf//2)
            kernel @=kernel.T
        if batch ==1 :
            self.down = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=sf-1,stride=sf,bias=None)
            self.down.weight.data[0,0] = kernel
            self.mean_bn = My_Bn()
            self.spadown = self.down
        self.avgpool = nn.AvgPool2d(kernel_size=sf,stride=sf)

        self.sf = sf
        self.act = nn.Tanh()

    def forward(self,x):
        x = x.transpose(1,0)
        x1_down =self.mean_bn(self.act(self.spadown(x)))
        x1_avg = self.avgpool(x)
        return (x1_avg+x1_down).transpose(1,0)


class SpaDown(nn.Module):
    def __init__(self,sf,predefine=False,iscal=False):
        super(SpaDown, self).__init__()
        if iscal:
            self.net = SpaDown_(sf,predefine)
        else:
            self.net = Spatial_Degradation(sf,predefine)
    def forward(self,x):
        return self.net(x)


def initialize_SpaDNet_test(module,msi,msi2):
    # 2022年7月26日
    trainer = torch.optim.Adam(params=module.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(trainer,50,0.5)
    losses = []
    for epoch in range(500):
        trainer.zero_grad()
        down_img = module(msi)

        l = -PSNR_GPU(down_img, msi2)
        l.backward()
        trainer.step()
        sched.step()
        losses.append(l.detach().numpy())
    print('Initialized results')

    print(PSNR_GPU(down_img, msi2))
    print(trainer.state_dict()['param_groups'][0]['lr'])



def initialize_SpaDNet(module,msi,msi2):
    # 2022年7月26日
    msi2 = msi2.detach()
    trainer = torch.optim.Adam(params=module.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(trainer,50,0.5)
    losses = []
    for epoch in range(500):
        trainer.zero_grad()
        down_img = module(msi)

        l = -PSNR_GPU(down_img, msi2)
        l.backward()
        trainer.step()
        sched.step()
        losses.append(l.detach().cpu().numpy())
    print('Initialize Spatial Degradation Net Successfully. Epoch:{}\tlr:{:.2e}\tPSNR:{:.2f}'.format(epoch,
                                                                                                      trainer.state_dict()[
                                                                                                          'param_groups'][
                                                                                                          0]['lr'],
                                                                                                      -l.detach().cpu().numpy()))

    return module



if __name__ =='__main__':
    ##------------------------Test getGaussianKernel----------------------------------
    # import cv2
    # cv_kernel = cv2.getGaussianKernel(15,1.5)
    # torch_kernel = getGaussiankernel(15,1.5)
    # print(True if (cv_kernel-torch_kernel.numpy()).mean()<1e-7 else False )


    ##------------------------Test Spatial_Degradation-----------------------------------------
    # downnet = Spatial_Degradation(32)
    # img = torch.rand(1,3,512,512)
    # down_img = downnet(img)


    ##------------------------Test SpaDown-----------------------------------------
    import scipy.io as sio
    import imgvision as iv
    from utils import tensor2array,array2tensor
    kernels = sio.loadmat('G:/FusionBasedHSISR/UMGAL/Kernel/motion_kernels.mat')['f_set']
    # spdd = Spatial_Degradation(32,predefine=kernels[0,9])
    spdd = Spatial_Degradation(32, predefine=None)
    downnet = SpaDown(32,iscal=True)
    #
    # downnet =Spatial_downsample(32)
    img = sio.loadmat('F:/CAVE/3.mat')['HSI']
    # img = torch.rand(1,3,512,512)
    im = array2tensor(img)
    MSI = array2tensor(iv.spectra().space(img,'nkd700'))
    lr = spdd(im)

    lrmsi = array2tensor(iv.spectra().space(tensor2array(lr),'nkd700'))
    initialize_SpaDNet(downnet.cuda(),MSI.cuda(),lrmsi.cuda())
