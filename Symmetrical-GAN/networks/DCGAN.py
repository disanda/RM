# ----------自己写的一套DCGAN网络，通过图像分辨率, Gscale, Dscale4G, 模型的输出输出维度(通道数):input_dim, output_dim 调整网络参数规模--------
# 测试网络规模:
import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math
import fc_modules
import sys 
sys.path.append('../')
import lreq as ln

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_para_GByte(parameter_number):
     x=parameter_number['Total']*8/1024/1024/1024
     y=parameter_number['Total']*8/1024/1024/1024
     return {'Total_GB': x, 'Trainable_BG': y}

class G(nn.Module): #Generator
    def __init__(self, input_dim=128, output_dim=3, image_size=256, Gscale=8,  hidden_scale = 2, BN = False, relu = False, elr = True): # output_dim = image_channels
        super().__init__()
        layers = []
        up_times = math.log(image_size,2)- 3 # 输入为4*4时,another_times=1
        first_hidden_dim = image_size*Gscale # 这里对应输入维度，表示《输入维度》对应《网络中间层维度（起点）》的放大倍数
        bias_flag = False

        # 1: 1x1 -> 4x4
        if elr == False:
            layers.append(nn.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=1,padding=0,bias=bias_flag)) # 1*1 input -> 4*4
        else:
            layers.append(ln.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=1,padding=0,bias=bias_flag))
        if BN == True:
            layers.append(nn.BatchNorm2d(first_hidden_dim))
        else:
            layers.append(nn.InstanceNorm2d(first_hidden_dim, affine=False, eps=1e-8))
        if relu == False:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            layers.append(nn.ReLU())

        # 2: upsamplings, (1x1) -> 4x4 -> 8x8 -> 16x16 -> 32*32 -> 64 -> 128 -> 256
        hidden_dim = first_hidden_dim
        while up_times>0:
            if elr == False:
                layers.append(nn.ConvTranspose2d(hidden_dim, int(hidden_dim/hidden_scale), kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
            else:
                layers.append(ln.ConvTranspose2d(hidden_dim, int(hidden_dim/hidden_scale), kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
            
            if BN == True:
                layers.append(nn.BatchNorm2d(int(hidden_dim/hidden_scale)))
            else:
                layers.append(nn.InstanceNorm2d(int(hidden_dim/hidden_scale), affine=False, eps=1e-8))
            if relu == False:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.ReLU())

            up_times = up_times - 1
            hidden_dim = hidden_dim // 2

        # 3:end
        if elr == False: 
            layers.append(nn.ConvTranspose2d(hidden_dim,output_dim,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        else:
            layers.append(ln.ConvTranspose2d(hidden_dim,output_dim,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())

        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x

class D(nn.Module): # Discriminator with SpectrualNorm, GDscale网络的参数规模，Dscale4G网络的缩小倍数
    def __init__(self, output_dim=256, input_dim=3, image_size=256, GDscale=8, Dscale4G=1, hidden_scale = 2, MB_STD = False, elr = False, Final_Linear='Linear'): #新版的GDscale是D中G的倍数(G输入首层特征即D最后输出的隐藏特征,默认和Gscale一样)，Dscale4G是相对G缩小的倍数
        super().__init__()
        layers=[]
        up_times = math.log(image_size,2)- 3
        first_hidden_dim = image_size * GDscale // (2**int(up_times) * Dscale4G) # 默认为input_dim , Dscale4G是D相对G缩小的倍数
        bias_flag = False
        self.MB_STD = MB_STD 
        self.Final_Linear = Final_Linear

        # 1:
        if elr == False:
            layers.append(spectral_norm(nn.Conv2d(input_dim, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        else:
            layers.append(spectral_norm(ln.Conv2d(input_dim, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        hidden_dim = first_hidden_dim
        while up_times>0:
            if elr == False:  
                layers.append(spectral_norm(nn.Conv2d(hidden_dim, int(hidden_dim*hidden_scale), kernel_size=4, stride=2, padding=1, bias=bias_flag)))
            else:
                layers.append(spectral_norm(ln.Conv2d(hidden_dim, int(hidden_dim*hidden_scale), kernel_size=4, stride=2, padding=1, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            hidden_dim = hidden_dim * 2
            up_times = up_times - 1

        # final layer
        if Final_Linear == 'Linear':
            if MB_STD == True: 
                self.mb_layer = fc_modules.MiniBatchSTDLayer()
                if elr == False:
                    self.last_layer = nn.Linear((hidden_dim+1)*4*4, output_dim, bias=bias_flag)
                else:
                    self.last_layer = ln.Linear((hidden_dim+1)*4*4, output_dim, bias=bias_flag)
            else:
                if elr == False:
                    self.last_layer = nn.Linear(hidden_dim*4*4, output_dim, bias=bias_flag)
                else:
                    self.last_layer = ln.Linear(hidden_dim*4*4, output_dim, bias=bias_flag)
        elif Final_Linear == 'Attn': #虚手动设置不同版本
            #self.last_layer = fc_modules.selfattention(2048,4) # in [n,2048,4,4]:  out [n,16,16]
            #self.last_layer = fc_modules.selfattention_v2(2048,4)
            self.last_layer = fc_modules.selfattention_v3(2048,4)
        else:
            if elr == False:
                layers.append(nn.Conv2d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=0, bias=bias_flag))
            else:
                layers.append(ln.Conv2d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=0, bias=bias_flag))

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x) # [1,1,1,1]
        #y = y.mean()
        #print(y.shape)
        if self.MB_STD == True:
            y = self.mb_layer(y)
        if self.Final_Linear  == 'Linear':
            y = y.view(y.shape[0],-1)
            y = self.last_layer(y)
        if self.Final_Linear  == 'Attn':
            y = self.last_layer(y)
            y = y.view(y.shape[0],-1)
        return y

# # test
net_D = D()
#print(net_D)
net_G = G()
#print(net_G)
x = torch.randn(10,3,256,256)
z = net_D(x)
print(z.shape)
#y = net_G(z.view(z.shape[0],z.shape[1],1,1))
#print(y.shape)

