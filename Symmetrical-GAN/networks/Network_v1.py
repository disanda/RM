# ----------自己写的一套DCGAN网络，通过图像分辨率, Gscale, Dscale4G, 模型的输出输出维度(通道数):input_dim, output_dim 调整网络参数规模--------
# 测试网络规模:
import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math
import sys 
sys.path.append('../')
import networks.fc_modules

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_para_GByte(parameter_number):
     x=parameter_number['Total']*8/1024/1024/1024
     y=parameter_number['Total']*8/1024/1024/1024
     return {'Total_GB': x, 'Trainable_BG': y}

class G(nn.Module): #Generator
    def __init__(self, input_dim=256, output_dim=3, image_size=256, Gscale=8,  hidden_scale = 2): # output_dim = image_channels
        super().__init__()
        layers = []
        up_times = math.log(image_size,2)- 3 # 输入为4*4时,another_times=1
        first_hidden_dim = image_size*Gscale # 这里对应输入维度，表示《输入维度》对应《网络中间层维度（起点）》的放大倍数
        bias_flag = False

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),#[1024,128*8*8]-input_size=32
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
        )

        # first conv layer
        layers.append(nn.ConvTranspose2d(input_dim, first_hidden_dim, kernel_size=4,stride=1,padding=0,bias=bias_flag)) # 1*1 input -> 4*4
        layers.append(nn.InstanceNorm2d(first_hidden_dim, affine=False, eps=1e-8))
        layers.append(nn.ReLU())

        # 2: upsamplings, (1x1) -> 4x4 -> 8x8 -> 16x16 -> 32*32 -> 64 -> 128 -> 256
        hidden_dim = first_hidden_dim
        while up_times>0:
            layers.append(nn.ConvTranspose2d(hidden_dim, int(hidden_dim/hidden_scale), kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
            layers.append(nn.InstanceNorm2d(int(hidden_dim/hidden_scale), affine=False, eps=1e-8))
            layers.append(nn.ReLU())
            up_times = up_times - 1
            hidden_dim = hidden_dim // 2

        # 3:end
        layers.append(nn.ConvTranspose2d(hidden_dim,output_dim,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers) #all

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.shape[0],z.shape[1],1,1)
        y = self.net(x)
        return y

class D(nn.Module): # Discriminator with SpectrualNorm, GDscale网络的参数规模，Dscale4G网络的缩小倍数
    def __init__(self, output_dim=256, input_dim=3, image_size=256, GDscale=8, Dscale4G=1, hidden_scale = 2): #新版的GDscale是D中G的倍数(G输入首层特征即D最后输出的隐藏特征,默认和Gscale一样)，Dscale4G是相对G缩小的倍数
        super().__init__()
        layers=[]
        up_times = math.log(image_size,2)- 3
        first_hidden_dim = image_size * GDscale // (2**int(up_times) * Dscale4G) # 默认为input_dim , Dscale4G是D相对G缩小的倍数
        bias_flag = False

        # 1:
        layers.append(spectral_norm(nn.Conv2d(input_dim, first_hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        hidden_dim = first_hidden_dim
        while up_times>0:
            layers.append(spectral_norm(nn.Conv2d(hidden_dim, int(hidden_dim*hidden_scale), kernel_size=4, stride=2, padding=1, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            hidden_dim = hidden_dim * 2
            up_times = up_times - 1

        layers.append(nn.Conv2d(hidden_dim, output_dim, kernel_size=4, stride=1, padding=0, bias=bias_flag))


        self.net = nn.Sequential(*layers)

        self.fc = nn.Sequential(
                nn.Linear(output_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2, inplace=True)
                nn.Linear(1024, output_dim),#[1024,128*8*8]-input_size=32
                #nn.BatchNorm1d(output_dim),
                #nn.Sigmoid(),
            )

    def forward(self, x):
        y = self.net(x) # [1,1,1,1]
        y = y.view(y.shape[0],-1)
        z = self.fc(y)
        return z

# # test
net_D = D()
#print(net_D)
net_G = G()
#print(net_G)
x = torch.randn(10,3,256,256)
z = net_D(x)
print(z.shape)
y = net_G(z)
print(y.shape)

