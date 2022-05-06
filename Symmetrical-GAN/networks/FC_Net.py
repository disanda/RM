# ----------自己写的一套DCGAN网络，通过图像分辨率, Gscale, Dscale4G, 模型的输出输出维度(通道数):input_dim, output_dim 调整网络参数规模--------
# 测试网络规模:
import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math
import sys 
sys.path.append('../')
import networks.lreq as ln

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_para_GByte(parameter_number):
     x=parameter_number['Total']*8/1024/1024/1024
     y=parameter_number['Total']*8/1024/1024/1024
     return {'Total_GB': x, 'Trainable_BG': y}

class D(torch.nn.Module):
    def __init__(self, in_nodes=32*32, out_nodes=256, d = 2):
        super().__init__()
        bias_f = True # default: True
        lrmul_f = 1 # default: 1
        self.block1 = ln.Linear( in_nodes,  in_nodes//d, bias=bias_f,  lrmul=lrmul_f) # torch.nn.Linear(in_nodes,512*args.d)
        self.block2 = ln.Linear( in_nodes//d,     in_nodes//d, bias=bias_f,  lrmul=lrmul_f)
#        self.block3 = ln.Linear( 512*d,     512*d, bias=bias_f,  lrmul=lrmul_f)
#         self.block4 = ln.Linear( 512*d,     512*d, bias=bias_f,  lrmul=lrmul_f)
#         self.block5 = ln.Linear( 512*d,     512*d, bias=bias_f,  lrmul=lrmul_f)
#         self.block6 = ln.Linear( 512*d,     512*d, bias=bias_f,  lrmul=lrmul_f)
#         self.block7 = ln.Linear( 512*d,     512*d, bias=bias_f,  lrmul=lrmul_f)
#         self.block8 = ln.Linear( 512*d,     512*d, bias=bias_f,  lrmul=lrmul_f)
#         self.block9 = ln.Linear( 512*d,     512*d, bias=bias_f,  lrmul=lrmul_f)
        # self.block10 = ln.Linear(512*args.d,  512*args.d, bias=bias_f,  lrmul=lrmul_f)
        # self.block11 = ln.Linear(512*args.d,  512*args.d, bias=bias_f,  lrmul=lrmul_f)
        # self.block12 = ln.Linear(512*args.d,  512*args.d, bias=bias_f,  lrmul=lrmul_f)
        self.block_f = ln.Linear( in_nodes//d,  out_nodes, bias=bias_f,  lrmul=lrmul_f)
        self.ac = torch.nn.ReLU() # ELU, ReLU
        #self.ac2 = torch.nn.Sigmoid()
    def forward(self, w):
        x = self.block1(w)
        x = self.ac(x)
        x = self.block2(x)
        x = self.ac(x)
#         x = self.block3(x)
#         x = self.ac(x)
#         x = self.block4(x)
#         x = self.ac(x)
#         x = self.block5(x)
#         x = self.ac(x)
#         x = self.block6(x)
#         x = self.ac(x)
#         x = self.block7(x)
#         x = self.ac(x)
#         x = self.block8(x)
#         x = self.ac(x)
#         x = self.block9(x)
#         x = self.ac(x)
#         x = self.block10(x)
#         x = self.ac(x)
#         x = self.block11(x)
#         x = self.ac(x)
#         x = self.block12(x)
#         x = self.ac(x)
        x = self.block_f(x)
        #x = self.ac2(x) #[0,1]
        return x

class G(torch.nn.Module):
    def __init__(self, in_nodes=256, out_nodes=32*32, d = 2):
        super().__init__()
        bias_f = True # default: True
        lrmul_f = 1 # default: 1
        self.block1 = ln.Linear( in_nodes,  out_nodes//d, bias=bias_f,  lrmul=lrmul_f) # torch.nn.Linear(in_nodes,512*args.d)
        self.block2 = ln.Linear( out_nodes//d,  out_nodes//d, bias=bias_f,  lrmul=lrmul_f)
#        self.block3 = ln.Linear( 512*d*14,  512*d*12, bias=bias_f,  lrmul=lrmul_f)
#         self.block4 = ln.Linear( 512*d*12,  512*d*10, bias=bias_f,  lrmul=lrmul_f)
#         self.block5 = ln.Linear( 512*d*10,  512*d*8, bias=bias_f,  lrmul=lrmul_f)
#         self.block6 = ln.Linear( 512*d*8,   512*d*6, bias=bias_f,  lrmul=lrmul_f)
#         self.block7 = ln.Linear( 512*d*6,   512*d*4, bias=bias_f,  lrmul=lrmul_f)
#         self.block8 = ln.Linear( 512*d*4,   512*d*1, bias=bias_f,  lrmul=lrmul_f)
#        self.block9 = ln.Linear( 512*d*9,   512*d*10, bias=bias_f,  lrmul=lrmul_f)
#         self.block10 = ln.Linear(512*d*10,  512*d*11, bias=bias_f,  lrmul=lrmul_f)
#         self.block11 = ln.Linear(512*d*11,  512*d*12, bias=bias_f,  lrmul=lrmul_f)
#         self.block12 = ln.Linear(512*d*12,  512*d*13, bias=bias_f,  lrmul=lrmul_f)
        self.block_f = ln.Linear(out_nodes//d,  out_nodes, bias=bias_f,  lrmul=lrmul_f)
        self.ac = torch.nn.ReLU() # ELU, ReLU
        #self.ac2 = torch.nn.Sigmoid()
    def forward(self, w):
        x = self.block1(w)
        x = self.ac(x)
        x = self.block2(x)
        x = self.ac(x)
#         x = self.block3(x)
#         x = self.ac(x)
#         x = self.block4(x)
#         x = self.ac(x)
#         x = self.block5(x)
#         x = self.ac(x)
#         x = self.block6(x)
#         x = self.ac(x)
#         x = self.block7(x)
#         x = self.ac(x)
#         x = self.block8(x)
#         x = self.ac(x)
#         x = self.block9(x)
#         x = self.ac(x)
#         x = self.block10(x)
#         x = self.ac(x)
#         x = self.block11(x)
#         x = self.ac(x)
#         x = self.block12(x)
#         x = self.ac(x)
        x = self.block_f(x)
        #x = self.ac2(x) #[0,1]
        return x

# # test
net_D = D()
#print(net_D)
net_G = G()
#print(net_G)
x = torch.randn(10,32,32).view(10,-1)
z = net_D(x)
print(z.shape)
#y = net_G(z.view(z.shape[0],z.shape[1],1,1))
#print(y.shape)

