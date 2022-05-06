import torch
import torch.nn as nn

class MiniBatchSTDLayer(torch.nn.Module):
    """Implements the minibatch standard deviation layer."""

    def __init__(self, group_size=15, epsilon=1e-8):
        super().__init__()
        self.group_size = group_size
        self.epsilon = epsilon

    def forward(self, x):
        if self.group_size <= 1:
            return x
        group_size = min(self.group_size, x.shape[0])                  # [NCHW]
        y = x.view(group_size, -1, x.shape[1], x.shape[2], x.shape[3]) # [GMCHW]
        y = y - torch.mean(y, dim=0, keepdim=True)                     # [GMCHW]
        y = torch.mean(y ** 2, dim=0)                                  # [MCHW]
        y = torch.sqrt(y + self.epsilon)                               # [MCHW]
        y = torch.mean(y, dim=[1, 2, 3], keepdim=True)                 # [M111]
        y = y.repeat(group_size, 1, x.shape[2], x.shape[3])            # [N1HW]
        return torch.cat([x, y], dim=1)

# https://github.com/heykeetae/Self-Attention-GAN
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention


class selfattention(nn.Module):
    def __init__(self, in_channels, size):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1) # input: B, C, H, W -> q: B, H * W, C // 8
        k = self.key(x).view(batch_size, -1, height * width) #input: B, C, H, W -> k: B, C // 8, H * W
        v = self.value(x).view(batch_size, -1, height * width) #input: B, C, H, W -> v: B, C, H * W         #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        
        attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix. B, WH * WH
        attn_matrix = self.softmax(attn_matrix)#经过一个softmax进行缩放权重大小. 

        #out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        #out = out.view(*x.shape) 
        #return self.gamma * out + x, attn_matrix
        return attn_matrix

# Linear+Conv ATT
class selfattention_v2(nn.Module):
    def __init__(self, in_channels, size):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Linear(in_channels*size*size, in_channels*size*size//8)
        self.key   = nn.Linear(in_channels*size*size, in_channels*size*size//8)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x_linear = x.view(batch_size,-1)
        q = self.query(x_linear).view(batch_size, -1, height * width).permute(0, 2, 1) # input: B, C, H, W -> q: B, H * W, C // 8
        k = self.key(x_linear).view(batch_size, -1, height * width) #input: B, C, H, W -> k: B, C // 8, H * W
        v = self.value(x).view(batch_size, -1, height * width) #input: B, C, H, W -> v: B, C, H * W         #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        
        attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix. B, WH * WH
        attn_matrix = self.softmax(attn_matrix)#经过一个softmax进行缩放权重大小. 
        
        #out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        #out = out.view(*x.shape)
        #return self.gamma * out + x, attn_matrix
        return attn_matrix

# Linear ATT
class selfattention_v3(nn.Module):
    def __init__(self, in_channels,size):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Linear(in_channels*size*size, in_channels*size*size//8)
        self.key   = nn.Linear(in_channels*size*size, in_channels*size*size//8)
        self.value = nn.Linear(in_channels*size*size, in_channels*size*size)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        x_linear = x.view(batch_size,-1)
        q = self.query(x_linear).view(batch_size, -1, height * width).permute(0, 2, 1) # input: B, C, H, W -> q: B, H * W, C // 8
        k = self.key(x_linear).view(batch_size, -1, height * width) #input: B, C, H, W -> k: B, C // 8, H * W
        v = self.value(x_linear).view(batch_size, -1, height * width) #input: B, C, H, W -> v: B, C, H * W         #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        
        attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix. B, WH * WH
        attn_matrix = self.softmax(attn_matrix)#经过一个softmax进行缩放权重大小. 
        
        #out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        #out = out.view(*x.shape)
        #return self.gamma * out + x, attn_matrix
        return attn_matrix

# # #test
# at = selfattention_v3(64,16)
# z  =  torch.randn(5,64,16,16)
# x,am  = at(z)
# print(x.shape) # (5,64,16,16)
# print(am.shape) # (5, 16*16, 16*16)