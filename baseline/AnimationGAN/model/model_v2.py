# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import loss_norm_gp
import torch.nn.utils.spectral_norm as spectral_norm
#最稳定版本！

#-----------------MWM-GAN-v1--------------------多一个网络Q输出C即可
class generator_mwm(nn.Module):
    def __init__(self, z_dim=100, output_channel=1, input_size=64, len_discrete_code=10, len_continuous_code=2):
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_channel
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.len_discrete_code + self.len_continuous_code, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 8) * (self.input_size // 8)),#[1024,128*8*8]-input_size=32
            nn.BatchNorm1d(128 * (self.input_size // 8) * (self.input_size // 8)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input, dist_code, cont_code):
        x = torch.cat([input, dist_code, cont_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 8), (self.input_size // 8))#[-1,128,8,8]
        x = self.deconv(x)
        return x

class discriminator_mwm(nn.Module):
    # 输入是图片，输出是按照参数分为 [-1, output_dim] , [-1, len_continuous_code] , [-1 , len_continuous_code]
    def __init__(self, input_channel=1, output_dim=1, input_size=64, len_discrete_code=10, len_continuous_code=2, sp_norm=False):
        super().__init__()
        self.input_dim = input_channel
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        if sp_norm == False:
            self.conv = nn.Sequential(
                nn.Conv2d(self.input_dim, 32, 4, 2, 1),#input_size/2
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),#input_size/4
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),#input_size/8
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * (self.input_size // 8) * (self.input_size // 8), 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
                #nn.BatchNorm1d(self.output_dim + self.len_continuous_code + self.len_discrete_code),
                #nn.LeakyReLU(0.2),
                # nn.Sigmoid(),
            )
        else:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(self.input_dim, 32, 4, 2, 1)),#input_size/2
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),#input_size/4
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),#input_size/8
                nn.LeakyReLU(0.2),
            )
            self.fc = nn.Sequential(
                spectral_norm(nn.Linear(128 * (self.input_size // 8) * (self.input_size // 8), 1024)),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
                #nn.BatchNorm1d(self.output_dim + self.len_continuous_code + self.len_discrete_code),
                #nn.LeakyReLU(0.2),
                # nn.Sigmoid(),
            )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 8) * (self.input_size // 8)) 
        x = self.fc(x)
        a = torch.sigmoid(x[:, self.output_dim])
        b = x[:, self.output_dim:self.output_dim + self.len_discrete_code]
        c = x[:, self.output_dim + self.len_discrete_code:]
        return a, b, c

#-----------------MWM-GAN-v2--------------------全部为fc层
class generator_mwm_2fc(nn.Module):
    def __init__(self, z_mdim=100, img_channel=1, input_size=64, len_discrete_code=10, len_continuous_code=2):
        super().__init__()
        self.z_dim = z_mdim
        self.img_channel = img_channel
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.len_discrete_code + self.len_continuous_code + 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(1024, 2048),
            # nn.BatchNorm1d(2048),
            # nn.ReLU(),
            nn.Linear(1024, self.input_size*self.input_size),#[1024,128*8*8]-input_size=32
            #nn.BatchNorm1d(self.input_size*self.input_size),
            #nn.ReLU(),
            nn.Tanh()
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, flag, dist_code, multi_dist_code, cont_code):
        x = torch.cat([flag, dist_code, multi_dist_code , cont_code], 1)
        x = self.fc(x)
        x = x.view(-1, self.img_channel, self.input_size, self.input_size)#[-1,128,8,8]
        return x

class discriminator_mwm_2fc(nn.Module):
    # 输入是图片，输出是按照参数分为 [-1, output_dim] , [-1, len_continuous_code] , [-1 , len_continuous_code]
    def __init__(self, z_mdim=1, img_channel=1, input_size=64, len_discrete_code=10, len_continuous_code=2, sp_norm=True):
        super().__init__()
        self.z_mdim = z_mdim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(self.input_size*self.input_size, 1024)),
            nn.LeakyReLU(0.2),
            # spectral_norm(nn.Linear(1024, 2048)),
            # nn.LeakyReLU(0.2),
            nn.Linear(1024, self.z_mdim + self.len_continuous_code + self.len_discrete_code + 1),
            #nn.BatchNorm1d(self.output_dim + self.len_continuous_code + self.len_discrete_code),
            #nn.LeakyReLU(0.2),
            # nn.Sigmoid(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input_img):
        #x = self.conv(input)
        x = input_img.view(-1, self.input_size *self.input_size) 
        x = self.fc(x)
        flag = x[:, 0]
        dist_code = x[:,1:self.len_discrete_code+1] # torch.sigmoid() # single_label
        multi_dist_code = x[:, 1+self.len_discrete_code:1+self.len_discrete_code + self.z_mdim] # multi_labels, torch.sigmoid
        len_continuous_code = x[:, 1+self.z_mdim + self.len_discrete_code:] # continue_label
        return flag, dist_code, multi_dist_code, len_continuous_code


class generator_mwm_3fc(nn.Module):
    def __init__(self, z_mdim=100, img_channel=1, input_size=64, len_discrete_code=10, len_continuous_code=2):
        super().__init__()
        self.z_dim = z_mdim
        self.img_channel = img_channel
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.len_discrete_code + self.len_continuous_code + 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.input_size*self.input_size),#[1024,128*8*8]-input_size=32
            #nn.BatchNorm1d(self.input_size*self.input_size),
            #nn.ReLU(),
            nn.Tanh()
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, flag, dist_code, multi_dist_code, cont_code):
        x = torch.cat([flag, dist_code, multi_dist_code , cont_code], 1)
        x = self.fc(x)
        x = x.view(-1, self.img_channel, self.input_size, self.input_size)#[-1,128,8,8]
        return x

class discriminator_mwm_3fc(nn.Module):
    # 输入是图片，输出是按照参数分为 [-1, output_dim] , [-1, len_continuous_code] , [-1 , len_continuous_code]
    def __init__(self, z_mdim=1, img_channel=1, input_size=64, len_discrete_code=10, len_continuous_code=2, sp_norm=True):
        super().__init__()
        self.z_mdim = z_mdim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(self.input_size*self.input_size, 1024)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(1024, 2048)),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, self.z_mdim + self.len_continuous_code + self.len_discrete_code + 1),
            #nn.BatchNorm1d(self.output_dim + self.len_continuous_code + self.len_discrete_code),
            #nn.LeakyReLU(0.2),
            # nn.Sigmoid(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input_img):
        #x = self.conv(input)
        x = input_img.view(-1, self.input_size *self.input_size) 
        x = self.fc(x)
        flag = x[:, 0]
        dist_code = x[:,1:self.len_discrete_code+1] # torch.sigmoid() # single_label
        multi_dist_code = x[:, 1+self.len_discrete_code:1+self.len_discrete_code + self.z_mdim] # multi_labels, torch.sigmoid
        len_continuous_code = x[:, 1+self.z_mdim + self.len_discrete_code:] # continue_label
        return flag, dist_code, multi_dist_code, len_continuous_code

#-----------------MWM-GAN-v3--------------------都一组变量 multi-labels
class generator_mwm_v3(nn.Module):
    def __init__(self, z_dim=100, output_channel=1, input_size=64, len_discrete_code=10, len_continuous_code=2, len_multi_discrete_code=0):
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_channel
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.len_multi_discrete_code = len_multi_discrete_code
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.len_discrete_code + self.len_continuous_code + self.len_multi_discrete_code, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 8) * (self.input_size // 8)),#[1024,128*8*8]-input_size=32
            nn.BatchNorm1d(128 * (self.input_size // 8) * (self.input_size // 8)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input_z, dist_code, cont_code, multi_dist_code):
        x = torch.cat([input_z, dist_code, cont_code, multi_dist_codes], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 8), (self.input_size // 8))#[-1,128,8,8]
        x = self.deconv(x)
        return x

class discriminator_mwm_v3(nn.Module):
    # 输入是图片，输出是按照参数分为 [-1, output_dim] , [-1, len_continuous_code] , [-1 , len_continuous_code]
    def __init__(self, input_channel=1, output_dim=1, input_size=64, len_discrete_code=10, len_continuous_code=2, len_multi_discrete_code=0):
        super().__init__()
        self.input_dim = input_channel
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.len_multi_discrete_code = len_multi_discrete_code
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 4, 2, 1),#input_size/2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),#input_size/4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),#input_size/8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 8) * (self.input_size // 8), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code + self.len_multi_discrete_code),
            #nn.BatchNorm1d(self.output_dim + self.len_continuous_code + self.len_discrete_code),
            #nn.LeakyReLU(0.2),
            #nn.Sigmoid(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 8) * (self.input_size // 8)) 
        x = self.fc(x)
        a = torch.sigmoid(x[:, self.output_dim])
        b = x[:, self.output_dim:self.output_dim + self.len_discrete_code]
        c = x[:, self.output_dim + self.len_discrete_code:self.len_multi_discrete_code]
        d = torch.sigmoid(x[:, self.output_dim + self.len_discrete_code + self.len_multi_discrete_code:])
        return a, b, c, d




