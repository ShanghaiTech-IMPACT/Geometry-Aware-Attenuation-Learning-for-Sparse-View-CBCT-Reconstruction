import torch
import torch.nn as nn
import torch.nn.functional as F

class localfusor(nn.Module):
    # f_i
    def __init__(self, conf=None):
        super(localfusor, self).__init__()
        self.latent_size = conf.latent_size
        self.activation = conf.activation
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        self.input_fc =  nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(self.latent_size, 1),
            self.act
        )
        self.output_fc = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )

    def forward(self, latent):   # [nviews, C, npoints]
        feat = latent.transpose(1,2)  # [nviews, npoints, C]
        feat = self.input_fc(feat)
        weight = F.softmax(self.weight_fc(feat),dim=0)
        weighted_feat = torch.sum(feat*weight,dim=0)
        output_feat = self.output_fc(weighted_feat).transpose(0,1)
        return output_feat  # [C, npoints]

class meanfusor(nn.Module):
    # f_i cat f_mean
    def __init__(self, conf=None):
        super(meanfusor, self).__init__()
        self.latent_size = conf.latent_size
        self.activation = conf.activation
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        self.input_fc = nn.Sequential(
            nn.Linear(self.latent_size * 2 , self.latent_size),
            self.act
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(self.latent_size, 1),
            self.act
        )
        self.output_fc = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )
    
    def forward(self, latent):
        N = latent.shape[0]
        mean = torch.mean(latent,dim=0).repeat(N,1,1)
        feat = torch.cat([latent,mean],dim=1).transpose(1,2)
        global_feat = self.input_fc(feat)
        weight = F.softmax(self.weight_fc(global_feat),dim=0)
        weighted_global_feat = torch.sum(global_feat*weight,dim=0)
        output_feat = self.output_fc(weighted_global_feat).transpose(0,1)
        return output_feat

class varfusor(nn.Module):
    # f_i cat f_var
    def __init__(self, conf=None):
        super(varfusor, self).__init__()
        self.latent_size = conf.latent_size
        self.activation = conf.activation
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        self.input_fc = nn.Sequential(
            nn.Linear(self.latent_size * 2 , self.latent_size),
            self.act
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(self.latent_size, 1),
            self.act
        )
        self.output_fc = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )
    
    def forward(self, latent):
        N = latent.shape[0]
        var = torch.var(latent,dim=0).repeat(N,1,1)
        feat = torch.cat([latent,var],dim=1).transpose(1,2)
        global_feat = self.input_fc(feat)
        weight = F.softmax(self.weight_fc(global_feat),dim=0)
        weighted_global_feat = torch.sum(global_feat*weight,dim=0)
        output_feat = self.output_fc(weighted_global_feat).transpose(0,1)
        return output_feat

class adafusor(nn.Module):
    # f_i cat f_mean cat f_var
    # adaptive fusion strategy proposed in our paper
    def __init__(self, conf=None):
        super(adafusor, self).__init__()
        self.latent_size = conf.latent_size
        self.activation = conf.activation
        if self.activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            self.act = nn.GELU()
        self.input_fc =  nn.Sequential(
            nn.Linear(self.latent_size * 3, self.latent_size),
            self.act
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(self.latent_size, 1),
            self.act
        )
        self.output_fc = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            self.act
        )

    def forward(self, latent):
        N = latent.shape[0]
        mean = torch.mean(latent,dim=0).repeat(N,1,1)
        var = torch.var(latent,dim=0).repeat(N,1,1)
        feat = torch.cat([latent,mean,var],dim=1).transpose(1,2)
        global_feat = self.input_fc(feat)
        weight = F.softmax(self.weight_fc(global_feat),dim=0)
        weighted_global_feat = torch.sum(global_feat*weight,dim=0)
        output_feat = self.output_fc(weighted_global_feat).transpose(0,1)
        return output_feat