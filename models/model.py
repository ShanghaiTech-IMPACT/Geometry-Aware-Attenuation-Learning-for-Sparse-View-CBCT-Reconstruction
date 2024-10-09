import torch
import torch.nn as nn
from models.ResEncoder import ResEncoder
from models.SRGAN import generator
from models.aggregator import adafusor, localfusor, meanfusor, varfusor

# Main Model
class model(nn.Module):
    def __init__(self, model_conf=None, device=None):
        super(model, self).__init__()
        self.device = device
        self.encoder_conf = model_conf['encoder']
        self.decoder_conf = model_conf['SRGAN.generator']
        self.last_layer = model_conf['last_layer']
        self.fusion = model_conf['fusion']
        self.encoder = ResEncoder(self.encoder_conf).to(device)
        self.decoder = generator(self.decoder_conf).to(device)

        self.aggregator_conf = model_conf['aggregator']
        if self.fusion == 'local':
            self.aggregator = localfusor(self.aggregator_conf).to(device)
        if self.fusion == 'meanmlp':
            self.aggregator = meanfusor(self.aggregator_conf).to(device)
        if self.fusion == 'varmlp':
            self.aggregator = varfusor(self.aggregator_conf).to(device)
        if self.fusion == 'ada':
            self.aggregator = adafusor(self.aggregator_conf).to(device)

        if self.last_layer.act == 'ReLU':
            self.last_layer_act = nn.ReLU(inplace=True)
        elif self.last_layer.act == 'GELU':
            self.last_layer_act = nn.GELU()

    def forward(self, xyz_world):
        x,y,z = xyz_world.shape[:3]
        points = xyz_world.contiguous().reshape(-1,3)
        pnts_split = torch.split(points,100000)
        h = []
        for pnts in pnts_split:
            latent = self.encoder.queryfeature(pnts)
            if self.fusion=='max':
                h.append(torch.max(latent, dim=0)[0])
            elif self.fusion=='mean':
                h.append(torch.mean(latent, dim=0))
            else:
                h.append(self.aggregator(latent))
            
        h = torch.cat(h,dim=1)
        outputs = h.reshape(1,-1,x,y,z)
        outputs = self.decoder(outputs)[0,0,:,:,:].transpose(0,2)  # align with ITK-SNAP display format
        outputs = self.last_layer_act(outputs)
        return outputs
