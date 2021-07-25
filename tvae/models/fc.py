from torch import nn
from torch.nn.utils import weight_norm as wn

def FC_Encoder_w_WN(s_dim, n_cin, n_hw):
    model = nn.Sequential(
                wn(nn.Conv2d(n_cin, s_dim*2,
                    kernel_size=n_hw, stride=1, padding=0)))
    return model

def FC_Decoder_w_WN(s_dim, n_cout, n_hw):
    model = nn.Sequential(
                wn(nn.ConvTranspose2d(s_dim, n_cout, 
                    kernel_size=n_hw, stride=1, padding=0))
                )
    return model

def FC_Encoder(s_dim, n_cin, n_hw):
    model = nn.Sequential(
                nn.Conv2d(n_cin, s_dim*2,
                    kernel_size=n_hw, stride=1, padding=0))
    return model

def FC_Decoder(s_dim, n_cout, n_hw):
    model = nn.Sequential(
                nn.ConvTranspose2d(s_dim, n_cout, 
                    kernel_size=n_hw, stride=1, padding=0)
                )
    return model