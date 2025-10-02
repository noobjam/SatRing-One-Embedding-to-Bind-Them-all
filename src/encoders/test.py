import torch
from l8_9_sncoder import EncoderL8
from s2_encoder import S2Encoder

dummy_l8 = torch.randn(1, 11, 224, 224)
dummy_s2 = torch.randn(1, 13, 224, 224)

l8_encoder = EncoderL8(in_channels=11)
s2_encoder = S2Encoder(in_channels=13)

z_l8 = l8_encoder(dummy_l8)
z_s2 = s2_encoder(dummy_s2)

print(z_l8.shape, z_s2.shape)
