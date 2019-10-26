import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim, activation_dim):
        super(Concat_embed, self).__init__()
        self.act_dim = activation_dim
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(self.act_dim, self.act_dim, 1, 1).permute(2,  3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat

class generator(nn.Module):
	def __init__(self):
		super(generator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.noise_dim = 100
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.latent_dim = self.noise_dim + self.projected_embed_dim
		self.ngf = 64

		self.projection = nn.Sequential(
			nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)

		# based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
		self.netG1 = nn.Sequential(
			nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.ngf * 8),
			nn.ReLU(True))
		self.p1 = Concat_embed(self.embed_dim, self.projected_embed_dim, 4)
		self.netG2 = nn.Sequential(
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(self.ngf * 8+self.projected_embed_dim, self.ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(True))
		self.p2 = Concat_embed(self.embed_dim, self.projected_embed_dim, 8)
		self.netG3 = nn.Sequential(
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(self.ngf * 4+self.projected_embed_dim, self.ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 2),
			nn.ReLU(True))
		self.p3 = Concat_embed(self.embed_dim, self.projected_embed_dim, 16)
		self.netG4 = nn.Sequential(		
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(self.ngf * 2 + self.projected_embed_dim,self.ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(True))
		self.p4 = Concat_embed(self.embed_dim, 64, 32)
		self.netG5 = nn.Sequential(
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(self.ngf+64, self.num_channels, 4, 2, 1, bias=False),
			nn.Tanh()
			 # state size. (num_channels) x 64 x 64
			)


	def forward(self, embed_vector, z):

		projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
		latent_vector = torch.cat([projected_embed, z], 1)
		g1 = self.netG1(latent_vector)
		p1 = self.p1(g1, embed_vector)
		g2 = self.netG2(p1)
		p2 = self.p2(g2, embed_vector)
		g3 = self.netG3(p2)
		p3 = self.p3(g3, embed_vector)
		g4 = self.netG4(p3)
		p4 = self.p4(g4, embed_vector)

		output = self.netG5(p4)

		return output

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16

		self.netD_1 = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
		)

		# adding 128 layers of embedded data where each layer is of size 4 x 4 with same value repeated everywhere
		self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim, 4)

		self.netD_2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			)	

	def forward(self, inp, embed):
		x_intermediate = self.netD_1(inp)
		x = self.projector(x_intermediate, embed)
		x = self.netD_2(x)

		return x.view(-1, 1).squeeze(1) , x_intermediate