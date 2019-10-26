"""
Project: ResGAN
Owner:Group 6 @EE6934 NUS
Description: Implement train and prediction part
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from txt2image_dataset import Text2ImageDataset
from GAN import wgan_cls, gan_cls_new, gan_cls_original
from utils import Utils


DEVICE = torch.device('cuda')
torch.cuda.set_device(1)


class ResGAN(object):
    def __init__(self, dataset='flowers', split=0, lr=2e-4, diter=5, save_path='./Log', l1_coef=90, l2_coef=100, pre_trained_gen=False, pre_trained_disc=False, batch_size=64, num_workers=16, epochs=800):
        self.generator = gan_cls_new.generator().to(DEVICE)
        self.discriminator = gan_cls_new.discriminator().to(DEVICE)
        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load('./Log/checkpoints/disc_190.pth'))
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load('./Log/checkpoints/gen_190.pth'))
        else:
            self.generator.apply(Utils.weights_init)

        # choose smaller flower data set
        if dataset == 'flowers':
            self.dataset = Text2ImageDataset('./data/flowers.hdf5', split=split)
        elif dataset == 'birds':
            self.dataset = Text2ImageDataset('./data/birds.hdf5', split=split)
        else:
            print('Data not supported, please select either birds.hdf5 or flowers.hdf5')
            exit()
        # print(self.dataset.__len__()) # 29390 training samples
        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.DITER = diter

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.checkpoints_path = 'checkpoints/2gen_800epochs'
        self.save_path = save_path

    def train(self, cls):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0
        # tensorboardX
        localtime = time.asctime(time.localtime(time.time()))
        localtime = localtime.replace(" ", "_")
        localtime = localtime.replace(":", "_")
        print(localtime)
        writer = SummaryWriter(log_dir=os.path.join('./Log/tensorboard_file', localtime))

        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                wrong_images = sample['wrong_images']

                right_images = Variable(right_images.float()).to(DEVICE)
                right_embed = Variable(right_embed.float()).to(DEVICE)
                wrong_images = Variable(wrong_images.float()).to(DEVICE)

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.4))

                real_labels = Variable(real_labels).to(DEVICE)
                smoothed_real_labels = Variable(smoothed_real_labels).to(DEVICE)
                fake_labels = Variable(fake_labels).to(DEVICE)

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                noise = Variable(torch.randn(right_images.size(0), 100)).to(DEVICE)
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss

                if cls:
                    d_loss = d_loss + wrong_loss

                self.optimD.zero_grad()
                d_loss.backward()
                self.optimD.step()


                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).to(DEVICE)
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)
                _, activation_real = self.discriminator(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                # ======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                # ===========================================
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)

                self.optimG.zero_grad()
                g_loss.backward(retain_graph=True)
                g_loss.backward()
                self.optimG.step()
                # using tensorboardX to plot loss
                writer.add_scalar('Train/Discriminator_loss', d_loss.data, iteration)
                writer.add_scalar('Train/Generator_loss', g_loss.data, iteration)

                if iteration % 20 == 0:
                    # printing current loss
                    print('Epoch:', epoch, '|Iteration:', iteration, '|G_loss:', g_loss.data.cpu().numpy(), '|D_loss:', d_loss.data.cpu().numpy())

            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.save_path, self.checkpoints_path, epoch)

    def predict(self, trained=False):
        # making prediction
        test_data = Text2ImageDataset('./data/flowers.hdf5', split=0)
        test_data_loader = DataLoader(test_data, batch_size=40, shuffle=False)
        for i in (0, 20, 50, 100, 200, 400, 790):
            ourmodel = 'gen_' + str(i + 200) + '.pth'
            our_model_path = os.path.join('./Log/checkpoints/800_gan_cls_new', ourmodel)
            originalmodel = 'gen_' + str(i) + '.pth'
            original_model_path = os.path.join('./Log/checkpoints/800_gan_cls', originalmodel)
            our_save_path = os.path.join('./results/', str(i), 'our')
            original_save_path = os.path.join('./results/', str(i), 'original')
            # loading trained model
            if trained:
                # loading trained model for prediction
                # construct model
                original_generator = gan_cls_original.generator().to(DEVICE)
                original_generator.load_state_dict(torch.load(original_model_path))
                our_generator = gan_cls_new.generator().to(DEVICE)
                our_generator.load_state_dict(torch.load(our_model_path))
                original_generator.eval()
                our_generator.eval()
            # self.generator.eval()
            count =0
            for sample in test_data_loader:  # only generate 100 batches
                count += 1
                if count > 1000:
                    break
                print(count)
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                txt = sample['txt']

                right_images = Variable(right_images.float()).cuda()
                right_embed = Variable(right_embed.float()).cuda()

                # Train the generator
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                original_fake_images = original_generator(right_embed, noise)
                our_fake_images = our_generator(right_embed, noise)
                # save
                for original_fake_image, our_fake_image, t in zip(original_fake_images, our_fake_images, txt):
                    original_im = Image.fromarray(original_fake_image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                    our_im = Image.fromarray(
                        our_fake_image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                    t = t.replace("/", "")
                    original_im.save('{0}/{1}.jpg'.format(original_save_path, t.replace("\n", "")[:200]))
                    our_im.save('{0}/{1}.jpg'.format(our_save_path, t.replace("\n", "")[:200]))





demo = ResGAN()
demo.train(cls=True)
demo.predict(trained=False)