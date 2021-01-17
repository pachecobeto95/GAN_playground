import torch 
import torch.nn as nn
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision.utils import save_image
import numpy as np
import sys
from torchsummary import summary
import GPUtil as GPU
GPUs = GPU.getGPUs()
import time

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
  def __init__(self, latent_dim: int, img_size: int):
    super(Generator, self).__init__()
    self.img_size = img_size
    self.latent_dim = latent_dim

    self.init_size = img_size // 4

    self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

    self.conv_blocks = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.BatchNorm2d(128, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 3, 3, stride=1, padding=1),
        nn.Tanh(),
        )

  def forward(self, z):
    out = self.l1(z)
    out = out.view(out.shape[0], 128, self.init_size, self.init_size)
    img = self.conv_blocks(out)
    return img
    
class Discriminator(nn.Module):
  def __init__(self, img_size: int):
    super(Discriminator, self).__init__()

    def discriminator_block(in_filters, out_filters, bn=True):
      block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
      if bn:
        block.append(nn.BatchNorm2d(out_filters, 0.8))
      return block

    self.model = nn.Sequential(
        *discriminator_block(3, 16, bn=False),
        *discriminator_block(16, 32),
        *discriminator_block(32, 64),
        *discriminator_block(64, 128),
        )

    # The height and width of downsampled image
    ds_size = img_size // 2 ** 4
    self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

  def forward(self, img):
    out = self.model(img)
    out = out.view(out.shape[0], -1)
    validity = self.adv_layer(out)

    return validity



def data_loader(train_path, test_path, batch_size, transform):
  train_dataset = datasets.ImageFolder(root=train_path,
                                 transform=transform)

  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size, shuffle=True)
  
  test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

  return train_loader, test_loader

train_path = "./data/celeba_hq_128/train/"
val_path = "./data/celeba_hq_128/val/"
batch_size = 64
img_size = 128
# data_tranformation = transforms.Compose([
#         transforms.Resize(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
data_tranformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

train_loader, test_loader = data_loader(train_path, val_path, batch_size, data_tranformation)

def train(train_loader, epoch):
  Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

  gen_imgs = None
  trans = transforms.ToPILImage()
  gpu = GPUs[0]

  g_loss_list, d_loss_list, loss_total_list = [], [], []
  for i, (imgs, _) in enumerate(train_loader):
    start  = time.time()
    
    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

    real_imgs = Variable(imgs.type(Tensor))

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], z_dim))))

    # Generate a batch of images
    gen_imgs = generator(z)       # generate images, come back here later
    
    d = discriminator(gen_imgs)
    g_loss = adversarial_loss(d, valid)


    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()


    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()

    loss_total = d_loss.item() + g_loss.item()
    g_loss_list.append(g_loss.item())
    d_loss_list.append(d_loss.item())
    loss_total_list.append(loss_total)

    end = time.time()
    batch_time = end - start

    if (i % 10==0):
      print(
          "[Epoch %d/%d] [Batch %d/%d]"
          % (epoch, n_epochs, i, len(train_loader))
          )
      print("GPU RAM Free: %s MB| Used: %s | Util: %s | Total : %s | Batch Time: %s"%(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal, 
                                                                     batch_time))



    #if (i == len(train_loader) - 1):
    #  print(
    #      "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [loss total: %f]"
    #      % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item(), loss_total)
    #      )

    batches_done = epoch * len(train_loader) + i
    
  mean_g_loss = np.mean(g_loss_list)
  mean_d_loss = np.mean(d_loss_list)
  mean_loss_total = np.mean(loss_total_list)
  print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [loss total: %f]"
  %(epoch, n_epochs, i, len(train_loader), mean_d_loss, mean_g_loss, mean_loss_total))
  save_image(gen_imgs.data[:10], "./drive/MyDrive/DCGAN-CelebAHQ-128/images/%d.jpg" % batches_done, nrow=5, normalize=True)
  
def eval(test_loader, epoch):
  Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

  gen_imgs = None
  trans = transforms.ToPILImage()
  best_loss = np.inf

  loss_total_list = []
  for i, (imgs, _) in enumerate(test_loader):
    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

    real_imgs = Variable(imgs.type(Tensor))

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], z_dim))))

    # Generate a batch of images
    gen_imgs = generator(z)       # generate images, come back here later

    d = discriminator(gen_imgs)
    g_loss = adversarial_loss(d, valid)

    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    loss_total = d_loss.item()+ g_loss.item()
    loss_total_list.append(loss_total)

    if (i == len(test_loader)):
      print(
          "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [loss total: %f]"
          % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item(), loss_total)
          )
    
  for j in range(len(gen_imgs)):
    save_image(gen_imgs.data[j], "./drive/MyDrive/DCGAN-CelebAHQ-128/images_val/%d.jpg" % j, normalize=True)
  
  loss_mean = np.mean(loss_total_list)
  if (loss_mean <  best_loss):
    best_loss = loss_total
    torch.save({
            'epoch': epoch,
            'model_generator_state_dict': generator.state_dict(),
            'model_discriminator_state_dict': discriminator.state_dict(),
            'generator_optimizer_state_dict': optimizer_G.state_dict(),
            'discriminator_optimizer_state_dict': optimizer_D.state_dict(),
            'loss': loss_mean,
            }, "./drive/MyDrive/DCGAN-CelebAHQ-128/best_model_GAN_2.pth")

z_dim = 128
img_size = 128
ngpu = 1
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 100
sample_interval = 400
modelPath = "./drive/MyDrive/DCGAN-CelebAHQ-128/best_model_GAN_2.pth"

# Initialize generator and discriminator
generator = Generator(z_dim, img_size)
discriminator = Discriminator(img_size)

# Loss function
adversarial_loss = nn.BCELoss()

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
cuda = torch.cuda.is_available()

generator.to(device)
discriminator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

initial_epoch = 0

try:

  generator.load_state_dict(torch.load(modelPath)['model_generator_state_dict'])
  discriminator.load_state_dict(torch.load(modelPath)['model_discriminator_state_dict'])
  optimizer_G.load_state_dict(torch.load(modelPath)['generator_optimizer_state_dict'])
  optimizer_D.load_state_dict(torch.load(modelPath)['discriminator_optimizer_state_dict'])

  initial_epoch = torch.load(modelPath)['epoch']

except Exception as e:
  print(e)

for epoch in range(n_epochs):
  current_epoch = initial_epoch + epoch
  train(train_loader, current_epoch)
  eval(test_loader, current_epoch)
