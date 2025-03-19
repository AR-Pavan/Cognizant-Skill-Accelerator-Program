import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_size = 64
batch_size = 128
nz = 100  
gen_features = 64
disc_features = 64
lr = 0.0002
beta1 = 0.5
epochs = 50


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root="./abstract_art", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, gen_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 8, gen_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_features * 2, gen_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(gen_features, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, disc_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features, disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features * 2, disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features * 4, disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(disc_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(tqdm(dataloader)):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Train Discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        outputs = discriminator(real_images).view(-1, 1)
        loss_real = criterion(outputs, real_labels)
        loss_real.backward()
        
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach()).view(-1, 1)
        loss_fake = criterion(outputs, fake_labels)
        loss_fake.backward()
        optimizer_d.step()
        
        # Train Generator
        generator.zero_grad()
        outputs = discriminator(fake_images).view(-1, 1)
        loss_gen = criterion(outputs, real_labels)
        loss_gen.backward()
        optimizer_g.step()
        
    print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {loss_real+loss_fake:.4f}, Loss_G: {loss_gen:.4f}")
    
    # To save the generated images
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake_images = generator(noise).detach().cpu()
        vutils.save_image(fake_images, f"generated_epoch_{epoch+1}.png", normalize=True)

# Generate final artistic images
def generate_art(n=5):
    generator.eval()
    noise = torch.randn(n, nz, 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(np.transpose(fake_images[i], (1, 2, 0))*0.5 + 0.5)
        plt.axis("off")
    plt.show()

generate_art(5)
