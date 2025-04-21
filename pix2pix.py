import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

satellite_dataset = ImageFolder(root="./data/satellite", transform=transform)
map_dataset = ImageFolder(root="./data/map", transform=transform)

batch_size = 1
satellite_loader = DataLoader(satellite_dataset, batch_size=batch_size, shuffle=True)
map_loader = DataLoader(map_dataset, batch_size=batch_size, shuffle=True)

# Show sample
sample_image, _ = satellite_dataset[0]
sample_image = sample_image * 0.5 + 0.5  # De-normalize
plt.imshow(sample_image.permute(1, 2, 0))
plt.title("Sample Image - Satellite Domain")
plt.axis("off")
plt.show()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # 256 → 128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128 → 64
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),    # 128 → 256
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # 256 → 128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 → 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 64 → 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)            # [B, 256, 1, 1]
        x = torch.flatten(x, 1)         # [B, 256]
        x = self.fc(x)                  # [B, 1]
        x = torch.sigmoid(x)           # Output between 0 and 1
        return x


G_sat2map = Generator()
D_map = Discriminator()


adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

optimizer_G = optim.Adam(G_sat2map.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D_map.parameters(), lr=0.0002)


epochs = 10
for epoch in range(epochs):
    for (sat_images, _), (map_images, _) in zip(satellite_loader, map_loader):

       
        fake_map = G_sat2map(sat_images)

   
        optimizer_D.zero_grad()
        pred_real = D_map(map_images)
        pred_fake = D_map(fake_map.detach())
        real_loss = adversarial_loss(pred_real, torch.ones_like(pred_real))
        fake_loss = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        
        optimizer_G.zero_grad()
        pred_fake = D_map(fake_map)
        g_loss = adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        recon_loss = l1_loss(fake_map, map_images)
        total_g_loss = g_loss + 10 * recon_loss
        total_g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {total_g_loss.item():.4f}")
