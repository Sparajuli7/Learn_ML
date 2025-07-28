# Computer Vision Advanced: Segmentation, GANs, and 3D Vision

*"Pushing the boundaries of visual understanding with cutting-edge techniques"*

---

## 📚 Table of Contents

1. [Image Segmentation](#image-segmentation)
2. [Generative Adversarial Networks](#generative-adversarial-networks)
3. [3D Computer Vision](#3d-computer-vision)
4. [Vision Transformers](#vision-transformers)
5. [Real-World Applications](#real-world-applications)
6. [Exercises and Projects](#exercises-and-projects)
7. [Further Reading](#further-reading)

---

## 🎯 Image Segmentation

### Semantic Segmentation

Semantic segmentation assigns a class label to each pixel in an image, enabling pixel-level understanding.

#### 1. **U-Net Architecture**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        return torch.sigmoid(self.outc(x))
```

#### 2. **DeepLab Architecture**

```python
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.atrous_6 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
        self.atrous_12 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
        self.atrous_18 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=256, mode='bilinear', align_corners=False)
        )
        
        self.conv_1x1 = nn.Conv2d(out_channels * 5, out_channels, 1, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        size = x.size()[2:]
        
        atrous_1 = self.atrous_1(x)
        atrous_6 = self.atrous_6(x)
        atrous_12 = self.atrous_12(x)
        atrous_18 = self.atrous_18(x)
        global_avg_pool = self.global_avg_pool(x)
        
        x = torch.cat([atrous_1, atrous_6, atrous_12, atrous_18, global_avg_pool], dim=1)
        x = self.conv_1x1(x)
        x = self.dropout(x)
        
        return x

class DeepLab(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Use ResNet as backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.aspp = ASPP(2048, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.size()[2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        return x
```

### Instance Segmentation

Instance segmentation goes beyond semantic segmentation by distinguishing between different instances of the same class.

#### 1. **Mask R-CNN Implementation**

```python
class MaskRCNN(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # RPN (Region Proposal Network)
        self.rpn = RPN(256, 512)
        
        # ROI Head
        self.roi_head = ROIHead(256, num_classes)
        
        # Mask Head
        self.mask_head = MaskHead(256, num_classes)
    
    def forward(self, images, targets=None):
        # Extract features
        features = self.backbone(images)
        
        # Generate proposals
        proposals, proposal_losses = self.rpn(features, targets)
        
        # ROI pooling and classification
        detections, detector_losses = self.roi_head(features, proposals, targets)
        
        # Mask prediction
        if self.training:
            mask_losses = self.mask_head(features, detections, targets)
            return {**proposal_losses, **detector_losses, **mask_losses}
        else:
            masks = self.mask_head(features, detections)
            return detections, masks

class RPN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.cls_head = nn.Conv2d(hidden_channels, 9, 1)  # 9 anchors per position
        self.reg_head = nn.Conv2d(hidden_channels, 36, 1)  # 4 coords * 9 anchors
    
    def forward(self, features, targets=None):
        x = F.relu(self.conv(features))
        cls_scores = self.cls_head(x)
        bbox_preds = self.reg_head(x)
        
        # Generate anchors and proposals
        anchors = self.generate_anchors(features.size()[2:])
        proposals = self.generate_proposals(anchors, bbox_preds, cls_scores)
        
        if self.training:
            losses = self.compute_losses(anchors, bbox_preds, cls_scores, targets)
            return proposals, losses
        else:
            return proposals, {}

class ROIHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.cls_head = nn.Linear(1024, num_classes)
        self.reg_head = nn.Linear(1024, num_classes * 4)
    
    def forward(self, features, proposals, targets=None):
        # ROI pooling
        pooled_features = self.roi_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Classification and regression
        fc_features = self.fc(pooled_features)
        cls_scores = self.cls_head(fc_features)
        bbox_preds = self.reg_head(fc_features)
        
        if self.training:
            losses = self.compute_losses(cls_scores, bbox_preds, targets)
            return (cls_scores, bbox_preds), losses
        else:
            return (cls_scores, bbox_preds), {}

class MaskHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, features, detections):
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        
        return torch.sigmoid(x)
```

---

## 🎨 Generative Adversarial Networks

### GAN Fundamentals

GANs consist of two competing networks: a generator that creates fake data and a discriminator that tries to distinguish real from fake data.

#### 1. **Basic GAN Implementation**

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 8 x 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 16 x 16
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: 64 x 32 x 32
            
            nn.ConvTranspose2d(64, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: num_channels x 64 x 64
        )
    
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: num_channels x 64 x 64
            nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 32 x 32
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 16 x 16
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 8 x 8
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 512 x 4 x 4
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

class GAN:
    def __init__(self, latent_dim=100, num_channels=3):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, num_channels)
        self.discriminator = Discriminator(num_channels)
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
    
    def train_step(self, real_images):
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size).to(real_images.device)
        fake_labels = torch.zeros(batch_size).to(real_images.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real images
        real_outputs = self.discriminator(real_images)
        d_real_loss = self.criterion(real_outputs, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(real_images.device)
        fake_images = self.generator(z)
        fake_outputs = self.discriminator(fake_images.detach())
        d_fake_loss = self.criterion(fake_outputs, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        fake_outputs = self.discriminator(fake_images)
        g_loss = self.criterion(fake_outputs, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate_images(self, num_images=16):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.latent_dim, 1, 1)
            fake_images = self.generator(z)
        return fake_images
```

#### 2. **Conditional GAN (cGAN)**

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        self.main = nn.Sequential(
            # Input: (latent_dim + latent_dim) x 1 x 1
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(label_embedding.size(0), -1, 1, 1)
        
        # Concatenate noise and label embedding
        z = z.view(z.size(0), -1, 1, 1)
        x = torch.cat([z, label_embedding], dim=1)
        
        return self.main(x)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super().__init__()
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, num_channels * 64 * 64)
        
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        # Embed labels and reshape
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(label_embedding.size(0), -1, 64, 64)
        
        # Concatenate image and label embedding
        x = torch.cat([x, label_embedding], dim=1)
        
        return self.main(x).view(-1, 1).squeeze(1)
```

---

## 🔍 3D Computer Vision

### Point Cloud Processing

#### 1. **PointNet Architecture**

```python
class PointNet(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        
        # Shared MLP for point feature learning
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # T-Net for input transformation
        self.input_transform = TNet(3)
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # T-Net for feature transformation
        self.feature_transform = TNet(64)
        
        # Global feature processing
        self.global_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(1)
        
        # Input transformation
        trans = self.input_transform(x)
        x = torch.bmm(x, trans)
        
        # Point feature learning
        x = x.transpose(2, 1)  # (batch_size, 3, num_points)
        x = self.mlp1(x)
        
        # Feature transformation
        trans_feat = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans_feat)
        x = x.transpose(2, 1)
        
        # Global feature learning
        x = self.mlp2(x)
        x = torch.max(x, 2, keepdim=True)[0]  # Global max pooling
        x = x.view(batch_size, -1)
        
        # Classification
        x = self.global_mlp(x)
        
        return x

class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Global feature
        x = self.mlp(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Transformation matrix
        x = self.fc(x)
        x = x.view(batch_size, self.k, self.k)
        
        # Initialize as identity matrix
        iden = torch.eye(self.k).repeat(batch_size, 1, 1).to(x.device)
        x = x + iden
        
        return x
```

#### 2. **3D Object Detection**

```python
class VoxelNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
        # Voxel Feature Encoding
        self.vfe_layers = nn.ModuleList([
            VFELayer(4, 32),
            VFELayer(32, 64),
            VFELayer(64, 128)
        ])
        
        # Convolutional Middle Layers
        self.middle_conv = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        # Region Proposal Network
        self.rpn = RPN3D(64, num_classes)
    
    def forward(self, voxels, coordinates, num_points):
        # Voxel Feature Encoding
        x = voxels
        for vfe in self.vfe_layers:
            x = vfe(x, num_points)
        
        # Sparse to dense
        x = self.sparse_to_dense(x, coordinates)
        
        # Convolutional Middle Layers
        x = self.middle_conv(x)
        
        # Region Proposal Network
        x = self.rpn(x)
        
        return x

class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.units = out_channels // 2
        self.linear = nn.Linear(in_channels, self.units)
        self.bn = nn.BatchNorm1d(self.units)
    
    def forward(self, x, num_points):
        # x: (batch_size, max_points, in_channels)
        # num_points: (batch_size,)
        
        batch_size = x.size(0)
        max_points = x.size(1)
        
        # Linear transformation
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x)
        
        # Element-wise max pooling
        x_max = torch.max(x, dim=1)[0]
        x_max = x_max.unsqueeze(1).expand(-1, max_points, -1)
        
        # Concatenate
        x = torch.cat([x, x_max], dim=2)
        
        return x

class RPN3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(256, 256, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 3, 2, 1, 1)
        
        # Output layers
        self.cls_head = nn.Conv2d(256, 2, 1)  # foreground/background
        self.reg_head = nn.Conv2d(256, 7, 1)  # 3D box parameters
    
    def forward(self, x):
        # Apply blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Deconvolution
        x = self.deconv1(x)
        x = self.deconv2(x)
        
        # Outputs
        cls_output = self.cls_head(x)
        reg_output = self.reg_head(x)
        
        return cls_output, reg_output
```

---

## 🔄 Vision Transformers

### ViT (Vision Transformer) Implementation

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Classification
        x = x[:, 0]  # Use class token
        x = self.classifier(x)
        
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        x = x + self.dropout(self.attention(self.norm1(x)))
        
        # MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
```

---

## 🎯 Real-World Applications

### 1. **Medical Image Segmentation**

```python
class MedicalSegmenter:
    def __init__(self, model_type='unet'):
        if model_type == 'unet':
            self.model = UNet(n_channels=1, n_classes=1)  # Binary segmentation
        elif model_type == 'deeplab':
            self.model = DeepLab(num_classes=1)
    
    def segment_organs(self, image):
        """Segment organs in medical images"""
        # Preprocess
        processed = self.preprocess_medical_image(image)
        
        # Predict
        with torch.no_grad():
            mask = self.model(processed.unsqueeze(0))
            mask = torch.sigmoid(mask)
        
        return mask.squeeze().numpy()
    
    def preprocess_medical_image(self, image):
        """Preprocess medical image for segmentation"""
        # Normalize
        image = (image - image.min()) / (image.max() - image.min())
        
        # Convert to tensor
        image = torch.FloatTensor(image).unsqueeze(0)
        
        return image
```

### 2. **Style Transfer with GANs**

```python
class StyleTransferGAN:
    def __init__(self):
        self.generator = StyleGenerator()
        self.discriminator = StyleDiscriminator()
    
    def transfer_style(self, content_image, style_image):
        """Transfer artistic style to content image"""
        # Generate stylized image
        stylized = self.generator(content_image, style_image)
        
        return stylized
    
    def train_step(self, content_batch, style_batch, real_batch):
        """Train the style transfer GAN"""
        # Train discriminator
        fake_images = self.generator(content_batch, style_batch)
        real_outputs = self.discriminator(real_batch)
        fake_outputs = self.discriminator(fake_images.detach())
        
        d_loss = self.compute_discriminator_loss(real_outputs, fake_outputs)
        
        # Train generator
        fake_outputs = self.discriminator(fake_images)
        g_loss = self.compute_generator_loss(fake_outputs, content_batch, fake_images)
        
        return d_loss, g_loss
```

---

## 🧪 Exercises and Projects

### Exercise 1: Implement U-Net Training

```python
def train_unet_segmentation():
    """Train U-Net for image segmentation"""
    model = UNet(n_channels=3, n_classes=1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (images, masks) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
    
    return model
```

### Exercise 2: Build a Simple GAN

```python
def build_and_train_gan():
    """Build and train a basic GAN"""
    gan = GAN(latent_dim=100, num_channels=3)
    
    for epoch in range(num_epochs):
        for batch_idx, real_images in enumerate(train_loader):
            d_loss, g_loss = gan.train_step(real_images)
    
    return gan
```

### Project 1: 3D Object Detection System

Build a system that can:
- Process point cloud data
- Detect 3D objects in real-time
- Provide bounding box predictions
- Handle multiple object classes

### Project 2: Medical Image Analysis Pipeline

Create a pipeline that can:
- Segment different organs
- Detect abnormalities
- Provide confidence scores
- Generate detailed reports

### Project 3: Style Transfer Application

Build an application that can:
- Transfer artistic styles to photos
- Support multiple style options
- Provide real-time preview
- Save and share results

---

## 📖 Further Reading

### Essential Papers
- Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Goodfellow, I., et al. (2014). "Generative Adversarial Networks"
- Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition"

### Books
- "Deep Learning for Computer Vision" by Adrian Rosebrock
- "Generative Deep Learning" by David Foster

### Online Resources
- [PyTorch Vision Tutorials](https://pytorch.org/vision/stable/index.html)
- [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### Next Steps
- **[RL Basics](10_rl_basics.md)**: Reinforcement learning fundamentals
- **[ML Engineering](21_data_engineering.md)**: Production computer vision systems
- **[Specialized ML](12_time_series_forecasting.md)**: Time series and forecasting

---

## 🎯 Key Takeaways

1. **Image Segmentation**: Pixel-level understanding through U-Net and DeepLab architectures
2. **Generative Models**: GANs for creating realistic images and style transfer
3. **3D Vision**: Point cloud processing and 3D object detection
4. **Vision Transformers**: Attention-based models for visual understanding
5. **Real-World Impact**: Powers medical imaging, autonomous vehicles, and creative AI
6. **2025 Relevance**: Integration with multimodal AI and edge deployment

---

*"Advanced computer vision is the bridge between pixels and understanding."*

**Next: [RL Basics](10_rl_basics.md) → Markov processes, Q-learning, and reinforcement learning** 