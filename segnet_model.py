"""
Medical Segmentation Models - SegNet3D, DenseNet3D, ResNet3D
============================================================
Multiple architectures for 3D medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SEGNET3D (Existing - Encoder-Decoder with Pooling Indices)
# =============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv=2):
        super().__init__()
        layers = []
        for i in range(n_conv):
            layers.extend([
                nn.Conv3d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            ])
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool3d(2, 2, return_indices=True)
    
    def forward(self, x):
        x = self.conv(x)
        size = x.size()
        x, idx = self.pool(x)
        return x, idx, size


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv=2):
        super().__init__()
        self.unpool = nn.MaxUnpool3d(2, 2)
        layers = []
        for i in range(n_conv):
            layers.extend([
                nn.Conv3d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x, idx, size):
        x = self.unpool(x, idx, output_size=size)
        return self.conv(x)


class LightSegNet3D(nn.Module):
    """Lightweight SegNet3D for CPU training"""
    def __init__(self, in_ch=1, out_ch=2, f=16):
        super().__init__()
        self.enc1 = EncoderBlock(in_ch, f, 1)
        self.enc2 = EncoderBlock(f, f*2, 1)
        self.enc3 = EncoderBlock(f*2, f*4, 2)
        self.enc4 = EncoderBlock(f*4, f*8, 2)
        self.dec4 = DecoderBlock(f*8, f*4, 2)
        self.dec3 = DecoderBlock(f*4, f*2, 2)
        self.dec2 = DecoderBlock(f*2, f, 1)
        self.dec1 = DecoderBlock(f, f, 1)
        self.final = nn.Conv3d(f, out_ch, 1)
    
    def forward(self, x):
        x, i1, s1 = self.enc1(x)
        x, i2, s2 = self.enc2(x)
        x, i3, s3 = self.enc3(x)
        x, i4, s4 = self.enc4(x)
        x = self.dec4(x, i4, s4)
        x = self.dec3(x, i3, s3)
        x = self.dec2(x, i2, s2)
        x = self.dec1(x, i1, s1)
        return self.final(x)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SegNet3D(nn.Module):
    """Full SegNet3D architecture"""
    def __init__(self, in_ch=1, out_ch=2, f=32):
        super().__init__()
        self.enc1 = EncoderBlock(in_ch, f, 2)
        self.enc2 = EncoderBlock(f, f*2, 2)
        self.enc3 = EncoderBlock(f*2, f*4, 3)
        self.enc4 = EncoderBlock(f*4, f*8, 3)
        self.dec4 = DecoderBlock(f*8, f*4, 3)
        self.dec3 = DecoderBlock(f*4, f*2, 3)
        self.dec2 = DecoderBlock(f*2, f, 2)
        self.dec1 = DecoderBlock(f, f, 2)
        self.final = nn.Conv3d(f, out_ch, 1)
    
    def forward(self, x):
        x, i1, s1 = self.enc1(x)
        x, i2, s2 = self.enc2(x)
        x, i3, s3 = self.enc3(x)
        x, i4, s4 = self.enc4(x)
        x = self.dec4(x, i4, s4)
        x = self.dec3(x, i3, s3)
        x = self.dec2(x, i2, s2)
        x = self.dec1(x, i1, s1)
        return self.final(x)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# DENSENET3D - Dense connections for better gradient flow
# =============================================================================
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_ch)
        self.conv = nn.Conv3d(in_ch, growth, 3, padding=1)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer(in_ch + i * growth, growth))
        self.block = nn.Sequential(*layers)
        self.out_ch = in_ch + n_layers * growth
    
    def forward(self, x):
        return self.block(x)


class TransitionDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_ch)
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        self.pool = nn.AvgPool3d(2, 2)
    
    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class TransitionUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
    
    def forward(self, x):
        return self.up(x)


class DenseNet3D(nn.Module):
    """DenseNet3D for medical segmentation - better gradient flow"""
    def __init__(self, in_ch=1, out_ch=2, growth=12, n_layers=4):
        super().__init__()
        f = 24
        self.init_conv = nn.Conv3d(in_ch, f, 3, padding=1)
        
        self.db1 = DenseBlock(f, growth, n_layers)
        self.td1 = TransitionDown(self.db1.out_ch, self.db1.out_ch // 2)
        
        self.db2 = DenseBlock(self.db1.out_ch // 2, growth, n_layers)
        self.td2 = TransitionDown(self.db2.out_ch, self.db2.out_ch // 2)
        
        self.db3 = DenseBlock(self.db2.out_ch // 2, growth, n_layers)
        
        self.tu2 = TransitionUp(self.db3.out_ch, self.db3.out_ch)
        self.db4 = DenseBlock(self.db3.out_ch + self.db2.out_ch, growth, n_layers)
        
        self.tu1 = TransitionUp(self.db4.out_ch, self.db4.out_ch)
        self.db5 = DenseBlock(self.db4.out_ch + self.db1.out_ch, growth, n_layers)
        
        self.final = nn.Conv3d(self.db5.out_ch, out_ch, 1)
    
    def forward(self, x):
        x = self.init_conv(x)
        d1 = self.db1(x)
        x = self.td1(d1)
        d2 = self.db2(x)
        x = self.td2(d2)
        x = self.db3(x)
        x = self.tu2(x)
        x = F.interpolate(x, size=d2.shape[2:], mode='trilinear', align_corners=False)
        x = self.db4(torch.cat([x, d2], 1))
        x = self.tu1(x)
        x = F.interpolate(x, size=d1.shape[2:], mode='trilinear', align_corners=False)
        x = self.db5(torch.cat([x, d1], 1))
        return self.final(x)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# RESNET3D - Residual connections for deep networks
# =============================================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride),
                nn.BatchNorm3d(out_ch)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ResNet3D(nn.Module):
    """ResNet3D for medical segmentation - residual connections"""
    def __init__(self, in_ch=1, out_ch=2, f=16):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv3d(in_ch, f, 3, 1, 1),
            nn.BatchNorm3d(f),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.enc1 = ResBlock(f, f*2, 2)
        self.enc2 = ResBlock(f*2, f*4, 2)
        self.enc3 = ResBlock(f*4, f*8, 2)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(f*8, f*4, 2, 2)
        self.dec3 = ResBlock(f*8, f*4, 1)
        self.up2 = nn.ConvTranspose3d(f*4, f*2, 2, 2)
        self.dec2 = ResBlock(f*4, f*2, 1)
        self.up1 = nn.ConvTranspose3d(f*2, f, 2, 2)
        self.dec1 = ResBlock(f*2, f, 1)
        
        self.final = nn.Conv3d(f, out_ch, 1)
    
    def forward(self, x):
        x0 = self.init(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        d3 = self.up3(e3)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], 1))
        
        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], 1))
        
        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=x0.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, x0], 1))
        
        return self.final(d1)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Model Factory
# =============================================================================
def get_model(name, in_ch=1, out_ch=2):
    """Get model by name"""
    models = {
        "segnet": LightSegNet3D(in_ch, out_ch, 16),
        "segnet_full": SegNet3D(in_ch, out_ch, 32),
        "densenet": DenseNet3D(in_ch, out_ch, growth=8, n_layers=3),
        "resnet": ResNet3D(in_ch, out_ch, f=16),
    }
    return models.get(name.lower(), LightSegNet3D(in_ch, out_ch, 16))


if __name__ == "__main__":
    print("Testing all models...")
    x = torch.randn(1, 1, 32, 32, 32)
    
    for name in ["segnet", "densenet", "resnet"]:
        model = get_model(name, 1, 2)
        y = model(x)
        print(f"{name}: {model.get_num_params():,} params, output {y.shape}")
    
    print("All models OK!")
