import torch
import torch.nn as nn
from nni.nas import pytorch as nas

from .build import META_ARCH_REGISTRY

__all__ = [
    'FakeNet3D',
    'FakeNet2D',
    '_FakeNet3D',
    '_FakeNet2D',
]

@META_ARCH_REGISTRY.register()
def FakeNet3D(cfg):
    return _FakeNet3D()

@META_ARCH_REGISTRY.register()
def FakeNet2D(cfg):
    return _FakeNet2D()

class _FakeNet3D(nn.Module):
    def __init__(self):
        super(_FakeNet3D, self).__init__()
        op_candidates = [
            nn.Conv3d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(1,10,kernel_size=5,stride=1,padding=2),
            nn.Conv3d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(1,10,kernel_size=5,stride=1,padding=2),
            nn.Conv3d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(1,10,kernel_size=5,stride=1,padding=2),
            nn.Conv3d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(1,10,kernel_size=5,stride=1,padding=2),
            nn.Conv3d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv3d(1,10,kernel_size=5,stride=1,padding=2),
        ]
        self.conv = nas.mutables.LayerChoice(op_candidates, return_mask=False, key="conv0")
        depth = 16
        size = 64
        self.fc = nn.Linear(10*depth*size*size, 3)

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out.view(x.shape[0], -1))
        return out


@META_ARCH_REGISTRY.register()
class _FakeNet2D(nn.Module):
    def __init__(self):
        super(_FakeNet2D, self).__init__()
        op_candidates = [
            nn.Conv2d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2),
            nn.Conv2d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2),
            nn.Conv2d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2),
            nn.Conv2d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2),
            nn.Conv2d(1,10,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2),
        ]
        self.conv = nas.mutables.LayerChoice(op_candidates, return_mask=False, key="conv0")
        size = 64
        self.fc = nn.Linear(10*size*size, 3)

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out.view(x.shape[0], -1))
        return out
