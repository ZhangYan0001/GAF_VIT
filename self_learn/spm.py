"""
Spatial Prior Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
  def __init__(self, channel, reduction=16):
    super(SELayer, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(channel // reduction, channel, bias=False),
      nn.Sigmoid(),
    )
  
  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)


class SpatialPriorModule(nn.Module):
  def __init__(
    self,
    in_chans=3,
    out_dims=[64, 128, 320, 512],
    strides=[2, 2, 2, 2],
    depths=[1, 1, 1, 1],
    use_channel_attention=True,
  ):
    super().__init__()
    self.num_stages = len(out_dims)
    self.stages = nn.ModuleList()
    self.channel_attentions = nn.ModuleList() if use_channel_attention else None
    current_in_chans = in_chans
    
    for i in range(self.num_stages):
      stage_layers = []
      conv_block = nn.Sequential(
        nn.Conv2d(
          current_in_chans, out_dims[i], kernel_size=3, stride=strides[i], padding=1
        ),
        nn.BatchNorm2d(out_dims[i]),
        nn.ReLU(inplace=True),
      )
      stage_layers.append(conv_block)


# class VITAdapterBackbone(nn.Module):


class DiagConv(nn.Module):
  def __init__(self, in_chans, out_chans, kernel_size=3):
    super().__init__()
    self.conv = nn.Conv2d(
      in_chans, out_chans, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
    )
  
  def forward(self, x):
    x_rot = torch.rot90(x, 1, dims=[2, 3])
    f = self.conv(x_rot)
    f = torch.rot90(f, 3, dims=[2, 3])
    return f


class KANBlock(nn.Module):
  def __init__(self, in_chans, mid_chans=None, kernel_size=3):
    super().__init__()
    mid = mid_chans or in_chans
    self.diag_pos = DiagConv(in_chans, mid, kernel_size)
    self.diag_neg = DiagConv(in_chans, mid, kernel_size)
    self.fuse = nn.Sequential(
      nn.Conv2d(2*mid, in_chans, kernel_size=1, bias=False),
      nn.BatchNorm2d(in_chans),
      nn.ReLU(inplace=True)
    )
  
  def forward(self, x):
    f_pos = self.diag_pos(x)
    x_rot = torch.rot90(x, 3, dims=[2, 3])
    f_neg = self.diag_neg(x_rot)
    f_neg = torch.rot90(f_neg, 1, dims=[2, 3])
    f = torch.cat([f_pos, f_neg], dim=1)
    f = self.fuse(f)
    
    return f + x
