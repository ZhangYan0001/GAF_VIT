import timm
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def to_2tuple(x):
  return tuple([x] * 2)


trunc_normal_ = init.trunc_normal_
zeros_ = init.zeros_
ones_ = init.ones_


# 无操作的网络层
class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, input):
    return input


# 图像分块、Embedding
class PatchEmbed(nn.Module):
  def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
    super().__init__()
    # 原始大小为int，转为tuple，即：img_size原始输入224，变换后为[224,224]
    img_size = to_2tuple(img_size)
    patch_size = to_2tuple(patch_size)
    # 图像块的个数
    num_patches = (img_size[1] // patch_size[1]) * \
                  (img_size[0] // patch_size[0])
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = num_patches
    # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
    # 输入维度为3，输出维度为块向量长度
    # 与原文中：分块、展平、全连接降维保持一致
    # 输出为[B, C, H, W]
    self.proj = nn.Conv2d(
      in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

  def forward(self, x):
    B, C, H, W = x.shape
    assert H == self.img_size[0] and W == self.img_size[1], \
      "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    # [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
    x = self.proj(x).flatten(2).transpose((0, 2, 1))
    return x


# Multi-head Attention
class Attention(nn.Module):
  def __init__(self,
               dim,
               num_heads=8,
               qkv_bias=False,
               qk_scale=None,
               attn_drop=0.,
               proj_drop=0.):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = qk_scale or head_dim ** -0.5
    # 计算 q,k,v 的转移矩阵
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    # 最终的线性层
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
    batch_size, N, C = x.size()
    # 线性变换
    qkv = self.qkv(x).reshape((batch_size, N, 3, self.num_heads, C //
                               self.num_heads)).permute((2, 0, 3, 1, 4))
    # 分割 query key value
    q, k, v = qkv[0], qkv[1], qkv[2]
    # Scaled Dot-Product Attention
    # Matmul + Scale
    attn = (q @ k.transpose(-2, -1)) * self.scale
    # SoftMax
    attn = F.softmax(attn, dim=-1)
    attn = self.attn_drop(attn)
    # Matmul
    x = (attn @ v).transpose(1, 2).reshape(batch_size, N, C)
    # 线性变换
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class Mlp(nn.Module):
  def __init__(self,
               in_features,
               hidden_features=None,
               out_features=None,
               act_layer=nn.GELU,
               drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop = nn.Dropout(drop)

  def forward(self, x):
    # 输入层：线性变换
    x = self.fc1(x)
    # 应用激活函数
    x = self.act(x)
    # Dropout
    x = self.drop(x)
    # 输出层：线性变换
    x = self.fc2(x)
    # Dropout
    x = self.drop(x)
    return x


def drop_path(x, drop_prob=0., training=False):
  if drop_prob == 0. or not training:
    return x
  keep_prob = torch.tensor(1 - drop_prob)
  shape = (x.shape[0],) + (1,) * (x.ndim - 1)
  random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
  random_tensor = torch.floor(random_tensor)
  output = x.div(keep_prob) * random_tensor
  return output


class DropPath(nn.Module):
  def __init__(self, drop_prob: float = 0.):
    super().__init__()
    self.drop_prob = drop_prob

  def forward(self, x: torch.Tensor):
    return drop_path(x, self.drop_prob, self.training)

  def extra_repr(self) -> str:
    return f"drop_prob = {self.drop_prob}"


class Block(nn.Module):
  def __init__(self,
               dim,
               num_heads=8,
               mlp_ratio=4.,
               qkv_bias=False,
               qk_scale=None,
               drop=0.,
               attn_drop=0.,
               drop_path=0.,
               act_layer=nn.GELU,
               norm_layer="nn.LayerNorm",
               epsilon=1e-5):
    super().__init__()
    self.norm1 = eval(norm_layer)(dim, eps=epsilon)
    # Multi-head self-attention
    self.attn = Attention(
      dim,
      num_heads=num_heads,
      qkv_bias=qkv_bias,
      qk_scale=qk_scale,
      attn_drop=attn_drop,
      proj_drop=drop
    )
    # DropPath
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.norm2 = eval(norm_layer)(dim, eps=epsilon)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

  def forward(self, x):
    # Multi-head Self-attent , add, LayerNorm
    x = x + self.drop_path(self.attn(self.norm1(x)))
    # Feed Forward, Add, LayerNorm
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


class VisionTransformer(nn.Module):
  def __init__(
    self,
    img_size=224,
    patch_size = 16,
    in_chans = 3,
    class_dim = 1000,
    embed_dim = 768,
    depth = 12,
    num_heads = 12,
    mlp_ratio = 4,
    qkv_bias = False,
    qk_scale = None,
    drop_rate = 0.,
    attn_drop_rate=0.,
    drop_path_rate = 0.,
    norm_layer='nn.LayerNorm',
    epsilon=1e-5,
    **args
  ):
    super().__init__()
    self.class_dim = class_dim
    self.num_features = self.embed_dim = embed_dim
    # 图片分块和降维，块大小为patch_size，最终块向量维度为768
    self.patch_embed = PatchEmbed(
      img_size = img_size,
      patch_size = patch_size,
      in_chans = in_chans,
      embed_dim = embed_dim,
    )
    # 分块数量
    num_patches = self.patch_embed.num_patches
    # 可学习的位置编码
    self.pos_embed = self.create_parameter(
      shape = (1, num_patches +1, embed_dim),default_initializer = zeros_
    )
    self.add_parameter("pos_embed", self.pos_embed)
    # 人为追加class token, 并使用该向量进行分类预测
    self.cls_token = self.create_parameter(shape=(1, 1, embed_dim), default_initializer = zeros_)
    self.add_parameter("cls_token", self.cls_token)
    self.pos_drop = nn.Dropout(p=drop_rate)

    dpr = np.linspace(0, drop_path_rate, depth)

    # transformer
    self.blocks = nn.ModuleList([
      Block(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio = mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale = qk_scale,
        drop=drop_rate,
        attn_drop=attn_drop_rate,
        drop_path=dpr[i],
        norm_layer=norm_layer,
        epsilon=epsilon
      ) for i in range(depth)
    ])
    self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)
    self.head = nn.Linear(embed_dim, class_dim) if class_dim > 0 else Identity()
    trunc_normal_(self.pos_embed)
    trunc_normal_(self.cls_token)
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight)
      if isinstance(m, nn.Linear) and m.bias is not None:
        zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
      zeros_(m.bias)
      ones_(m.weight)

  def forward_features(self, x):
    B = x.size()[0]
    # 将图片分块，并调整每个快向量的维度
    x = self.patch_embed(x)
    # 将class token与前面的分块进行拼接
    cls_tokens = self.cls_token.expand((B, -1, -1))
    x = torch.concat((cls_tokens, x),dim=1)
    # 将编码向量中加入位置编码
    x = x + self.pos_embed
    x = self.pos_drop(x)
    # 堆叠 transformer结构
    for blk in self.blocks:
      x = blk(x)
    # LayerNorm
    x = self.norm(x)
    # 提取分类tokens的输出
    return x[:,0]

  def forward(self, x):
    # 获取图像特征
    x = self.forward_features(x)
    # 图像分类
    x = self.head(x)
    return x


class ViTRegressor(nn.Module):
  def __init__(self):
    super().__init__()
    self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    self.regressor = nn.Linear(self.vit.embed_dim, 1)

  def forward(self, x):
    x = self.vit(x)
    return self.regressor(x).squeeze()

def test_multi_device():
  devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
  for device in devices:
    block = Block(dim=256).to(device)
    x = torch.randn(2, 16, 256, device=device)
    output = block(x)
    assert output.device.type == device

def test_numerical_stability():
  block = Block(dim=256)
  x = torch.zeros(1, 16, 256)
  output = block(x)
  assert not torch.isnan(output).any(), "输出包含NaN"

def test_stack_blocks():
  depth = 4
  model = nn.Sequential(*[Block(dim=256) for _ in range(depth)])
  x = torch.randn(2, 16, 256)
  output = model(x)
  assert output.shape == x.shape

# 测试代码
if __name__ == "__main__":
  dim = 512
  num_heads = 8
  batch_size = 4
  seq_len = 16

  # 生成随机输入 (PyTorch 使用 torch.randn)
  x = torch.randn(batch_size, seq_len, dim)
  # 初始化注意力模块
  attn = Attention(dim, num_heads)
  # 前向传播
  output = attn(x)

  print("输入形状:", x.shape)  # 输出: torch.Size([4, 16, 512])
  print("输出形状:", output.shape)  # 输出: torch.Size([4, 16, 512])
  # 参数定义
  in_dim = 512
  hidden_dim = 1024
  out_dim = 256
  batch_size = 4

  # 初始化 MLP
  mlp = Mlp(in_features=in_dim,
            hidden_features=hidden_dim,
            out_features=out_dim,
            act_layer=nn.GELU,
            drop=0.1)

  # 随机输入数据
  x = torch.randn(batch_size, in_dim)

  # 前向传播
  output = mlp(x)
  print(f"输入形状: {x.shape}")  # torch.Size([4, 512])
  print(f"输出形状: {output.shape}")  # torch.Size([4, 256])
  # 参数定义
  batch_size = 4
  dim = 512
  drop_prob = 0.2

  # 初始化模块
  drop_path_layer = DropPath(drop_prob)

  # 训练模式
  drop_path_layer.train()
  x_train = torch.randn(batch_size, dim)
  out_train = drop_path_layer(x_train)
  print(f"训练模式输出中0的数量: {(out_train == 0).sum()}")  # 部分元素被置0

  # 评估模式
  drop_path_layer.eval()
  x_eval = torch.randn(batch_size, dim)
  out_eval = drop_path_layer(x_eval)
  print(f"评估模式输出是否一致: {torch.allclose(x_eval, out_eval)}")  # 输出True
