import torch
from torch import nn
from torchvision.models.video.resnet import BasicBlock as BasicBlock3D, Bottleneck as Bottleneck3D

from modeling.backbones.conv3d_blocks import Conv2Plus1D, Conv3DSimple
from modeling.backbones.resnet import ResNet
from modeling.backbones.swin_transformer_3d import BasicLayer, PatchEmbed3D
from modeling.modules import SetBlockWrapper

block_map_3d = {'BasicBlock': BasicBlock3D,
                'Bottleneck': Bottleneck3D}

conv_map_3d = {'Conv2Plus1D': Conv2Plus1D,
               'Conv3DSimple': Conv3DSimple}


class Interpolation2D(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(Interpolation2D, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, mode=self.mode, align_corners=False)


class SwinGait(ResNet):
    def __init__(self, block_2d, block_3d, conv_3d, channels=None, in_channel=1, layers=None, strides=None,
                 maxpool=True,
                 attn_drop=0.0, drop_rate=0.0, drop_path_rate=0.2, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm):
        if strides is None:
            strides = [1, 2, 2, 1]
        if layers is None:
            layers = [1, 2, 2, 1]
        if channels is None:
            channels = [32, 64, 128, 256]
        super(SwinGait, self).__init__(block=block_2d, channels=channels, in_channel=in_channel, layers=layers,
                                       strides=strides, maxpool=maxpool)

        if block_3d in block_map_3d.keys():
            block_3d = block_map_3d[block_3d]
        else:
            raise ValueError("Error type for -block_3d-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")

        if conv_3d in conv_map_3d.keys():
            conv_3d = conv_map_3d[conv_3d]
        else:
            raise ValueError(
                "Error type for -conv_3d-Cfg-, supported: 'Conv3DNoTemporal', 'Conv2Plus1D' or 'Conv3DSimple'.")

        self.conv1 = SetBlockWrapper(nn.Sequential(self.conv1, self.bn1))
        self.layer1 = SetBlockWrapper(self.layer1)
        # 3D Residual block_2d
        self.inplanes = channels[0]
        self.layer2 = self._make_layer_3d(block_3d, conv_3d, channels[1], layers[1], stride=strides[1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        # Bilinear interpolation
        self.bilinear_interp = SetBlockWrapper(Interpolation2D(size=(30, 20), mode='bilinear'))
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=(1, 2, 2), in_chans=channels[1], embed_dim=channels[2],
            norm_layer=nn.LayerNorm)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3D Swin Transformer Block
        self.layer3 = BasicLayer(dim=channels[2],
                                 depth=layers[2],
                                 num_heads=channels[2] // 32,
                                 window_size=(3, 3, 5),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop_rate,
                                 attn_drop=attn_drop,
                                 drop_path=dpr[sum(layers[:2]):sum(layers[:3])],
                                 norm_layer=norm_layer,
                                 downsample=None,
                                 )
        self.linear_embedding = nn.Conv3d(channels[2], channels[3], kernel_size=1, stride=1, padding=0)
        self.layer4 = BasicLayer(dim=channels[3],
                                 depth=layers[3],
                                 num_heads=channels[3] // 32,
                                 window_size=(3, 3, 5),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop_rate,
                                 attn_drop=attn_drop,
                                 drop_path=dpr[sum(layers[:3]):sum(layers[:4])],
                                 norm_layer=norm_layer,
                                 downsample=None,
                                 )

    def _make_layer_3d(self, block, conv_builder,
                       planes: int,
                       blocks: int,
                       stride: int = 1,
                       ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride,
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, conv_builder, stride, downsample)]

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.conv1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # bilinear interpolation
        x = self.bilinear_interp(x)
        # patch partition
        x = self.patch_embed(x)
        x = self.layer3(x)
        x = self.linear_embedding(x)
        x = self.layer4(x)
        return x
