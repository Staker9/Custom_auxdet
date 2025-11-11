# local_backbone_resnet.py
import torch
import torch.nn as nn
from typing import Tuple, List

from local_layers import ConvModule, BaseModule
from Custom.local_mmdet.registry import MODELS

# ------------------------------------------------------------
# Bottleneck block (ResNet-50/101 style, expansion=4)
# ------------------------------------------------------------
class Bottleneck(BaseModule):
    expansion = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        norm_cfg: dict = None,
        act_cfg: dict = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.conv1 = ConvModule(inplanes, planes, kernel_size=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
        self.conv2 = ConvModule(planes, planes, kernel_size=3, stride=stride, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
        # 마지막 1x1은 보통 활성화 없이 BN까지만 두지만, ConvModule 일관성 위해 act를 끔
        self.conv3 = ConvModule(planes, planes * self.expansion, kernel_size=1,
                                norm_cfg=norm_cfg, act_cfg=None, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# ------------------------------------------------------------
# ResNet-50 backbone (C2~C5 출력)
#   - stem: 7x7 stride2 conv + 3x3 maxpool stride2  → /4
#   - layer1..4 strides: [1,2,2,2] → /4,/8,/16,/32
#   - channels: [64,128,256,512] with expansion=4
# ------------------------------------------------------------
@MODELS.register_module()
class ResNetV1cLocal(BaseModule):
    """ResNet-50 like backbone without mmcv/mmengine deps.
    Args:
        depth (int): 50 or 101 (block counts [3,4,6,3] or [3,4,23,3])
        in_channels (int): input image channels
        stem_channels (int): first conv output channels
        base_channels (int): layer1 inner channels (before expansion)
        norm_cfg (dict): passed to ConvModule
        act_cfg (dict): passed to ConvModule
        out_indices (tuple): which stages to return (default (0,1,2,3) -> C2..C5)
        frozen_stages (int): freeze 0=none, 1=stem, 2=layer1, ... up to 5
    """
    arch_settings = {
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
    }

    def __init__(
        self,
        depth: int = 50,
        in_channels: int = 3,
        stem_channels: int = 64,
        base_channels: int = 64,
        norm_cfg: dict = None,
        act_cfg: dict = dict(type='ReLU'),  # local ConvModule가 인식하는 형태라면 그대로 사용
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        frozen_stages: int = 0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert depth in self.arch_settings, f'Unsupported depth: {depth}'
        block, stage_blocks = self.arch_settings[depth]

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # ---- Stem ----
        self.stem = ConvModule(
            in_channels, stem_channels, kernel_size=7, stride=2, padding=3,
            norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- Stages ----
        self.inplanes = stem_channels
        self.layer1 = self._make_layer(block, base_channels,  stage_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2, stage_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels*4, stage_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels*8, stage_blocks[3], stride=2)

        # 출력 채널 기록 (C2..C5)
        self.out_channels = [
            base_channels * block.expansion,        # 64*4=256
            base_channels * 2 * block.expansion,    # 128*4=512
            base_channels * 4 * block.expansion,    # 256*4=1024
            base_channels * 8 * block.expansion,    # 512*4=2048
        ]

        # stage freeze 설정(옵션)
        self._freeze_stages()

    # --------------------------------------------------------
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # shortcut이 채널/stride가 다르면 projection
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ConvModule(self.inplanes, planes * block.expansion,
                           kernel_size=1, stride=stride,
                           norm_cfg=self.norm_cfg, act_cfg=None, inplace=False)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample,
                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, downsample=None,
                      norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            )
        return nn.Sequential(*layers)

    # --------------------------------------------------------
    def _freeze_stages(self):
        # 1: stem, 2: layer1, 3: layer2, 4: layer3, 5: layer4
        if self.frozen_stages >= 1:
            for p in self.stem.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 2:
            for p in self.layer1.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 3:
            for p in self.layer2.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 4:
            for p in self.layer3.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 5:
            for p in self.layer4.parameters():
                p.requires_grad = False

    # --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # stem → /4
        x = self.stem(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)   # /4,  256
        c3 = self.layer2(c2)  # /8,  512
        c4 = self.layer3(c3)  # /16, 1024
        c5 = self.layer4(c4)  # /32, 2048

        outs: List[torch.Tensor] = [c2, c3, c4, c5]
        return tuple(outs[i] for i in self.out_indices)
