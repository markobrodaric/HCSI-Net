# --------------------------------------------------------
# Based on EfficientNet-PyTorch
# Original authors: lukemelas et al.
#
# Modified for HCSI-Net stage-wise backbone access.
# --------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ENet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size,
)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck block used in EfficientNet.
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()

        self.block_args = block_args
        self.bn_momentum = 1 - global_params.batch_norm_momentum
        self.bn_epsilon = global_params.batch_norm_epsilon
        self.has_se = (
            self.block_args.se_ratio is not None
            and 0 < self.block_args.se_ratio <= 1
        )
        self.use_skip = block_args.id_skip

        input_channels = self.block_args.input_filters
        expanded_channels = input_channels * self.block_args.expand_ratio

        if self.block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self.expand_conv = Conv2d(
                in_channels=input_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                bias=False,
            )
            self.bn0 = nn.BatchNorm2d(
                num_features=expanded_channels,
                momentum=self.bn_momentum,
                eps=self.bn_epsilon,
            )

        kernel_size = self.block_args.kernel_size
        stride = self.block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.depthwise_conv = Conv2d(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            groups=expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=expanded_channels,
            momentum=self.bn_momentum,
            eps=self.bn_epsilon,
        )
        image_size = calculate_output_image_size(image_size, stride)

        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            squeezed_channels = max(
                1, int(self.block_args.input_filters * self.block_args.se_ratio)
            )
            self.se_reduce = Conv2d(
                in_channels=expanded_channels,
                out_channels=squeezed_channels,
                kernel_size=1,
            )
            self.se_expand = Conv2d(
                in_channels=squeezed_channels,
                out_channels=expanded_channels,
                kernel_size=1,
            )

        output_channels = self.block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.project_conv = Conv2d(
            in_channels=expanded_channels,
            out_channels=output_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=output_channels,
            momentum=self.bn_momentum,
            eps=self.bn_epsilon,
        )

        self.swish = MemoryEfficientSwish()

    def forward(self, x: torch.Tensor, drop_connect_rate: float | None = None) -> torch.Tensor:
        identity = x

        if self.block_args.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.swish(x)

        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.swish(x)

        if self.has_se:
            se = F.adaptive_avg_pool2d(x, 1)
            se = self.se_reduce(se)
            se = self.swish(se)
            se = self.se_expand(se)
            x = torch.sigmoid(se) * x

        x = self.project_conv(x)
        x = self.bn2(x)

        same_shape = (
            self.block_args.stride == 1
            and self.block_args.input_filters == self.block_args.output_filters
        )

        if self.use_skip and same_shape:
            if drop_connect_rate is not None and drop_connect_rate > 0:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + identity

        return x

    def set_swish(self, memory_efficient: bool = True) -> None:
        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()


class StagewiseEfficientNetB4(nn.Module):
    """
    EfficientNet-B4 backbone with explicit stage-wise access for HCSI-Net.

    Stage convention:
      stage 0 -> stem + blocks 0..3     -> (B, 32, 48, 48)
      stage 1 -> blocks 4..7            -> (B, 56, 24, 24)
      stage 2 -> blocks 8..17           -> (B, 160, 12, 12)
      stage 3 -> blocks 18..26          -> (B, 272, 12, 12)
      stage 4 -> blocks 27..35 + head   -> (B, 1792, 6, 6)

    Input convention:
      - stage 0 expects an image tensor       (B, 3, H, W)
      - stages 1..4 expect stage feature maps
    """

    STAGE_BLOCK_RANGES = (
        (0, 3),
        (4, 7),
        (8, 17),
        (18, 26),
        (27, 31),
    )

    def __init__(self, blocks_args, global_params, in_channels: int = 3):
        super().__init__()

        if not isinstance(blocks_args, list) or len(blocks_args) == 0:
            raise ValueError("blocks_args must be a non-empty list.")

        self.global_params = global_params
        self.blocks_args = blocks_args
        self.num_stages = len(self.STAGE_BLOCK_RANGES)

        bn_momentum = 1 - self.global_params.batch_norm_momentum
        bn_epsilon = self.global_params.batch_norm_epsilon

        image_size = self.global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        stem_out_channels = round_filters(32, self.global_params)
        self.conv_stem = Conv2d(
            in_channels,
            stem_out_channels,
            kernel_size=3,
            stride=2,
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(
            num_features=stem_out_channels,
            momentum=bn_momentum,
            eps=bn_epsilon,
        )
        image_size = calculate_output_image_size(image_size, 2)

        self.blocks = nn.ModuleList()
        for block_args in self.blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self.global_params),
                output_filters=round_filters(block_args.output_filters, self.global_params),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params),
            )

            self.blocks.append(MBConvBlock(block_args, self.global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)

            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters,
                    stride=1,
                )

            for _ in range(block_args.num_repeat - 1):
                self.blocks.append(MBConvBlock(block_args, self.global_params, image_size=image_size))

        head_in_channels = block_args.output_filters
        head_out_channels = round_filters(1280, self.global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.conv_head = Conv2d(
            head_in_channels,
            head_out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=head_out_channels,
            momentum=bn_momentum,
            eps=bn_epsilon,
        )

        self.swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient: bool = True) -> None:
        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self.blocks:
            block.set_swish(memory_efficient)

    def _run_block_range(
        self,
        x: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        for block_idx in range(start_idx, end_idx + 1):
            drop_rate = self.global_params.drop_connect_rate
            if drop_rate:
                drop_rate *= float(block_idx) / len(self.blocks)
            x = self.blocks[block_idx](x, drop_connect_rate=drop_rate)
        return x

    def forward_stage(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        """
        Run exactly one EfficientNet stage.

        Args:
            x:
              - image tensor for stage 0
              - feature tensor for stages 1..4
            stage:
              integer in [0, 4]
        """
        if not 0 <= stage < self.num_stages:
            raise ValueError(f"stage must be in [0, {self.num_stages - 1}], got {stage}.")

        start_idx, end_idx = self.STAGE_BLOCK_RANGES[stage]

        if stage == 0:
            x = self.swish(self.bn0(self.conv_stem(x)))

        x = self._run_block_range(x, start_idx, end_idx)

        if stage == self.num_stages - 1:
            x = self.swish(self.bn1(self.conv_head(x)))

        return x

    def extract_features(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        return self.forward_stage(x, stage)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the full EfficientNet-B4 backbone and return the final feature map.
        """
        for stage in range(self.num_stages):
            x = self.forward_stage(x, stage)
        return x


def build_efficientnet_backbone(
    in_channels: int = 3,
) -> StagewiseEfficientNetB4:
    """
    Build the EfficientNet-B4 backbone used in HCSI-Net.
    """
    blocks_args, global_params = get_model_params(
        "efficientnet-b4",
        {"include_top": False},
    )

    model = StagewiseEfficientNetB4(
        blocks_args=blocks_args,
        global_params=global_params,
        in_channels=in_channels,
    )

    return model