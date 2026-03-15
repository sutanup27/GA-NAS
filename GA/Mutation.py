"""
Mutation.py
-----------
One mutation method per block type defined in Blocks.py.

Each method:
  - Accepts an instantiated block (nn.Module) as its first argument.
  - Randomly (or explicitly) picks new architectural values:
      out_channels, hidden channels, stride, dilation, expand_ratio, ...
  - Rebuilds affected layers with the new shapes.
  - Transplants weights from old layers into new ones wherever sizes overlap
    (preserving as much learned knowledge as possible).
  - Returns the mutated block (a *new* nn.Module, same type as input).
"""

import random
import torch
import torch.nn as nn

from GA.Blocks import (
    BLOCKS,
    Conv3x3,
    DilConv,
    DepthwiseSeparableConv,
    SepConv,
    InvertedResidual,
    BasicBlock,
    Bottleneck,
    FactorizedReduce,
)


def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    return round(channels*(1-prune_ratio))


# function to sort the channels from important to non-important
def get_input_channel_importance(weight):
    in_channels = weight.shape[1]
    importances = []
    # compute the importance for each input channel
    for i_c in range(weight.shape[0]):
        channel_weight = weight.detach()[i_c]
        importance = torch.linalg.norm(channel_weight) # L2 norm as importance score
        importances.append(importance.view(1))
    return torch.cat(importances)

@torch.no_grad()
def sort_channels_by_importance(weight):
    importances = get_input_channel_importance(weight)
    sorted_indices = torch.argsort(importances, descending=True)
    weight.copy_(torch.index_select(
            weight.detach(), 0, sorted_indices))
    return sorted_indices

@torch.no_grad()
def sort_bn_channels_by_importance(block: nn.BatchNorm2d, sorted_indices):
    for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
        tensor_to_apply = getattr(block, tensor_name)
        tensor_to_apply.copy_(
            torch.index_select(tensor_to_apply.detach(), 0, sorted_indices)
        )

def prune_or_pad_weight(weight, num_channels_to_keep):
    if weight.shape[0] > num_channels_to_keep:
        weight.set_(weight.detach()[:num_channels_to_keep])
    else:        # Pad the weight tensor with zeros to increase the number of channels
        padding_size = num_channels_to_keep - weight.shape[0]
        padding = torch.zeros((padding_size, *weight.shape[1:]), device=weight.device)
        weight.set_(torch.cat([weight, padding], dim=0))

def prune_or_pad_weight_in(weight, num_channels_to_keep):
    if weight.shape[1] > num_channels_to_keep:
        weight.set_(weight.detach()[:, :num_channels_to_keep])
    else:        # Pad the weight tensor with zeros to increase the number of channels
        padding_size = num_channels_to_keep - weight.shape[1]
        padding = torch.zeros((weight.shape[0], padding_size, *weight.shape[2:]), device=weight.device)
        weight.set_(torch.cat([weight, padding], dim=1))

def prune_or_pad_bn(block: nn.BatchNorm2d, num_channels_to_keep):
    for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
        tensor_to_apply = getattr(block, tensor_name)
        if tensor_to_apply.shape[0] > num_channels_to_keep:
            tensor_to_apply.set_(tensor_to_apply.detach()[:num_channels_to_keep])
        else:
            padding_size = num_channels_to_keep - tensor_to_apply.shape[0]

            if tensor_name == 'weight' or tensor_name == 'running_var':
                pad = torch.ones((padding_size,), device=tensor_to_apply.device, dtype=tensor_to_apply.dtype)
            else:  # bias, running_mean
                pad = torch.zeros((padding_size,), device=tensor_to_apply.device, dtype=tensor_to_apply.dtype)

            tensor_to_apply.set_(torch.cat([tensor_to_apply, pad], dim=0))

@torch.no_grad()
def mutate_conv3x3(block: Conv3x3, new_out_channels=None,stride=None) :
    if new_out_channels is not None:
        sorted_indices = sort_channels_by_importance(block.conv.weight)
        sort_bn_channels_by_importance(block.bn, sorted_indices)
        prune_or_pad_weight(block.conv.weight, new_out_channels)
        prune_or_pad_bn(block.bn, new_out_channels)
        block.out_ch = new_out_channels
    if stride is not None:
        if stride == 1:
            block.stride = 2
            block.conv.stride = (2, 2)
        if stride == 2:
            block.stride = 1
            block.conv.stride = (1, 1)

@torch.no_grad()
def mutate_dilconv(block: DilConv, new_out_channels=None, stride=None) :
    if new_out_channels is not None:
        sorted_indices = sort_channels_by_importance(block.pw.weight)
        sort_bn_channels_by_importance(block.bn2, sorted_indices)
        prune_or_pad_weight(block.pw.weight, new_out_channels)
        prune_or_pad_bn(block.bn2, new_out_channels)
        block.out_ch = new_out_channels
    if stride is not None:
        if stride == 1:
            block.stride = 2
            block.dw.stride = (2, 2)
        if stride == 2:
            block.stride = 1
            block.dw.stride = (1, 1)

@torch.no_grad()
def mutate_depthwise_separable_conv(block: DepthwiseSeparableConv, new_out_channels=None, stride=None):
    if new_out_channels is not None:
        sorted_indices = sort_channels_by_importance(block.pw.weight)
        sort_bn_channels_by_importance(block.bn2, sorted_indices)
        prune_or_pad_weight(block.pw.weight, new_out_channels)
        prune_or_pad_bn(block.bn2, new_out_channels)
        block.out_ch = new_out_channels
    if stride is not None:
        if stride == 1:
            block.stride = 2
            block.dw.stride = (2, 2)
        if stride == 2:
            block.stride = 1
            block.dw.stride = (1, 1)

@torch.no_grad()
def mutate_sepconv(block: SepConv, new_out_channels=None, stride=None) :
    if new_out_channels is not None:
        sorted_indices = sort_channels_by_importance(block.pw2.weight)
        sort_bn_channels_by_importance(block.bn4, sorted_indices)
        prune_or_pad_weight(block.pw2.weight, new_out_channels)
        prune_or_pad_bn(block.bn4, new_out_channels)
        block.out_ch = new_out_channels
    if stride is not None:
        if stride == 1:
            block.stride = 2
            block.dw1.stride = (2, 2)
        if stride == 2:
            block.stride = 1
            block.dw1.stride = (1, 1)


@torch.no_grad()
def mutate_inverted_residual(block: InvertedResidual, new_out_channels=None, new_hidden_channels=None, residual=None) :
    if new_hidden_channels is not None:
        sorted_indices = sort_channels_by_importance(block.expand.weight)
        sort_bn_channels_by_importance(block.bn1, sorted_indices)
        prune_or_pad_weight(block.expand.weight, new_hidden_channels)
        prune_or_pad_bn(block.bn1, new_hidden_channels)

        sort_channels_by_importance(block.dw.weight)
        sort_bn_channels_by_importance(block.bn2, sorted_indices)  
        prune_or_pad_weight(block.dw.weight, new_hidden_channels)
        prune_or_pad_bn(block.bn2, new_hidden_channels)
        block.dw.groups = new_hidden_channels
        prune_or_pad_weight_in(block.project.weight, new_hidden_channels)
        block.hidden_ch = new_hidden_channels
    if new_out_channels is not None:
        sorted_indices = sort_channels_by_importance(block.project.weight)
        sort_bn_channels_by_importance(block.bn3, sorted_indices)
        prune_or_pad_weight(block.project.weight, new_out_channels)
        prune_or_pad_bn(block.bn3, new_out_channels)
        block.out_ch = new_out_channels
    if residual is not None:
        if block.use_residual == True:
            block.use_residual = False
        else:
            block.use_residual = True
        
    
@torch.no_grad()
def mutate_basic_block(block: BasicBlock, new_out_channels=None, stride=None) :
    if new_out_channels is not None:
        sorted_indices = sort_channels_by_importance(block.conv1.weight)
        sort_bn_channels_by_importance(block.bn1, sorted_indices)
        prune_or_pad_weight(block.conv1.weight, new_out_channels)
        prune_or_pad_bn(block.bn1, new_out_channels)

        prune_or_pad_weight_in(block.conv2.weight, new_out_channels)

        sorted_indices = sort_channels_by_importance(block.conv2.weight)
        sort_bn_channels_by_importance(block.bn2, sorted_indices)
        prune_or_pad_weight(block.conv2.weight, new_out_channels)
        prune_or_pad_bn(block.bn2, new_out_channels)
        if block.downsample is not None:
            sorted_indices = sort_channels_by_importance(block.downsample[0].weight)
            sort_bn_channels_by_importance(block.downsample[1], sorted_indices)
            prune_or_pad_weight(block.downsample[0].weight, new_out_channels)
            prune_or_pad_bn(block.downsample[1], new_out_channels)
        block.out_ch = new_out_channels
    if stride is not None:
        if stride == 1:
            block.stride = 2
            block.conv1.stride = (2, 2)
            block.downsample[0].stride = (2, 2)
        if stride == 2:
            block.stride = 1
            block.conv1.stride = (1, 1)
            block.downsample[0].stride = (1, 1)


@torch.no_grad()
def mutate_bottleneck(block: Bottleneck, new_out_channels=None, new_hidden_channels=None, stride=None) -> Bottleneck:
    if new_hidden_channels is not None:
        sorted_indices = sort_channels_by_importance(block.conv1.weight)
        sort_bn_channels_by_importance(block.bn1, sorted_indices)
        prune_or_pad_weight(block.conv1.weight, new_hidden_channels)
        prune_or_pad_bn(block.bn1, new_hidden_channels)

        sorted_indices = sort_channels_by_importance(block.conv2.weight)
        sort_bn_channels_by_importance(block.bn2, sorted_indices)
        prune_or_pad_weight(block.conv2.weight, new_hidden_channels)
        prune_or_pad_bn(block.bn2, new_hidden_channels)
        prune_or_pad_weight_in(block.conv3.weight, new_hidden_channels)
        block.hidden_ch = new_hidden_channels

    if new_out_channels is not None:
        sorted_indices = sort_channels_by_importance(block.conv3.weight)
        sort_bn_channels_by_importance(block.bn3, sorted_indices)
        prune_or_pad_weight(block.conv3.weight, new_out_channels)
        prune_or_pad_bn(block.bn3, new_out_channels)

        if block.downsample is not None:
            sorted_indices = sort_channels_by_importance(block.downsample[0].weight)
            sort_bn_channels_by_importance(block.downsample[1], sorted_indices)
            prune_or_pad_weight(block.downsample[0].weight, new_out_channels)
            prune_or_pad_bn(block.downsample[1], new_out_channels)

        block.out_ch = new_out_channels
    if stride is not None:
        if stride == 1:
            block.stride = 2
            block.conv2.stride = (2, 2)
            block.downsample[0].stride = (2, 2)
        if stride == 2:
            block.stride = 1
            block.conv2.stride = (1, 1)
            block.downsample[0].stride = (1, 1)

@torch.no_grad()
def mutate_factorized_reduce(block: FactorizedReduce, new_out_channels=None) -> FactorizedReduce:
    if new_out_channels is not None:
        sorted_indices = sort_channels_by_importance(block.conv1.weight)
        sort_bn_channels_by_importance(block.bn, sorted_indices)
        prune_or_pad_weight(block.conv1.weight, new_out_channels//2)
        prune_or_pad_weight(block.conv2.weight, new_out_channels//2)
        prune_or_pad_bn(block.bn, new_out_channels)
        block.out_ch = new_out_channels