import copy
from torch import nn
from typing import Union
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    ##################### YOUR CODE STARTS HERE #####################
    return round(channels*(1-prune_ratio))
    ##################### YOUR CODE ENDS HERE #####################


# function to sort the channels from important to non-important
def get_input_channel_importance(weight):
    in_channels = weight.shape[1]
    importances = []
    # compute the importance for each input channel
    for i_c in range(weight.shape[1]):
        channel_weight = weight.detach()[:, i_c]
        ##################### YOUR CODE STARTS HERE #####################
        importance = torch.linalg.norm(channel_weight)
        ##################### YOUR CODE ENDS HERE #####################
        importances.append(importance.view(1))
    return torch.cat(importances)


def count_resnet_blocks(model):
    return len(model.layer1)+len(model.layer2)+len(model.layer3)+len(model.layer4)

def count_densenet_blocks(model):
    return sum(model.block_config)+1

def count_prunable_layers(model,model_type='Vgg-16'):
    if model_type=='Vgg-16':
        return len([m for m in model.backbone if isinstance(m, nn.Conv2d)])
    

@torch.no_grad()
def apply_channel_sorting_on_vgg(model):
    model = copy.deepcopy(model)  # do not modify the original model
    # fetch all the conv and bn layers from the backbone
    all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    # iterate through conv layers
    for i_conv in range(len(all_convs) - 1):
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]
        # note that we always compute the importance according to input channels
        importance = get_input_channel_importance(next_conv.weight)
        # sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)

        # apply to previous conv and its following bn
        prev_conv.weight.copy_(torch.index_select(
            prev_conv.weight.detach(), 0, sort_idx))
        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(prev_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            )

        # apply to the next conv input (hint: one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.copy_(torch.index_select(
            next_conv.weight.detach(), 1, sort_idx))
        ##################### YOUR CODE ENDS HERE #####################

    return model

@torch.no_grad()
def apply_channel_sorting_on_resnet(model):
    model = copy.deepcopy(model)  # do not modify the original model
    # fetch all the conv and bn layers from the backbone
    all_convs = [ layer for layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
    all_bns = [ layer for layer in model.named_modules() if isinstance(layer, nn.BatchNorm2d)]
    # iterate through conv layers
    for i_conv in range(len(all_convs) - 1):
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]
        # note that we always compute the importance according to input channels
        importance = get_input_channel_importance(next_conv.weight)
        # sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)

        # apply to previous conv and its following bn
        prev_conv.weight.copy_(torch.index_select(prev_conv.weight.detach(), 0, sort_idx))

        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(prev_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            )

        # apply to the next conv input (hint: one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.copy_(torch.index_select(
            next_conv.weight.detach(), 1, sort_idx))
        ##################### YOUR CODE ENDS HERE #####################
    return model

@torch.no_grad()
def apply_channel_sorting_on_densenet(model):
    model = copy.deepcopy(model)  # do not modify the original model
    # fetch all the conv and bn layers from the backbone
    all_convs = [ layer for layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
    all_bns = [ layer for layer in model.named_modules() if isinstance(layer, nn.BatchNorm2d)]
    # iterate through conv layers
    for i_conv in range(len(all_convs) - 1):
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]
        # note that we always compute the importance according to input channels
        importance = get_input_channel_importance(next_conv.weight)
        # sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)

        # apply to previous conv and its following bn
        prev_conv.weight.copy_(torch.index_select(prev_conv.weight.detach(), 0, sort_idx))

        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(prev_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            )

        # apply to the next conv input (hint: one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.copy_(torch.index_select(
            next_conv.weight.detach(), 1, sort_idx))
        ##################### YOUR CODE ENDS HERE #####################
    return model

def apply_channel_sorting(model,model_type):
    if model_type=='Vgg-16':
        return apply_channel_sorting_on_vgg(model)
    elif model_type[:6]=='Resnet':
        return apply_channel_sorting_on_resnet(model)
    elif model_type[:9]=='Densenet':
        return apply_channel_sorting_on_densenet(model)
    else:
        print('model_type doesn\'t exists')
        exit(0)


@torch.no_grad()
def channel_prune_vgg(model,
                  prune_ratio: Union[float, dict,list]) :
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    # sanity check of provided prune_ratio
    assert isinstance(prune_ratio, (float, dict,list))
    n_conv = len([m for m in model.backbone if isinstance(m, nn.Conv2d)])
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...

    if isinstance(prune_ratio, dict):
        prune_ratio=list(prune_ratio.values())
        assert len(prune_ratio) == n_conv 
    elif isinstance(prune_ratio, float):  # convert float to list
        prune_ratio = [prune_ratio] * (n_conv )
    else:
        assert len(prune_ratio) == n_conv 
    # we prune the convs in the backbone with a uniform ratio
    # we only apply pruning to the backbone features
    all_params = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    # apply pruning. we naively keep the first k channels
    assert len(all_params) == len(all_bns)
    for i_ratio, p_ratio in enumerate(prune_ratio):
        prev_conv = all_params[i_ratio]
        prev_bn = all_bns[i_ratio]
        if i_ratio!=n_conv-1:
            next_conv = all_params[i_ratio + 1]
        original_channels = prev_conv.out_channels  # same as next_conv.in_channels

        n_keep = get_num_channels_to_keep(original_channels, p_ratio)

        # prune the output of the previous conv and bn
        prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
        prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
        prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
        prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
        prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])
        if i_ratio!=n_conv-1:
            next_conv.weight.set_(next_conv.weight.detach()[:,:n_keep])
    model.fc2.weight.set_(model.fc2.weight.detach()[:,:n_keep])
    return model

@torch.no_grad()
def channel_prune_resnet(model, prune_ratios: Union[float, dict,list]):    
    def prune_block(block, prune_ratios,prev_n_keep):
        if block.shortcut:
            block.shortcut[0].weight.set_(block.shortcut[0].weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel change

        block.conv1.weight.set_(block.conv1.weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel chang

        original_channels=block.conv1.out_channels
        n_keep = get_num_channels_to_keep(original_channels, prune_ratios)
        block.conv1.weight.set_(block.conv1.weight.detach()[:n_keep])
        block.bn1.weight.set_(block.bn1.weight.detach()[:n_keep])
        block.bn1.bias.set_(block.bn1.bias.detach()[:n_keep])
        block.bn1.running_mean.set_(block.bn1.running_mean.detach()[:n_keep])
        block.bn1.running_var.set_(block.bn1.running_var.detach()[:n_keep])

        block.conv2.weight.set_(block.conv2.weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change

        block.conv2.weight.set_(block.conv2.weight.detach()[:n_keep])
        block.bn2.weight.set_(block.bn2.weight.detach()[:n_keep])
        block.bn2.bias.set_(block.bn2.bias.detach()[:n_keep])
        block.bn2.running_mean.set_(block.bn2.running_mean.detach()[:n_keep])
        block.bn2.running_var.set_(block.bn2.running_var.detach()[:n_keep])
        if block.shortcut:
            block.shortcut[0].weight.set_(block.shortcut[0].weight.detach()[:n_keep])
            block.shortcut[1].weight.set_(block.shortcut[1].weight.detach()[:n_keep])
            block.shortcut[1].bias.set_(block.shortcut[1].bias.detach()[:n_keep])
            block.shortcut[1].running_mean.set_(block.shortcut[1].running_mean.detach()[:n_keep])
            block.shortcut[1].running_var.set_(block.shortcut[1].running_var.detach()[:n_keep])
        return n_keep #we will need n_keep to fix next conv' inchannels fixing

    def prune_bn_block(block, prune_ratios, prev_n_keep):
        if block.shortcut:
            block.shortcut[0].weight.set_(block.shortcut[0].weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel change

        block.conv1.weight.set_(block.conv1.weight.detach()[:,:prev_n_keep]) #fixing number of inchannels due to previous channel chang

        original_channels=block.conv1.out_channels
        n_keep = get_num_channels_to_keep(original_channels, prune_ratios)
        block.conv1.weight.set_(block.conv1.weight.detach()[:n_keep])
        block.bn1.weight.set_(block.bn1.weight.detach()[:n_keep])
        block.bn1.bias.set_(block.bn1.bias.detach()[:n_keep])
        block.bn1.running_mean.set_(block.bn1.running_mean.detach()[:n_keep])
        block.bn1.running_var.set_(block.bn1.running_var.detach()[:n_keep])

        block.conv2.weight.set_(block.conv2.weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change

        block.conv2.weight.set_(block.conv2.weight.detach()[:n_keep])
        block.bn2.weight.set_(block.bn2.weight.detach()[:n_keep])
        block.bn2.bias.set_(block.bn2.bias.detach()[:n_keep])
        block.bn2.running_mean.set_(block.bn2.running_mean.detach()[:n_keep])
        block.bn2.running_var.set_(block.bn2.running_var.detach()[:n_keep])

        block.conv3.weight.set_(block.conv3.weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change
        
        block.conv3.weight.set_(block.conv3.weight.detach()[:n_keep*block.expansion])
        block.bn3.weight.set_(block.bn3.weight.detach()[:n_keep*block.expansion])
        block.bn3.bias.set_(block.bn3.bias.detach()[:n_keep*block.expansion])
        block.bn3.running_mean.set_(block.bn3.running_mean.detach()[:n_keep*block.expansion])
        block.bn3.running_var.set_(block.bn3.running_var.detach()[:n_keep*block.expansion])
        if block.shortcut:
            block.shortcut[0].weight.set_(block.shortcut[0].weight.detach()[:n_keep*block.expansion]) #fixing number of inchannels due to previous channel change
            block.shortcut[1].weight.set_(block.shortcut[1].weight.detach()[:n_keep*block.expansion])
            block.shortcut[1].bias.set_(block.shortcut[1].bias.detach()[:n_keep*block.expansion])
            block.shortcut[1].running_mean.set_(block.shortcut[1].running_mean.detach()[:n_keep*block.expansion])
            block.shortcut[1].running_var.set_(block.shortcut[1].running_var.detach()[:n_keep*block.expansion])
        return n_keep*block.expansion #we will need n_keep to fix next conv' inchannels fixing

    assert isinstance(prune_ratios, (float, dict,list))
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratios, dict):
        prune_ratios=list(prune_ratios.values())
        assert len(prune_ratios) == count_resnet_blocks(model)+1
    elif isinstance(prune_ratios, float):  # convert float to list
        prune_ratios = [prune_ratios] * (count_resnet_blocks(model)+1)
    else:
        pass
    original_channels=model.conv1.out_channels
    n_keep = get_num_channels_to_keep(original_channels, prune_ratios[0])
    model.conv1.weight.set_(model.conv1.weight.detach()[:n_keep])
    model.bn1.weight.set_(model.bn1.weight.detach()[:n_keep])
    model.bn1.bias.set_(model.bn1.bias.detach()[:n_keep])
    model.bn1.running_mean.set_(model.bn1.running_mean.detach()[:n_keep])
    model.bn1.running_var.set_(model.bn1.running_var.detach()[:n_keep])
    i=1
    for name, layer in model.named_children():
        if isinstance(layer,nn.Sequential):
            for b in layer:
                p_ratios=prune_ratios[i]
                if hasattr(b, 'conv3'):
                    n_keep=prune_bn_block(b,p_ratios,n_keep)
                else:
                    n_keep=prune_block(b,p_ratios,n_keep)
                i=i+1
    
    model.fc.weight.set_(model.fc.weight.detach()[:,:n_keep]) #fixing number of inchannels due to previous channel change
    return model

@torch.no_grad()
def channel_prune_densenet(model, prune_ratios: Union[float, dict, list]):
    """Apply channel pruning to DenseNet model following ResNet-style block pruning.
    Prunes each dense block and transition layer sequentially.
    """
    def prune_denselayer(layer, n_keep_in):
        """Prune a single DenseLayer's input and internal conv1."""
        # Prune norm1 and conv1 input channels
        layer.norm1.weight.set_(layer.norm1.weight.detach()[:n_keep_in])
        layer.norm1.bias.set_(layer.norm1.bias.detach()[:n_keep_in])
        layer.norm1.running_mean.set_(layer.norm1.running_mean.detach()[:n_keep_in])
        layer.norm1.running_var.set_(layer.norm1.running_var.detach()[:n_keep_in])
        
        layer.conv1.weight.set_(layer.conv1.weight.detach()[:, :n_keep_in])
        
        # conv1 output goes to norm2, keep all channels from conv1
        # (conv1 -> norm2 -> relu2 -> conv2, growth_rate output to next layer)

    def prune_denseblock(block, n_keep_in, prune_ratio):
        """Prune all layers in a DenseBlock sequentially."""
        current_in = n_keep_in
        for layer_name, layer in block.items():
            prune_denselayer(layer, current_in)
            # After each layer: input grows by growth_rate (concatenation)
            current_in = n_keep_in + (int(layer_name.replace('denselayer', '')) * layer.conv2.out_channels)
        return current_in

    def prune_transition(trans, n_keep_in, prune_ratio):
        """Prune transition layer."""
        trans.norm.weight.set_(trans.norm.weight.detach()[:n_keep_in])
        trans.norm.bias.set_(trans.norm.bias.detach()[:n_keep_in])
        trans.norm.running_mean.set_(trans.norm.running_mean.detach()[:n_keep_in])
        trans.norm.running_var.set_(trans.norm.running_var.detach()[:n_keep_in])
        
        out_channels = trans.conv.out_channels
        n_keep = get_num_channels_to_keep(out_channels, prune_ratio)
        trans.conv.weight.set_(trans.conv.weight.detach()[:n_keep, :n_keep_in])
        
        return n_keep

    assert isinstance(prune_ratios, (float, dict, list))
    
    # Count pruning stages: conv0 + num_denseblocks + num_transitions
    num_denseblocks = sum(1 for layer in model.features if layer.__class__.__name__ == "DenseBlock")
    num_transitions = sum(1 for layer in model.features if layer.__class__.__name__ == "Transition")
    expected_len = 1 + num_denseblocks + num_transitions
    
    if isinstance(prune_ratios, dict):
        prune_ratios = list(prune_ratios.values())
        assert len(prune_ratios) == expected_len
    elif isinstance(prune_ratios, float):
        prune_ratios = [prune_ratios] * expected_len
    else:
        assert len(prune_ratios) == expected_len
    
    # Prune initial conv0
    n_keep = get_num_channels_to_keep(model.features.conv0.out_channels, prune_ratios[0])
    model.features.conv0.weight.set_(model.features.conv0.weight.detach()[:n_keep])
    model.features.norm0.weight.set_(model.features.norm0.weight.detach()[:n_keep])
    model.features.norm0.bias.set_(model.features.norm0.bias.detach()[:n_keep])
    model.features.norm0.running_mean.set_(model.features.norm0.running_mean.detach()[:n_keep])
    model.features.norm0.running_var.set_(model.features.norm0.running_var.detach()[:n_keep])
    
    current_features = n_keep
    stage_idx = 1
    
    # Iterate through denseblocks and transitions
    for name, layer in model.features.named_children():
        layer_type = layer.__class__.__name__
        
        if layer_type == "DenseBlock":
            current_features = prune_denseblock(layer, current_features, prune_ratios[stage_idx])
            stage_idx += 1
        elif layer_type == "Transition":
            current_features = prune_transition(layer, current_features, prune_ratios[stage_idx])
            stage_idx += 1
    
    # Prune final batch norm and classifier
    model.features.norm5.weight.set_(model.features.norm5.weight.detach()[:current_features])
    model.features.norm5.bias.set_(model.features.norm5.bias.detach()[:current_features])
    model.features.norm5.running_mean.set_(model.features.norm5.running_mean.detach()[:current_features])
    model.features.norm5.running_var.set_(model.features.norm5.running_var.detach()[:current_features])
    
    model.classifier.weight.set_(model.classifier.weight.detach()[:, :current_features])
    
    return model

def channel_prune(model, prune_ratio: Union[dict, float],model_type):
    if model_type=='Vgg-16':
        return channel_prune_vgg(model, prune_ratio)
    elif model_type[:6]=='Resnet':
        return channel_prune_resnet(model, prune_ratio)
    elif model_type[:9]=='Densenet':
        return channel_prune_densenet(model, prune_ratio)        
    else:
        print('model_type doesn\'t exists')
        exit(0)

def ChannelPrunner(model,channel_pruning_ratio,model_type):
    sorted_model = apply_channel_sorting(model,model_type)
    pruned_model = channel_prune(sorted_model, channel_pruning_ratio,model_type)
    return pruned_model