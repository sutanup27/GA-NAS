import torch
from PruningNAS.Models.ResNetBottleNeck import ResNet50
from PruningNAS.Utills.PrunUtillCP import ChannelPrunner, count_resnet_blocks

# Initialize ResNet-50 model from the proper file
model = ResNet50(classes=10)
model.eval()

# Example: create dummy input data (batch size 1, 3 channels, 32x32 for CIFAR-10)
dummy_input = torch.randn(1, 3, 32, 32)

# Prune the model using ChannelPrunner
prune_ratio_count = count_resnet_blocks(model) + 1
sparsity_dict = [0.7] * prune_ratio_count
pruned_model = ChannelPrunner(model, sparsity_dict, 'Resnet-50')
pruned_model.eval()

# Test the pruned model with dummy input
for name, module in pruned_model.named_modules():
    print(f"Module: {name}, shape: {list(module.parameters())[0].shape if list(module.parameters()) else 'N/A'  }")
dummy_output = pruned_model(dummy_input)
print('Output shape:', dummy_output.shape)
