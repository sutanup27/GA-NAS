import copy
import pickle
from pyexpat import model

import torch

from PruningNAS.Models.DenseNet import DenseNet121
from PruningNAS.PruneEvaluator import load_and_print_accuracies
from PruningNAS.Utills.PrunUtillCP import apply_channel_sorting_on_densenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=DenseNet121(classes=10)
sorted_model=copy.deepcopy(model)
sorted_model =apply_channel_sorting_on_densenet(sorted_model)
dummy_input = torch.randn(20, 3, 32, 32).to(device)




print("Original Model Architecture:")
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
        params = list(module.parameters())
        if params:
            print(f"{name}: {params[0].shape}")

print("Sorted Model Architecture:")
for name, module in sorted_model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
        params = list(module.parameters())
        if params:
            print(f"{name}: {params[0].shape}")

model.to(device)
sorted_model.to(device)
with torch.no_grad():
    dummy_output = model(dummy_input)    
print("\nOutput shape:", dummy_output.shape)
print("Output:", dummy_output)


with torch.no_grad():
    dummy_output = sorted_model(dummy_input)    
print("\nOutput shape:", dummy_output.shape)
print("Output:", dummy_output)