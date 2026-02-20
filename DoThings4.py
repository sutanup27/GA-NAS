from copy import copy
import pickle
from pyexpat import model

import torch

from PruningNAS.Models.DenseNet import DenseNet121
from PruningNAS.PruneEvaluator import load_and_print_accuracies
from PruningNAS.Utills.PrunUtillCP import apply_channel_sorting_on_densenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=DenseNet121(num_classes=10)
sorted_model=copy.deepcopy(model)
sorted_model =apply_channel_sorting_on_densenet(model, 0.7, device)
dummy_input = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    dummy_output = model(dummy_input)    
print("\nOutput shape:", dummy_output.shape)
print("Output:", dummy_output)


with torch.no_grad():
    dummy_output = sorted_model(dummy_input)    
print("\nOutput shape:", dummy_output.shape)
print("Output:", dummy_output)