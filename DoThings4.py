import copy
from multiprocessing.dummy import freeze_support
import pickle
from pyexpat import model

import torch

from PruningNAS.DataProcess.DataPreprocessing import get_dataloaders
from PruningNAS.Models.DenseNet import DenseNet121
from PruningNAS.Models.ResNetBasic import *
from PruningNAS.Models.ResNetBottleNeck import *
from PruningNAS.PruneEvaluator import load_and_print_accuracies
from PruningNAS.Utills.PrunUtillCP import apply_channel_sorting_on_resnet, channel_prune_resnet
from PruningNAS.Utills.TrainingModulesUtills import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Initialize the model
    basedir='PruningNAS'
    path='./dataset/cifar10'
    select_model='Resnet-50'
    prune_type='CP'
    #model_path=f'{basedir}/checkpoint/vgg_mrl_99.51375579833984.pth'
    model_path=r'PruningNAS\checkpoint\Resnet-50\Resnet-50_cifar_95.750000.pth'
    # Load the saved state_dict correctly
    model = torch.load(model_path, map_location=torch.device(device),weights_only=False)  # Use 'cpu' if necessary
    model.to(device)

    train_dataloader,test_dataloader=get_dataloaders(path )
    dense_model_accuracy,_=evaluate(model,test_dataloader)
    print(f"Original model accuracy: {dense_model_accuracy:.4f}")

    sorted_model=copy.deepcopy(model)
    sorted_model = apply_channel_sorting_on_resnet(sorted_model)
    sorted_model.eval()  # Set to evaluation mode
    # sorted_model = channel_prune_resnet(sorted_model,[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


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

    model_accuracy,_ =evaluate(model,test_dataloader)
    print("Original Model Accuracy:",model_accuracy)

    sorted_model_accuracy,_ =evaluate(sorted_model,test_dataloader)  
    print("Sorted Model Accuracy:",sorted_model_accuracy)



if __name__ == '__main__':
    freeze_support()
    main() 
