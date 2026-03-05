import copy
import random
from multiprocessing import freeze_support


import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *

from PruningNAS.Models.MobileNetV1 import MobileNetV1
from PruningNAS.Models.MobileNetV2 import MobileNetV2
from .Models.ResNetBasic import *
from .Models.ResNetBottleNeck import *
from .Models.DenseNet import *
from .Utills.TrainingModulesUtills import evaluate
from .DataProcess.DataPreprocessing import get_dataloaders
from .Models.VGG import *
from .Utills.TrainingModulesUtills import Training
from .Utills.ViewerUtills import plot_accuracy, plot_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    seed=0
    random.seed(seed)
    basedir='PruningNAS'

    path='./dataset/cifar10'
    classes=10
    train_dataloader, test_dataloader = get_dataloaders(path, batch_size=128)

    select_model='MobilenetV1'
    if select_model=='Vgg-16':
        model=VGG(classes=classes)
    elif select_model=='Resnet-18':
        model = ResNet18(classes=classes)
    elif select_model=='Resnet-34':
        model = ResNet34(classes=classes)
    elif select_model=='Resnet-50':
        model = ResNet50(classes=classes)
    elif select_model=='Resnet-101':
        model = ResNet101(classes=classes)
    elif select_model=='Resnet-152':
        model = ResNet152(classes=classes)
    elif select_model=='Densenet-121':
        model = DenseNet121(classes=classes)
    elif select_model=='Densenet-169':
        model = DenseNet169(classes=classes)
    elif select_model=='Densenet-201':
        model = DenseNet201(classes=classes)
    elif select_model=='MobilenetV1':
        model = MobileNetV1(classes=classes)
    elif select_model=='MobilenetV2':
        model = MobileNetV2(classes=classes)
    else:
        print("Model does not exists")
        exit()

    # ########load from path only for retraining #####
    # model_path=r'PruningNAS\checkpoint\MobilenetV1\MobilenetV1_cifar_85.909996.pth'
    # model = torch.load(model_path, map_location=torch.device(device),weights_only=False)  # Use 'cpu' if necessary
    # ################################################
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD( model.parameters(), lr=0.01,  momentum=0.9,  weight_decay=5e-4,)

    num_epochs = 300
    scheduler = CosineAnnealingLR(optimizer, num_epochs)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50)

    best_model, losses, test_losses, accs=Training( model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=num_epochs,scheduler=scheduler)

    model=copy.deepcopy(best_model)
    metric,_ = evaluate(model, test_dataloader)
    print(f"Best model accuray:", metric)

    plot_accuracy(accs,save_path=f'{basedir}/checkpoint/{select_model}/accuracy.png')
    plot_loss(losses,test_losses, save_path=f'{basedir}/checkpoint/{select_model}/loss.png')

    torch.save(model, f'{basedir}/checkpoint/{select_model}/{select_model}_cifar_{metric:0.6f}.pth')
    torch.save(model.state_dict(), f'{basedir}/checkpoint/{select_model}/{select_model}_cifar_{metric:0.6f}_state_dict.pth')


if __name__ == '__main__':
    freeze_support()
    main()

    
