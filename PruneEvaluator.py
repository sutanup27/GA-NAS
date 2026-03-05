from multiprocessing import freeze_support
import pickle
import torch
from PruningNAS.DataProcess.DataPreprocessing import get_dataloaders
from PruningNAS.Utills.TrainingModulesUtills import evaluate
from PruningNAS.Utills.Utill import plot_sensitivity_scan, sensitivity_scan
from PruningNAS.Utills.ViewerUtills import accumulate_plot_figures, plot_weight_distribution  # Ensure you import your correct model architecture

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Initialize the model
    basedir='PruningNAS'
    path='./dataset/cifar10'
    select_model='MobilenetV1'
    prune_type='FGP'
    #model_path=f'{basedir}/checkpoint/vgg_mrl_99.51375579833984.pth'
    model_path=r'PruningNAS\checkpoint\MobilenetV1\MobilenetV1_cifar_94.129997.pth'
    # Load the saved state_dict correctly
    model = torch.load(model_path, map_location=torch.device(device),weights_only=False)  # Use 'cpu' if necessary
    model.to(device)


    sparsities_path=f'{basedir}/checkpoint/{select_model}/{prune_type}/{select_model}_sparsities.pkl'
    accuracies_path=f'{basedir}/checkpoint/{select_model}/{prune_type}/{select_model}_accuracies.pkl'

    train_dataloader,test_dataloader=get_dataloaders(path )
    dense_model_accuracy,_=evaluate(model,test_dataloader)
    print(f"Original model accuracy: {dense_model_accuracy:.4f}")
    ############# calculate sparsities (optional) #############################################

    # sparsities, accuracies,names = sensitivity_scan(
    # model, test_dataloader, scan_step=0.1, scan_start=0.1, scan_end=1.0,prune_type=prune_type,select_model=select_model)

    # with open(sparsities_path, "wb") as f:
    #     pickle.dump(sparsities, f)

    # with open(accuracies_path, "wb") as f:
    #     pickle.dump((accuracies,names), f)

    ############################################################################################
    with open(sparsities_path, "rb") as f:
        sparsities = pickle.load(f)

    with open(accuracies_path, "rb") as f:
        accuracies,names = pickle.load(f)
        

    # save_image_path1=f'{basedir}/checkpoint/{select_model}/{prune_type}/param_plot/{select_model}_paramplot_{prune_type}'
    save_image_path2=f'{basedir}/checkpoint/{select_model}/{prune_type}/sensitivity_curves/{select_model}_sensitivity_{prune_type}'
    # plot_weight_distribution(model,names,save_path=save_image_path1)
    plot_sensitivity_scan( names, sparsities, accuracies, dense_model_accuracy,save_image_path2)
    accumulate_plot_figures(f'{basedir}/checkpoint/{select_model}/{prune_type}/sensitivity_curves')


def load_and_print_accuracies(pickle_file):
    """Load pickle file and print accuracies with parameter names."""
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Print accuracies with parameter names
    for i, name in enumerate(data[1]):
        print(f"{name}: {data[0][i]}")

    print(f'length of data: {len(data[0])}, length of names: {len(data[1])}')


if __name__ == '__main__':
    freeze_support()
    main()
