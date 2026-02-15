import pickle

from PruningNAS.PruneEvaluator import load_and_print_accuracies

# Path to the pickle file
pickle_file = r"D:\Sutanu_WorkSpace\PruningNAS\PruningNAS\checkpoint\Resnet-50\FGP\Resnet-50_accuracies.pkl"

load_and_print_accuracies(pickle_file)
