import pickle

from PruningNAS.PruneEvaluator import load_and_print_accuracies

# Path to the pickle file
pickle_file = r"PruningNAS\checkpoint\Resnet-50\CP\Resnet-50_accuracies.pkl"

load_and_print_accuracies(pickle_file)