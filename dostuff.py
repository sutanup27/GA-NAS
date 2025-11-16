
from ResNet import *
from torch import nn

from Utill import get_sparsity_dic_template, print_model
from Viewer import accumulate_plot_figures

model=ResNet34(classes=10)

get_sparsity_dic_template(model,'CP')
#accumulate_plot_figures('PruningNAS/checkpoint/Resnet-34/CP/sensitivity_curves')
