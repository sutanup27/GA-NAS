import numpy as np
import random
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *


#fix the randomness
seed=0
random.seed(seed)
np.random.seed(seed)



