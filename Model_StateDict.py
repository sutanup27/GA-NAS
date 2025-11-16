
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

model_path='checkpoint/Resnet-18/Resnet-18_cifar_95.48999786376953.pth'
# Load the saved state_dict correctly
model = torch.load(model_path, map_location=torch.device(device))  # Use 'cpu' if necessary
model.to(device)

def Model_to_sd(path):

    model = torch.load(path, map_location=torch.device(device))  # Use 'cpu' if necessary
    model.to(device)

    # torch.save(model.state_dict(), path)


