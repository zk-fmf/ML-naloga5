from model import ModelCT
from matplotlib import pyplot as plt
import numpy as np
import torch

model_folder = 'maxpool_newtraining_2020-12-07_16-41-41' # mapa naucenega modela
Model = ModelCT() # model
# Nalozimo model z utezmi
Model.load_state_dict(torch.load("trained_models/"+model_folder+"/BEST_model.pth"))

# Nalozimo npr. filtre iz prve konvolucijske plasti backbone.conv1
weights = Model.backbone.conv1.weight.data.numpy()
# weights.shape = (64,1,7,7) -> 64 filtrov, z 1 kanalom, velikosti 7x7

# Vizualizacija 21. filtra iz prve plasti (backbone.conv1)
plt.imshow(weights[20,0,:,:], cmap='gray')