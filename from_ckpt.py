import torch
from MQDD_model import ClsHeadModelMQDD

# https://github.com/kiv-air/MQDD

model = ClsHeadModelMQDD("UWB-AIR/MQDD-duplicates")
ckpt = torch.load("model.pt",  map_location="cpu")
model.load_state_dict(ckpt["model_state"])
