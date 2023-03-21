import torch
from transformers import AutoTokenizer

from web.search.brain.MQDD_model import ClsHeadModelMQDD


def load_nett():
    tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")
    model = ClsHeadModelMQDD("UWB-AIR/MQDD-duplicates")

    # https://drive.google.com/drive/folders/1CYiqF2GJ2fSQzx_oM4-X_IhpObi4af5Q
    ckpt = torch.load("./search/brain/model.pt", map_location="cpu")
    position_ids = ckpt["model_state"]["bert_model.embeddings.position_ids"]
    del ckpt["model_state"]["bert_model.embeddings.position_ids"]
    model.load_state_dict(ckpt["model_state"])
    return model, tokenizer, position_ids