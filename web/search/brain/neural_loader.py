import torch
from transformers import AutoTokenizer

from web.search.brain.MQDD_model import ClsHeadModelMQDD


def load_nett(model_location: str) -> tuple:
    """
    > Loads a model from a given location and returns the model and the model's input shape

    :param model_location: The location of the model file
    :type model_location: str
    """
    # Loading the pretrained model from HuggingFace.
    tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")

    # tokenizer eos_token -> sep , bos_token -> cls
    tokenizer.bos_token = "cls"
    tokenizer.eos_token = "sep"

    model = ClsHeadModelMQDD("UWB-AIR/MQDD-duplicates")

    # https://drive.google.com/drive/folders/1CYiqF2GJ2fSQzx_oM4-X_IhpObi4af5Q
    # Loading the model from the location specified in `model_location` and mapping it to the CPU.
    ckpt = torch.load(model_location, map_location="cpu")
    # A workaround for a bug in the HuggingFace library.
    position_ids = ckpt["model_state"]["bert_model.embeddings.position_ids"]
    del ckpt["model_state"]["bert_model.embeddings.position_ids"]
    model.load_state_dict(ckpt["model_state"])
    # Returning the model, the tokenizer, and the position ids.
    return model, tokenizer, position_ids
