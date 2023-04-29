from django.apps import AppConfig

from web.search.brain.neural_loader import load_nett
import torch
from faiss import read_index


class SearchConfig(AppConfig):
    name = 'search'
    model, tokenizer, position_ids = load_nett(model_location="./search/brain/model.pt")
    max_len = int(len(torch.squeeze(position_ids)) / 2.)
    indexed_post_bodies = read_index("./search/indexed_data/Body.index")
