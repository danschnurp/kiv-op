from django.apps import AppConfig

from web.search.brain.neural_loader import load_nett
import torch


class SearchConfig(AppConfig):
    name = 'search'
    model, tokenizer, position_ids = load_nett(model_location="./search/brain/model.pt")
    max_len = int(len(torch.squeeze(position_ids)) / 2.)
