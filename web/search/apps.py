from django.apps import AppConfig

from web.search.brain.neural_loader import load_nett
import torch


class SearchConfig(AppConfig):
    name = 'search'
    model, tokenizer, position_ids = load_nett()
    max_len = int(len(torch.squeeze(position_ids)) / 2.)
