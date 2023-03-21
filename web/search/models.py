import torch
from django.db import models

from web.search.brain.neural_loader import load_nett

model, tokenizer, position_ids = load_nett()
max_len = int(len(torch.squeeze(position_ids)) / 2.)