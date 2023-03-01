import torch
from pandas import read_parquet
from transformers import AutoTokenizer

from MQDD_model import ClsHeadModelMQDD

# https://drive.google.com/drive/folders/1JG6Fibvhs0Jz6JD83gwMqAmzUV9rsoX3
# Reading the parquet file and storing it in a dataframe.
data = read_parquet("SODD_dev.parquet.gzip")
print(data.label[0:20])
index = 9

first_post = data.first_post[index]
second_post = data.second_post[index]

tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")

# Encoding the first post into a sequence of integers.
encoding_first = tokenizer.encode(first_post, return_tensors="pt")

# Encoding the second post into a sequence of integers.
encoding_second = tokenizer.encode(second_post, return_tensors="pt")

model = ClsHeadModelMQDD("UWB-AIR/MQDD-duplicates")
ckpt = torch.load("model.pt",  map_location="cpu")
del ckpt["model_state"]["bert_model.embeddings.position_ids"]
model.load_state_dict(ckpt["model_state"])

# todo parse params

# model.forward()
