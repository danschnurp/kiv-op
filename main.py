import torch
from pandas import read_parquet
from transformers import AutoTokenizer

from MQDD_model import ClsHeadModelMQDD


def load_nett():
    tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")
    model = ClsHeadModelMQDD("UWB-AIR/MQDD-duplicates")
    ckpt = torch.load("model.pt", map_location="cpu")
    del ckpt["model_state"]["bert_model.embeddings.position_ids"]
    model.load_state_dict(ckpt["model_state"])
    return model, tokenizer


# https://drive.google.com/drive/folders/1JG6Fibvhs0Jz6JD83gwMqAmzUV9rsoX3
# Reading the parquet file and storing it in a dataframe.
data = read_parquet("SODD_dev.parquet.gzip")

data = data.head(50)
data = data[data.label == 0]
print(data.index)
print(data.get())
exit(0)

model, tokenizer = load_nett()

for i in data.index:
    first_post = data.first_post[i]
    second_post = data.second_post[i]

    # Encoding the first post into a sequence of integers.
    encoding_first = tokenizer.encode(first_post, return_tensors="pt")

    # Encoding the second post into a sequence of integers.
    encoding_second = tokenizer.encode(second_post, return_tensors="pt")
    print("index:", i)
    out = model.forward(encoding_first, encoding_second)

    print(out)
