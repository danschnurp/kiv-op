import numpy as np
import torch
from pandas import read_parquet
from transformers import AutoTokenizer, AutoModel

# https://drive.google.com/drive/folders/1JG6Fibvhs0Jz6JD83gwMqAmzUV9rsoX3
# Reading the parquet file and storing it in a dataframe.
data = read_parquet("SODD_dev.parquet.gzip")
# Printing the number of rows in the dataframe, the first post, the second post, and the label.
print(data.count())
first_post = data.first_post[0]
second_post = data.second_post[0]
# first_post = data.first_post[4]
# second_post = data.second_post[4]

# Loading the model and tokenizer from the HuggingFace model hub.
tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")
model = AutoModel.from_pretrained("UWB-AIR/MQDD-duplicates")

# Encoding the first post into a sequence of integers.
encoding_first = tokenizer.encode(first_post, return_tensors="pt")


# Encoding the second post into a sequence of integers.
encoding_second = tokenizer.encode(second_post, return_tensors="pt")

if encoding_second.shape < encoding_first.shape:
    encoding_first = torch.from_numpy(np.resize(encoding_first, encoding_second.shape))
elif encoding_second.shape > encoding_first.shape:
    encoding_second = torch.from_numpy(np.resize(encoding_second, encoding_first.shape))


output = model(encoding_first, encoding_second)

print()
