import torch
from pandas import read_parquet

# https://drive.google.com/drive/folders/1JG6Fibvhs0Jz6JD83gwMqAmzUV9rsoX3
# Reading the parquet file and storing it in a dataframe.
data = read_parquet("SODD_dev.parquet.gzip")

data = data[data.label == 0]
# print(data.iloc)
# print()
# exit(0)

model, tokenizer, position_ids = load_nett()
max_len = int(len(torch.squeeze(position_ids)) / 2.)
print("max_len", max_len)

for index, i in enumerate(data.iloc):
    first_post = i.first_post
    second_post = i.second_post

    # Encoding the first post into a sequence of integers.
    encoding_first = tokenizer.encode(first_post, max_length=max_len, truncation=True, return_tensors="pt")

    # Encoding the second post into a sequence of integers.
    encoding_second = tokenizer.encode(second_post, max_length=max_len, truncation=True, return_tensors="pt")
    out = model.forward(encoding_first, encoding_second)

    print(index, torch.argmax(out))
