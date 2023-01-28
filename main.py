import numpy as np
from pandas import read_parquet
from transformers import AutoTokenizer, AutoModel

from data.utils import sanitize_html_for_web

# https://drive.google.com/drive/folders/1JG6Fibvhs0Jz6JD83gwMqAmzUV9rsoX3
# Reading the parquet file and storing it in a dataframe.
data = read_parquet("SODD_dev.parquet.gzip")
# Printing the number of rows in the dataframe, the first post, the second post, and the label.
print(data.count())
first_post = sanitize_html_for_web(data.first_post[0])
second_post = sanitize_html_for_web(data.second_post[0])

print(second_post)

print(data.label[0])
exit(0)


# Loading the model and tokenizer from the HuggingFace model hub.
tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")
model = AutoModel.from_pretrained("UWB-AIR/MQDD-duplicates")


# Encoding the first post into a sequence of integers.
encoding_first = tokenizer.encode(first_post, return_tensors="pt")


# Encoding the second post into a sequence of integers.
encoding_second = tokenizer.encode(second_post, return_tensors="pt")


output1, output2 = model(encoding_first, encoding_second)

print(np.argmax(output1.detach().numpy()))
