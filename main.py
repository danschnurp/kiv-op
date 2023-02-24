from pandas import read_parquet
from transformers import AutoTokenizer, AutoModel

from MQDD_model import ClsHeadModelMQDD

# https://drive.google.com/drive/folders/1JG6Fibvhs0Jz6JD83gwMqAmzUV9rsoX3
# Reading the parquet file and storing it in a dataframe.
data = read_parquet("SODD_dev.parquet.gzip")
# Printing the number of rows in the dataframe, the first post, the second post, and the label.
# print(data.count())
print(data.label[0:20])
index = 9
#  output                             # index - label.
# [[0.9077884  0.99088585 0.9473789 ]] # 11 - 0.
# [[0.94933444 0.9992542  0.95759284]] # 16 - 0.
# [[0.9855168 0.9945899 0.9411869]]    # 10 - 3.

first_post = data.first_post[index]
second_post = data.second_post[index]

# Loading the model and tokenizer from the HuggingFace model hub.
tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")
encoder_instance = AutoModel.from_pretrained("UWB-AIR/MQDD-duplicates")
classification_layers = ClsHeadModelMQDD("UWB-AIR/MQDD-duplicates")

# Encoding the first post into a sequence of integers.
encoding_first = tokenizer.encode(first_post, return_tensors="pt")

# Encoding the second post into a sequence of integers.
encoding_second = tokenizer.encode(second_post, return_tensors="pt")

first_output_from_encoder = encoder_instance(encoding_first)
second_output_from_encoder = encoder_instance(encoding_second)

output = classification_layers.forward(first_output_from_encoder[1], second_output_from_encoder[1])

print(output)
