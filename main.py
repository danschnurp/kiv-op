from pandas import read_parquet
from transformers import AutoTokenizer, AutoModel

data = read_parquet("SODD_dev.parquet.gzip")
print(data.count())
print(data.first_post[0])
exit(0)


tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")
model = AutoModel.from_pretrained("UWB-AIR/MQDD-duplicates")



question, text = "W", "W"

encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
input_ids = encoding["input_ids"]


start_scores, end_scores = model(input_ids)
print(start_scores.shape)

print(end_scores.shape)



