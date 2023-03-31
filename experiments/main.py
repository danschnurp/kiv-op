import numpy as np
from faiss import read_index
from transformers import AutoTokenizer
from data.utils import sanitize_html_for_web

index = read_index("/Users/danschnurpfeil/SERVER_APPS/gamedev.stackexchange.com/posts/Body.index")

print(index.ntotal)
print(index.d)

max_length = 512
tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")

encoded_question = tokenizer.encode(
    # Removing the HTML tags from the text.
    sanitize_html_for_web("Is there a common practice for resolving polygon/polygon collision by sliding as shown in the image"),
    max_length=max_length,
    truncation=True, return_tensors="pt")

normalized_question = np.zeros((1, index.d))
encoded_question = np.squeeze(encoded_question.detach().numpy())
normalized_question[0, :len(encoded_question)] = encoded_question

D, I = index.search(normalized_question, 3)  # actual search
print(I[:5])  # neighbors of the 5 first queries

