import numpy as np
from faiss import read_index

from experiments.question_encoder import encode_question, prepare_tok_model

index = read_index("/Users/danschnurpfeil/SERVER_APPS/gamedev.stackexchange.com/faiss_indexed_data/Posts/Body.index")

print(index.ntotal)
print(index.d)

normalized_question = np.zeros((1, index.d))
tokenizer, model = prepare_tok_model()
# id = 1
encoded_question = encode_question(question="like to read up on path finding algorithms. Is there a primer available "
                                            "or any material or tutorials on the Internet that would be a good start "
                                            "for me", tokenizer=tokenizer,
                                   model=model)
normalized_question[0, :len(encoded_question)] = encoded_question
# print("encoded_question", encoded_question)
# print("normalized_question", normalized_question)
D, I, R = index.search_and_reconstruct(normalized_question, 5)  # actual search
# print(index.id_map)
print(I[:5])  # neighbors of the 5 first queries
print(D[:5])


