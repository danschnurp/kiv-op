import numpy as np
from faiss import read_index

from experiments.question_encoder import encode_question, prepare_tok_model

index = read_index("/Users/danschnurpfeil/SERVER_APPS/gamedev.stackexchange.com/faiss_indexed_data/Comments/Text.index")

print(index.ntotal)
print(index.d)

normalized_question = np.zeros((1, index.d))
tokenizer, model = prepare_tok_model()
# id = 8
encoded_question = encode_question(question="I find that the 3D aspect of unity though tends to get in the way "
                                            "insofar as user interface is concerned. Also I always end up making a "
                                            "dummy scene that has nothing but a camera with my root script "
                                            "attached.", tokenizer=tokenizer,
                                   model=model)
normalized_question[0, :len(encoded_question)] = encoded_question
# print("encoded_question", encoded_question)
# print("normalized_question", normalized_question)
D, I, R = index.search_and_reconstruct(normalized_question, 5)  # actual search
# print(index.id_map)
print(I[:5])  # neighbors of the 5 first queries
print(D[:5])


