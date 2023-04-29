from unittest import TestCase

import numpy as np
from faiss import read_index

from data.indexers.question_encoder import encode_question, prepare_tok_model


def test_faiss_indexed_data(index_data_path, question, true_id):
    """
    This function takes in an indexed data path, a question, an ID and performs a test if data are indexed correctly.

    :param index_data_path: The path to the file containing the indexed data
    :param question: The question for which we want to find the most similar data point in the indexed data
    :param true_id: The true ID of the document that should be the top result for the given question
    """
    index = read_index(index_data_path)

    normalized_question = np.zeros((1, index.d))
    tokenizer, model, _ = prepare_tok_model()
    encoded_question = encode_question(question=question, tokenizer=tokenizer,
                                       model=model)
    normalized_question[0, :len(encoded_question)] = encoded_question
    _, I, _ = index.search_and_reconstruct(normalized_question, 5)  # actual search
    print("ID:", true_id)
    print("found IDS:", I[:5])
    assert true_id in I[:5]


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.path = "D:/_KIV_OP/gamedev/faiss_indexed_data/Posts/"

    def test_body(self):
        test_faiss_indexed_data(
            index_data_path=self.path + "Body.index",
            question="like to read up on path finding algorithms. Is there a primer available "
                     "or any material or tutorials on the Internet that would be a good start "
                     "for me",
            true_id=1
        )

    def test_title(self):
        test_faiss_indexed_data(
            index_data_path=self.path + "Title.index",
            question="path finding algorithms",
            true_id=1
        )

