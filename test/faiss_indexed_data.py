from unittest import TestCase

import numpy as np
from faiss import read_index
from pandas import read_parquet
from tqdm import tqdm

from data.faiss_indexer import index_with_faiss_to_file
from data.question_encoder import encode_question, prepare_tok_model
from data.utils import sanitize_html_for_web


def merge_invlists(il_src, il_dest):
    """
    merge inverted lists from two ArrayInvertedLists
    may be added to main Faiss at some point
    """
    assert il_src.nlist == il_dest.nlist
    assert il_src.code_size == il_dest.code_size

    for list_no in range(il_src.nlist):
        il_dest.add_entries(
            list_no,
            il_src.list_size(list_no),
            il_src.get_ids(list_no),
            il_src.get_codes(list_no)
        )


def test_faiss_indexed_data(indexes, question, true_id, similarity, tokenizer, model, this_device):
    """
    This function takes in an indexed data path, a question, an ID and performs a test if data are indexed correctly.

    :param index_data_path: The path to the file containing the indexed data
    :param question: The question for which we want to find the most similar data point in the indexed data
    :param true_id: The true ID of the document that should be the top result for the given question
    """
    question = sanitize_html_for_web(question, display_code=False)
    encoded_question = encode_question(question=question, tokenizer=tokenizer,
                                       model=model, this_device=this_device)
    normalized_question = np.expand_dims(encoded_question, axis=0)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for index in indexes:
        D, I = index.search(normalized_question, 5)  # actual search
        # print("ID:", true_id)
        I = np.squeeze(I)
        # print("found IDS:", I[:5])
        # similar
        if similarity == 0 and true_id in I:
            tp += 1
        # different
        elif similarity == 3 and true_id not in I:
            tn += 1
        # similar negative
        elif similarity == 0 and true_id not in I:
            fp += 1
        # different negative
        elif similarity == 3 and true_id in I:
            fn += 1

    return tp, tn, fp, fn


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()

        # Reading the parquet file and storing it in a dataframe.
        data = read_parquet("SODD_dev.parquet.gzip")
        print("(len data)", len(data))
        data = data[:1000]
        data = data[(data.label == 0) | (data.label == 3)]
        print("duplicates:", len(data.label[data.label == 0]))
        print("diffs:", len(data.label[data.label == 3]))
        print("(len data) filtered", len(data))
        # exit(0)

        self.path = "./test.index"
        self.first_posts = list(data.first_post)
        index_with_faiss_to_file(self.first_posts, list(np.arange(0, len(self.first_posts))),
                                 "C:/Users/dartixus/PycharmProjects/kiv-op/test/test.index", 1, 0,
                                 len(self.first_posts))

        self.second_posts = data.second_post
        self.ids = data.label
        self.indexes = list(np.arange(0, len(self.second_posts)))

    def test_body(self):
        tokenizer, model, _, this_device = prepare_tok_model()
        indexes = [read_index(self.path)]
        tp, tn, fp, fn = 0, 0, 0, 0
        for id, index, post in (zip(self.ids, tqdm(self.indexes), self.second_posts)):
            tp1, tn1, fp1, fn1 =test_faiss_indexed_data(
                indexes=indexes,
                question=post,
                true_id=index,
                similarity=id, tokenizer=tokenizer, model=model, this_device=this_device
            )
            tp += tp1
            tn += tn1
            fp += fp1
            fn += fn1

        print("tp", tp, "fp", fp)
        print("fn", fn, "tn", tn)
