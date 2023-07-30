from unittest import TestCase

import numpy as np
import torch
from faiss import read_index
from pandas import read_parquet
from tqdm import tqdm
from faiss_indexer import index_with_faiss_to_file
from question_encoder import encode_question, prepare_tok_model
from utils import sanitize_html_for_index
from MQDD_model import load_nett, encode_classic


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


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()

        # Reading the parquet file and storing it in a dataframe.
        data = read_parquet("SODD_dev.parquet.gzip")
        print("(len data)", len(data))
        data = data[:1000]
        data = data[(data.label == 0) | (data.label == 3)]
        self.dup = ("duplicates:", len(data.label[data.label == 0]))
        self.diff = ("diffs:", len(data.label[data.label == 3]))
        print("(len data) filtered", len(data))

        self.path = "./test.index"
        self.first_posts = list(data.first_post)
        self.indexes = list(data.axes[0])
        self.second_posts = data.second_post
        self.ids = data.label

    def test_body(self):
        """
        tests original faiss indexed data and sanitizing them first
        """
        tokenizer, model, _, this_device = prepare_tok_model()
        index_with_faiss_to_file(self.first_posts, self.indexes,
                                 "./test.index", 1)
        indexes = [read_index(self.path)]
        tp, tn, fp, fn = 0, 0, 0, 0

        for id, index, post in (zip(self.ids, self.indexes, self.second_posts)):
            post = sanitize_html_for_index(post)
            tp1, tn1, fp1, fn1 = Test.compute_confusion_matrix(
                indexes=indexes,
                question=post,
                true_id=index,
                similarity=id, tokenizer=tokenizer, model=model, this_device="cpu"
            )
            tp += tp1
            tn += tn1
            fp += fp1
            fn += fn1

        assert self.print_stats(tp, tn, fp, fn) > 0.7

    def test_body_no_sanitizing(self):
        """
        tests original faiss indexed data
        """
        tokenizer, model, _, this_device = prepare_tok_model()
        index_with_faiss_to_file(self.first_posts, self.indexes,
                                 "./test.index", 1, sanitize=False)
        indexes = [read_index(self.path)]
        tp, tn, fp, fn = 0, 0, 0, 0

        for id, index, post in (zip(self.ids, self.indexes, self.second_posts)):
            tp1, tn1, fp1, fn1 = Test.compute_confusion_matrix(
                indexes=indexes,
                question=post,
                true_id=index,
                similarity=id, tokenizer=tokenizer, model=model, this_device="cpu"
            )
            tp += tp1
            tn += tn1
            fp += fp1
            fn += fn1

        assert self.print_stats(tp, tn, fp, fn) > 0.7

    @staticmethod
    def compute_confusion_matrix(indexes, question, true_id, similarity, tokenizer, model, this_device):
        """
        This function takes in an indexed data path, a question, an ID and performs a test if data are indexed correctly.
        """
        encoded_question = encode_question(question=question, tokenizer=tokenizer,
                                           model=model, this_device=this_device)
        normalized_question = np.expand_dims(encoded_question, axis=0)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for index in indexes:
            _, I = index.search(normalized_question, 5)  # actual search
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
                fn += 1
            # different negative
            elif similarity == 3 and true_id in I:
                fp += 1

        return tp, tn, fp, fn

    def print_stats(self, tp, tn, fp, fn):
        accuracy = (tp + tn) / (self.dup[1] + self.diff[1])
        print(*self.dup)
        print(*self.diff)
        print("tp", tp, "fp", fp)
        print("fn", fn, "tn", tn)
        print("accuracy", accuracy)
        return accuracy

    def test_MQDD(self):
        """
        tests original two tower model in MQDD_model.py
        """
        model, tokenizer, position_ids = load_nett(model_location="./model.pt")
        max_len = int(len(torch.squeeze(position_ids)) / 2.)
        print("max_len", max_len)
        tp, tn, fp, fn = 0, 0, 0, 0
        pred = []
        for first, second, label in zip(self.first_posts, self.second_posts, tqdm(self.ids)):
            first = encode_classic(first, tokenizer)
            second = encode_classic(second, tokenizer)
            res_id = model.forward({**first.data, **second.data})
            if res_id == "DUPLICATE" and label == 0:
                tp += 1
            # different
            elif res_id == "different" and label == 3:
                tn += 1

            elif res_id == "DUPLICATE" and label != 0:
                fp += 1
            # different negative
            elif res_id == "different" and label != 3:
                fn += 1
            pred.append(res_id)

        assert self.print_stats(tp, tn, fp, fn) > 0.7
