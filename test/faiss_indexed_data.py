import os
from unittest import TestCase

import numpy as np
from faiss import read_index, extract_index_ivf

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


def test_faiss_indexed_data(index_data_path, question, true_id):
    """
    This function takes in an indexed data path, a question, an ID and performs a test if data are indexed correctly.

    :param index_data_path: The path to the file containing the indexed data
    :param question: The question for which we want to find the most similar data point in the indexed data
    :param true_id: The true ID of the document that should be the top result for the given question
    """

    indexes = [read_index(index_data_path + "/" + i) for i in os.listdir(index_data_path)]

    normalized_question = np.zeros((1, indexes[0].d))
    tokenizer, model, _, this_device = prepare_tok_model()
    question = sanitize_html_for_web(question.replace("\n", ""))
    encoded_question = encode_question(question=question, tokenizer=tokenizer,
                                       model=model, this_device=this_device)
    normalized_question[0, :len(encoded_question)] = encoded_question.astype(np.float32)
    for index in indexes:
        D, I = index.search(normalized_question, 5) # actual search
        print("ID:", true_id)
        I = np.squeeze(I[:5])
        print("found IDS:", I[:5])


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.path = "C:/Users/dartixus/Downloads/apple.stackexchange.com/faiss_indexed_data/Posts"

    def test_body(self):
        test_faiss_indexed_data(
            index_data_path=self.path,
            question="""&lt;p&gt;The VPN software I use for work (&lt;a href=&quot;http://www.lobotomo.com/products/IPSecuritas/&quot;&gt;IPSecuritas&lt;/a&gt;) requires me to turn off Back To My Mac to start it's connection, so I frequently turn off Back To My Mac in order to use my VPN connection (the program does this for me). I forget to turn it back on however and I'd love to know if there was something I could run (script, command) to turn it back on.&lt;/p&gt;&#xA;""",
            true_id=2
        )

    def test_body2(self):
        test_faiss_indexed_data(
            index_data_path=self.path,
            question="""&lt;p&gt;As of Snow Leopard, this from any application. From the Actions Library, add the 'Launch Application' action to the workflow. Select the 'Terminal' application in the drop-down list of Applications. Save your new service and then assign a keyboard shortcut to it in:&lt;br&gt;&#xA;&lt;code&gt;System Preferences -&amp;gt; Keyboard -&amp;gt; Keyboard Shortcuts -&amp;gt; Services&lt;/code&gt;&lt;/p&gt;&#xA;"""
            ,true_id=170
        )
