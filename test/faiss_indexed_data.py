from unittest import TestCase

import numpy as np
from faiss import read_index

from data.question_encoder import encode_question, prepare_tok_model
from data.utils import sanitize_html_for_web


def test_faiss_indexed_data(index_data_path, question, true_id):
    """
    This function takes in an indexed data path, a question, an ID and performs a test if data are indexed correctly.

    :param index_data_path: The path to the file containing the indexed data
    :param question: The question for which we want to find the most similar data point in the indexed data
    :param true_id: The true ID of the document that should be the top result for the given question
    """
    index = read_index(index_data_path)

    normalized_question = np.zeros((1, index.d))
    tokenizer, model, _, this_device = prepare_tok_model()
    question = sanitize_html_for_web(question.replace("\n", ""))
    encoded_question = encode_question(question=question, tokenizer=tokenizer,
                                       model=model, this_device=this_device)
    normalized_question[0, :len(encoded_question)] = encoded_question.astype(np.float32)
    D, I = index.search(normalized_question, 5) # actual search
    print("ID:", true_id)
    I = np.squeeze(I[:5])
    print("found IDS:", I[:5])
    assert true_id in I[:5]


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.path = "D:/_KIV_OP/stackoverflow.com-Posts/faiss_indexed_data/Posts/Body_0_100.index"

    def test_body(self):
        test_faiss_indexed_data(
            index_data_path=self.path,
            question="""<p>I want to assign the decimal variable &quot;trans&quot; to the double variable &quot;this.Opacity&quot;.</p>
<pre class="lang-cs prettyprint-override"><code>decimal trans = trackBar1.Value / 5000;
this.Opacity = trans;
</code></pre>
<p>When I build the app it gives the following error:</p>
<blockquote>
<p>Cannot implicitly convert type decimal to double</p>
</blockquote>""",
            true_id=4
        )

