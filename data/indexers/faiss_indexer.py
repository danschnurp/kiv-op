#  date: 23. 3. 2023
#  author: Daniel Schnurpfeil
#
import time

import numpy as np
import torch
from faiss import write_index, IndexFlatL2
from lxml.etree import XMLParser, parse

from data.utils import sanitize_html_for_web
from web.search.brain.neural_loader import load_nett


def index_posts(tokenizer,
                position_ids,
                input_folder="../../../../SERVER_APPS/gamedev.stackexchange.com",
                desired_filename="/Posts.xml"):
    """
    Indexing xml file to vectors

    :param tokenizer: the tokenizer to use to tokenize the posts
    :param position_ids: a list of the position ids that you want to index
    :param input_folder: the folder where the Posts.xml file is located, defaults to
    ../../../SERVER_APPS/gamedev.stackexchange.com (optional)
    :param desired_filename: the name of the file you want to index, defaults to /Posts.xml (optional)
    """
    # Getting the length of the position ids and dividing it by 2.
    max_length = int(len(torch.squeeze(position_ids)) / 2.)
    # Parsing the XML file.
    root = parse(input_folder + desired_filename, parser=XMLParser(huge_tree=True))
    # Getting the title of each post.
    titles = root.xpath("//row/@Title")
    print(desired_filename, "loaded...")
    # Converting the text into a sequence of numbers.
    titles = [tokenizer.encode(
        # Removing the HTML tags from the text.
        sanitize_html_for_web(i),
        max_length=max_length,
        truncation=True, return_tensors="pt") for i in titles]

    #  todo "max_length" could be optimized to max value of "titles"
    indexer = IndexFlatL2(max_length)  # build the index

    # Creating a matrix of zeros with the length of the titles and the max length of the position ids.
    titles_matrix = np.zeros((len(titles), max_length))

    for index, i in enumerate(titles):
        # Converting the tensor into a numpy array and then adding it to the matrix.
        j = np.squeeze(i.detach().numpy())
        titles_matrix[index, :len(j)] = j

    # Adding the matrix to the indexer.
    indexer.add(titles_matrix)

    # Saving the indexed data to a file.
    write_index(indexer, "./titles.index")
    # testing
    query = np.pad(np.squeeze(titles[0].detach().numpy()), (0, max_length - len(np.squeeze(titles[0].detach().numpy()))), 'constant')
    query = np.expand_dims(query, axis=0)
    D, I = indexer.search(query, 2)
    print(D)
    print(I)


_, tok, pos_ids = load_nett(model_location="")
t1 = time.time()
index_posts(tok, pos_ids)
print(time.time() - t1, "sec")
