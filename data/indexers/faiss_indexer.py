#  date: 23. 3. 2023
#  author: Daniel Schnurpfeil
#
import time

import numpy as np
from faiss import write_index, IndexFlatL2, IndexIDMap2
from lxml.etree import XMLParser, parse

from data.utils import make_output_dir, sanitize_html_for_web
from experiments.question_encoder import encode_questions, prepare_tok_model


def index_part(input_folder: str, xml_file_name: str, part: str):
    # Loading the data from the xml file.
    input_data = load_xml_data(input_folder=input_folder,
                               desired_filename="/" + xml_file_name, xpath_query="//row/@" + part)
    ids = load_xml_data(input_folder=input_folder,
                        desired_filename="/" + xml_file_name, xpath_query="//row/@" + "Id")
    print(xml_file_name, part, "loaded...")
    t1 = time.time()
    input_folder = make_output_dir("faiss_indexed_data", input_folder)
    # Indexing the part.
    index_with_faiss_to_file(input_data=input_data, ids=ids,
                             output_file_path=make_output_dir(output_dir=input_folder,
                                                              output_filename=xml_file_name[:-4]) + "/" + part + ".index")
    print(part, "part processed in:", time.time() - t1, "sec")


def load_xml_data(input_folder: str, desired_filename: str, xpath_query: str) -> list:
    """
    Loads XML data from a file in a folder, and returns a list of the data.

    :param input_folder: The folder where the XML file is located
    :type input_folder: str
    :param desired_filename: The name of the file you want to load
    :type desired_filename: str
    :param xpath_query: This is the xpath query that will be used to extract the data from the XML file
    :type xpath_query: str
    """
    # Parsing the XML file.
    root = parse(input_folder + desired_filename, parser=XMLParser(huge_tree=True))
    # Getting the title of each post.
    data = root.xpath(xpath_query)
    return data


def index_with_faiss_to_file(input_data: list, ids: list, output_file_path: str,
                             ):
    """
    Indexes list of text with "UWB-AIR/MQDD-duplicates" tokenizer and saves it to file
    :param ids:
    :param output_file_path: output file path
    :param input_data: list of strings to be indexed
    :type input_data: list
    """

    part = 50
    input_data = input_data[:part]
    ids = ids[:part]

    tokenizer, model, tokenized_question_example = prepare_tok_model()

    t1 = time.time()
    data = [sanitize_html_for_web(i.replace("\n", ""),  display_code=False) for i in input_data]
    data = encode_questions(data, tokenizer, model, tokenized_question_example, batch_size=1)
    print("sanitized and encoded in:", time.time() - t1, "sec")

    # Finding the maximum length of the data for saving memory on disk 🤔
    max_indexed_length = 0
    for i in data:
        curr_len = len(i)
        if max_indexed_length < curr_len:
            max_indexed_length = curr_len
    # Creating a matrix of zeros with the size of the number of data and the maximum length of the data.
    data_matrix = np.zeros((len(data), max_indexed_length))
    for index, i in enumerate(data):
        # Converting the tensor into a numpy array and then adding it to the matrix.
        j = np.squeeze(i)
        data_matrix[index, :len(j)] = j

    indexer = IndexFlatL2(max_indexed_length)  # build the index

    indexer = IndexIDMap2(indexer)
    # Adding the matrix to the indexer.
    indexer.add_with_ids(data_matrix, ids)
    # Saving the indexed data to a file.
    write_index(indexer, output_file_path)
