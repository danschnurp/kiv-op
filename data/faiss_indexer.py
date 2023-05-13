#  date: 23. 3. 2023
#  author: Daniel Schnurpfeil
#
import time

import numpy as np
from faiss import write_index, IndexFlatL2, IndexIDMap2, read_index
from lxml.etree import XMLParser, parse

from utils import make_output_dir, sanitize_html_for_web
from question_encoder import encode_questions, prepare_tok_model


def index_part(input_folder: str, xml_file_name: str, part: str, batch_size=4, stop_at=None, output_dir_path=None):
    # Loading the data from the xml file.
    input_data = load_xml_data(input_folder=input_folder,
                               desired_filename="/" + xml_file_name, xpath_query="//row/@" + part)
    ids = load_xml_data(input_folder=input_folder,
                        desired_filename="/" + xml_file_name, xpath_query="//row/@" + "Id")
    print(len(ids), xml_file_name, part, "loaded...")
    t1 = time.time()
    if not output_dir_path:
        output_dir_path = input_folder
    output_dir_path = make_output_dir("faiss_indexed_data", output_dir_path)
    if stop_at == -1:
        stop_at = len(input_data)
    # Indexing the part.
    index_with_faiss_to_file(input_data=input_data,
                             ids=ids,
                             output_file_path=make_output_dir(output_dir=output_dir_path,
                                                              output_filename=xml_file_name[
                                                                              :-4]) + "/" + part + ".index",
                             stop_at=stop_at,
                             batch_size=batch_size
                             )
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


def index_with_faiss_to_file(input_data: list, ids: list, output_file_path: str, batch_size: int, stop_at: int):
    """
       Indexes list of text with "UWB-AIR/MQDD-duplicates" tokenizer and saves it to file

     :param input_data: A list of strings from xml file
     :type input_data: list
     :param ids: A list of unique identifiers for each item in the input_data list. These identifiers will be used to
     retrieve the corresponding item from the index later on
     :type ids: list
     :param output_file_path: The output file path is a string that specifies the location and name of the file where the
     indexed data will be saved
     :param stop_at: `stop_at` is an optional parameter that specifies the maximum number of search results to return.
     defaults is 50 (optional)
    :param batch_size: The batch size parameter determines how many data points are processed at once during the indexing
    process. This can affect the speed and memory usage of the indexing process
    :type batch_size: int
    """
    # must be divisible by len of input data
    if stop_at % batch_size > 0:
        batch_size -= stop_at % batch_size
    if stop_at % batch_size > 0:
        batch_size -= stop_at % batch_size
    # cuts of data (for testing purposes)
    print("batch_size is changed to:", batch_size)
    input_data = input_data[:stop_at]
    ids = ids[:stop_at]

    tokenizer, model, tokenized_question_example, this_device = prepare_tok_model()
    t1 = time.time()
    # sanitizing input data from html tags
    print("sanitizing input data from html tags")
    data = [sanitize_html_for_web(i.replace("\n", ""), display_code=False) for i in input_data]
    # encoding
    data = encode_questions(data, tokenizer, model, tokenized_question_example, this_device, batch_size=batch_size)
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


def concatenate_two_indexed_files(index1_path: str, index2_path: str, output_file_path: str):

    # Load the first index
    index1 = read_index(index1_path)
    n1, d1 = index1.ntotal, index1.d

    # Load the second index
    index2 = read_index(index2_path)
    n2, d2 = index2.ntotal, index2.d

    # Check that the two indexes have the same dimensionality
    if d1 != d2:
        raise ValueError("Dimensionality of indexes doesn't match")

    # Retrieve the vectors and their IDs from the first index
    ids1 = np.empty((n1,), dtype=np.int64)
    vectors1 = np.empty((n1, d1), dtype=np.float32)
    index1.search(np.zeros((1, d1), dtype=np.float32), n1)
    index1.reconstruct_n(0, n1, vectors1)
    index1.get_ids(ids1)

    # Retrieve the vectors and their IDs from the second index
    ids2 = np.empty((n2,), dtype=np.int64)
    vectors2 = np.empty((n2, d2), dtype=np.float32)
    index2.search(np.zeros((1, d2), dtype=np.float32), n2)
    index2.reconstruct_n(0, n2, vectors2)
    index2.get_ids(ids2)

    # Concatenate the vectors and IDs
    ids = np.concatenate((ids1, ids2))
    vectors = np.concatenate((vectors1, vectors2))

    # Create a new index with ID mapping
    new_index = IndexFlatL2(index1)

    # Add the merged vectors and IDs to the new index
    new_index.add_with_ids(vectors, ids)

    # Save the new index to a file
    write_index(new_index, output_file_path)
