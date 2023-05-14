#  date: 23. 3. 2023
#  author: Daniel Schnurpfeil
#
import time
from xml.etree.ElementTree import iterparse

import faiss
import numpy as np
from faiss import write_index, IndexFlatL2, read_index, IndexIVFFlat, METRIC_L2, vector_to_array, extract_index_ivf

from utils import make_output_dir, sanitize_html_for_web
from question_encoder import encode_questions, prepare_tok_model


def index_part(input_folder: str, xml_file_name: str, part: str, batch_size=4, offset=0., stop_at=float("inf"),
               output_dir_path=None):
    xpath_query = './/*[@class="' + part + '"]/@' + part
    xpath_query_ids = './/*[@class="' + "Id" + '"]/@' + 'Id'
    # Loading the data from the xml file.
    print("loading:", xml_file_name)
    input_data, ids = load_xml_data(input_folder=input_folder, offset=offset,
                                    stop_at=stop_at, desired_filename="/" + xml_file_name, part=part)

    print(len(ids), xml_file_name, part, "loaded...")
    t1 = time.time()
    if not output_dir_path:
        output_dir_path = input_folder
    output_dir_path = make_output_dir("faiss_indexed_data", output_dir_path)
    # Indexing the part.
    index_with_faiss_to_file(input_data=input_data,
                             ids=ids,
                             output_file_path=make_output_dir(output_dir=output_dir_path,
                                                              output_filename=xml_file_name[
                                                                              :-4]) + "/" + part + "_" + str(offset)[
                                                                                                         :-2] + "_" + str(
                                 stop_at)[:-2] + ".index",
                             stop_at=stop_at,
                             offset=offset,
                             batch_size=batch_size
                             )
    print(part, "part processed in:", time.time() - t1, "sec")


def load_xml_data(input_folder: str, desired_filename: str, stop_at: float, offset: float,
                  part: str, ids_only=False) -> tuple:
    """
    Loads XML data from a file in a folder, and returns a list of the data.

    :param input_folder: The folder where the XML file is located
    :type input_folder: str
    :param desired_filename: The name of the file you want to load
    :type desired_filename: str
    :param xpath_query: This is the xpath query that will be used to extract the data from the XML file
    :type xpath_query: str
    """

    data = []
    ids = []
    counter = 0.

    # open the XML file
    with open(input_folder + desired_filename, 'rb') as f:
        # create an iterator for the XML file
        context = iterparse(f, events=('start', 'end'))

        # get the root element
        _, root = next(context)

        # loop through the elements
        for event, elem in context:
            if event == 'end' and elem.tag == 'row':
                if offset <= counter < stop_at:
                    if not ids_only:
                        data.append(elem.attrib[part])
                    ids.append(int(elem.attrib["Id"]))
                counter += 1.
                # clear the element to free memory
                root.clear()
                if counter > stop_at:
                    break
    return data, ids


def index_with_faiss_to_file(input_data: list, ids: list, output_file_path: str, batch_size: int,
                             offset: float, stop_at: float):
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

    tokenizer, model, tokenized_question_example, this_device = prepare_tok_model()
    t1 = time.time()
    # sanitizing input data from html tags
    print("sanitizing input data from html tags")
    data = [sanitize_html_for_web(i.replace("\n", ""), display_code=False) for i in input_data]
    # encoding
    data = encode_questions(data, tokenizer, model, tokenized_question_example, this_device, batch_size=batch_size)
    print("sanitized and encoded in:", time.time() - t1, "sec")

    quantizer = IndexFlatL2(data.shape[1])  # the quantizer used for clustering
    indexer = IndexIVFFlat(quantizer, data.shape[1], data.shape[0], METRIC_L2)

    indexer.train(data)
    # Adding the matrix to the indexer.
    indexer.add_with_ids(data, ids)
    # Saving the indexed data to a file.
    print("saving to", output_file_path)
    write_index(indexer, output_file_path)


def concatenate_two_indexed_files(index1_path: str, index2_path: str, output_file_path: str):
    # Load the first index
    index1 = read_index(index1_path)

    # Load the second index from a file
    index2 = read_index(index2_path)

    # Merge the indexes todo this does not work at all
    faiss.merge_into(index1, index2, True)

    # Save the merged index to a file
    write_index(index1, output_file_path)
