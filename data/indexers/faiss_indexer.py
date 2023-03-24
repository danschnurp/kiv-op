#  date: 23. 3. 2023
#  author: Daniel Schnurpfeil
#
import numpy as np
from transformers import AutoTokenizer
from faiss import write_index, IndexFlatL2
from lxml.etree import XMLParser, parse

from data.utils import sanitize_html_for_web


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
    print(desired_filename, "loaded...")
    return data


def index_with_faiss_to_file(input_data: list, output_file_path: str, max_length=512,
                             tokenizer=AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")):
    """
    Indexes list of text with "UWB-AIR/MQDD-duplicates" tokenizer and saves it to file
    :param output_file_path: output file path
    :param input_data: list of strings to be indexed
    :type input_data: list
    :param max_length: the maximum length of the input sequence, defaults to 512 (optional)
    :param tokenizer: the tokenizer to use
    """
    # Converting the text into a sequence of numbers.
    data = [tokenizer.encode(
        # Removing the HTML tags from the text.
        sanitize_html_for_web(i),
        max_length=max_length,
        truncation=True, return_tensors="pt") for i in input_data]

    #  todo "max_length" could be optimized to max value of "data" for saving memory on disk 🤔
    indexer = IndexFlatL2(max_length)  # build the index
    # Creating a matrix of zeros with the size of the number of data and the maximum length of the data.
    data_matrix = np.zeros((len(data), max_length))
    for index, i in enumerate(data):
        # Converting the tensor into a numpy array and then adding it to the matrix.
        j = np.squeeze(i.detach().numpy())
        data_matrix[index, :len(j)] = j
    # Adding the matrix to the indexer.
    indexer.add(data_matrix)
    # Saving the indexed data to a file.
    write_index(indexer, output_file_path)
