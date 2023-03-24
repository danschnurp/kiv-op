#  date: 24. 3. 2023
#  author: Daniel Schnurpfeil
#
import time

from data.indexers.faiss_indexer import load_xml_data, index_with_faiss_to_file
from data.utils import make_output_dir


def index_comments(part: str):
    """
    indexes comments

    :param part: str
    :type part: str
    """
    # Loading the data from the xml file.
    input_data = load_xml_data(input_folder=input_folder,
                               desired_filename="/" + comments + ".xml", xpath_query="//row/@" + part)
    t1 = time.time()
    # Indexing the comments.
    index_with_faiss_to_file(input_data=input_data,
                             output_file_path=make_output_dir(output_dir=input_folder,
                                                              output_filename=comments) + "/" + part + ".index")
    print(part, "part processed in:", time.time() - t1, "sec")


input_folder = "../../../../../SERVER_APPS/gamedev.stackexchange.com/"
comments = "comments"

index_comments(part="Text")
