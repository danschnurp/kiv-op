#  date: 24. 3. 2023
#  author: Daniel Schnurpfeil
#
from data.index_config.faiss_indexer_config.cfg import input_folder
from data.indexers.faiss_indexer import index_part

xml_file_name = "posts"

index_part(input_folder=input_folder, xml_file_name=xml_file_name, part="Body")
index_part(input_folder=input_folder, xml_file_name=xml_file_name, part="Title")
