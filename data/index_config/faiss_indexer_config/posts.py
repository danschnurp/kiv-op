#  date: 24. 3. 2023
#  author: Daniel Schnurpfeil
#
from data.indexers.faiss_indexer import index_part

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir_path', required=True)
    parser.add_argument('-f', '--input_file_name', required=True)
    args = parser.parse_args()

    input_folder = args.input_dir_path
    input_filename = args.input_file_name

    if not os.path.isdir(input_folder):
        raise "bad input_file_path..."

    index_part(input_folder=input_folder, xml_file_name=input_filename, part="Body")
    index_part(input_folder=input_folder, xml_file_name=input_filename, part="Title")
