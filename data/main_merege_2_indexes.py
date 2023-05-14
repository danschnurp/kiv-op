#  date: 13. 5. 2023
#  author: Daniel Schnurpfeil
#
from faiss_indexer import concatenate_two_indexed_files

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input1_file_path', required=True)
    parser.add_argument('-i2', '--input2_file_path', required=True)
    parser.add_argument('-o', '--output_file_path', required=True)
    args = parser.parse_args()

    if os.path.isfile(args.input1_file_path) and os.path.isfile(args.input2_file_path):
        concatenate_two_indexed_files(args.input1_file_path, args.input2_file_path, args.output_file_path,
                                      input_folder="D:/_KIV_OP/stackoverflow.com-Posts", offset=0, stop_at=200)
