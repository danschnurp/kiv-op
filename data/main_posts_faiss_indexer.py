#  date: 24. 3. 2023
#  author: Daniel Schnurpfeil
#
from faiss_indexer import index_part

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir_path', required=True)
    parser.add_argument('-b', '--batch_size', default=5, type=int)
    parser.add_argument('-s', '--stop_at', default=-1, type=int)
    parser.add_argument('-o', '--output_dir_path', default=None)
    args = parser.parse_args()

    input_folder = args.input_dir_path

    output_dir_path = args.output_dir_path

    if os.path.isdir(input_folder) and (not output_dir_path or os.path.isdir(output_dir_path)):

        index_part(input_folder=input_folder, xml_file_name="Posts.xml", part="Body",
                   batch_size=args.batch_size, stop_at=args.stop_at, output_dir_path=output_dir_path)
        # index_part(input_folder=input_folder, xml_file_name="Posts.xml", part="Title",
        #            batch_size=args.batch_size, stop_at=args.stop_at, output_dir_path=output_dir_path)

    else:
        raise "bad input_file_path..."
