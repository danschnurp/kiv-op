import argparse
import os

parser = argparse.ArgumentParser(description='Simple indexer')
parser.add_argument('-i', '--input_file_path',
                    required=True)
args = parser.parse_args()
if not os.path.isfile(args.input_file_path):
    raise "bad input_file_path..."

input_folder = args.input_file_path

