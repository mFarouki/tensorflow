import random
import shutil
import sys
from os import path, listdir, remove

import tensorflow as tf

sys.path.append('../')
from utilities.utility_functions import display_info


def remove_file(file_path):
    try:
        display_info(f'Removing file {file_path} if exists')
        remove(file_path)
    except OSError:
        pass


def download_and_unzip(filename: str, url: str, output_directory: str):
    if not path.isdir(output_directory):
        tf.keras.utils.get_file(filename, url, untar=True, cache_dir='..', cache_subdir='')
    compressed_file = path.basename(url)
    dataset_dir = path.join(path.dirname(compressed_file), output_directory)
    remove_file(path.join(path.dirname(compressed_file), compressed_file))
    return dataset_dir


def check_directory_contents(directory_path: str, expected_contents: list):
    display_info(f'Checking contents of {directory_path} against expected contents')
    if listdir(directory_path).sort() != expected_contents.sort():
        raise ValueError(f'Contents of directory {directory_path} do not match expected contents {expected_contents}')


def read_random_review(data_dir: str):
    display_info('Showing random review')
    random_sentiment = random.randint(0, 1)
    if random_sentiment:
        sentiment = 'neg'
    else:
        sentiment = 'pos'
    sentiment_directory = path.join(data_dir, f'{sentiment}/')
    sample_file = random.choice(listdir(sentiment_directory))
    with open(path.join(sentiment_directory, sample_file)) as f:
        print(f'Sample review {sample_file} reads as follows:')
        print(f.read())


def remove_unneeded_directories(parent_directory: str, child_directory: str):
    remove_dir = path.join(parent_directory, child_directory)
    if path.isdir(remove_dir):
        shutil.rmtree(remove_dir)

