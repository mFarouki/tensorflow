import random
import sys
from os import listdir
from pathlib import Path

import tensorflow as tf

sys.path.append('../')
from utilities.utility_functions import display_info

imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
expected_top_level_contents = ['train', 'test', 'README', 'imdbEr.txt', 'imdb.vocab']
expected_train_contents = ['labeledBow.feat', 'urls_pos.txt', 'pos', 'urls_neg.txt', 'neg']
output_directory = 'aclImdb'


def remove_contents(parent_directory: Path, child: str):
    remove_path = parent_directory / child
    try:
        if remove_path.is_dir():
            remove_path.rmdir()
        elif remove_path.is_file():
            remove_path.unlink()
    except OSError:
        pass


def download_and_unzip(filename: str, url: str, output_directory_name: str):
    output_directory_parent = Path.cwd().parent
    dataset_directory = output_directory_parent / output_directory_name
    if not dataset_directory.is_dir():
        tf.keras.utils.get_file(filename, url, untar=True, cache_dir='../', cache_subdir='')
    remove_contents(output_directory_parent, Path(url).name)
    return dataset_directory


def check_directory_contents(directory_path: Path, expected_contents: list):
    display_info(f'Checking contents of {directory_path} against expected contents')
    if listdir(directory_path).sort() != expected_contents.sort():
        raise ValueError(f'Contents of directory {directory_path} do not match expected contents {expected_contents}')


def read_random_review(data_dir: Path):
    display_info('Showing random review')
    random_sentiment = random.randint(0, 1)
    if random_sentiment:
        sentiment = 'neg'
    else:
        sentiment = 'pos'
    sentiment_directory = data_dir.joinpath(f'{sentiment}/')
    sample_file = random.choice(listdir(sentiment_directory))
    with open(sentiment_directory / sample_file) as f:
        print(f'Sample review {sample_file} reads as follows:')
        print(f.read())


def remove_unneeded_from_train(train_dir: Path):
    for item in ['unsup', 'urls_unsup.txt', 'unsupBow.feat']:
        remove_contents(train_dir, item)


def create_directory_structure():
    dataset_dir = download_and_unzip("aclImdb_v1", imdb_url, output_directory)
    check_directory_contents(dataset_dir, expected_top_level_contents)
    train_dir = dataset_dir / 'train'
    remove_unneeded_from_train(train_dir)
    check_directory_contents(train_dir, expected_train_contents)
    read_random_review(train_dir)
    return dataset_dir
