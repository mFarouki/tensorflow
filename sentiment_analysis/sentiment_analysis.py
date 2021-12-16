import os

from download_and_view import *
from load_raw_datasets import *
from preprocess_data import *

imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
expected_top_level_contents = ['train', 'test', 'README', 'imdbEr.txt', 'imdb.vocab']
expected_train_contents = ['labeledBow.feat', 'urls_pos.txt', 'urls_unsup.txt', 'unsup', 'pos', 'unsupBow.feat',
                           'urls_neg.txt', 'neg']


def create_directory_structure():
    dataset_dir = download_and_unzip("aclImdb_v1", imdb_url, 'aclImdb')
    check_directory_contents(dataset_dir, expected_top_level_contents)
    train_dir = os.path.join(dataset_dir, 'train')
    read_random_review(train_dir)
    remove_unneeded_directories(train_dir, 'unsup')
    return dataset_dir


def main():
    dataset_dir = create_directory_structure()
    raw_train_dataset, raw_validation_dataset, raw_test_dataset = build_raw_datasets(dataset_dir)
    view_dataset(raw_validation_dataset)
    vectorised_layer = apply_vectorisation(raw_train_dataset)
    view_sample_vectorisation(raw_train_dataset, vectorised_layer)
    train_dataset, validation_dataset, test_dataset = preprocess_dataset(raw_train_dataset), \
        preprocess_dataset(raw_validation_dataset), preprocess_dataset(raw_test_dataset)


if __name__ == "__main__":
    main()
