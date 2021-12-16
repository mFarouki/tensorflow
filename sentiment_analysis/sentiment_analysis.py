import os

from download_and_view import *

imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
expected_top_level_contents = ['train', 'test', 'README', 'imdbEr.txt', 'imdb.vocab']
expected_train_contents = ['labeledBow.feat', 'urls_pos.txt', 'urls_unsup.txt', 'unsup', 'pos', 'unsupBow.feat',
                           'urls_neg.txt', 'neg']


def main():
    dataset_dir = download_and_unzip("aclImdb_v1", imdb_url, 'aclImdb')
    check_directory_contents(dataset_dir, expected_top_level_contents)
    train_dir = os.path.join(dataset_dir, 'train')
    read_random_review(train_dir)
    remove_unneeded_directories(train_dir, 'unsup')


if __name__ == "__main__":
    main()
