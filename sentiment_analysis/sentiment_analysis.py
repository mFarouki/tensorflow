import os

from download_and_view import download_and_unzip, check_directory_contents, read_random_review, \
    remove_unneeded_directories
from load_raw_datasets import build_raw_datasets, view_dataset
from preprocess_data import apply_vectorisation, view_sample_vectorisation, preprocess_dataset
from build_model import embedded_neural_net, compile_model

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
    train_dataset, validation_dataset, test_dataset = \
        preprocess_dataset(raw_train_dataset, vectorised_layer), \
        preprocess_dataset(raw_validation_dataset, vectorised_layer), \
        preprocess_dataset(raw_test_dataset,vectorised_layer)
    model_structure = embedded_neural_net(10000)
    model = compile_model(model_structure)
    trained_model = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
    loss, accuracy = model.evaluate(test_dataset)


if __name__ == "__main__":
    main()
