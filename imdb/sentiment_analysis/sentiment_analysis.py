import sys

sys.path.append('../../')

from imdb.download_and_view import get_data
from imdb.load_raw_datasets import build_raw_datasets, view_dataset
from preprocess_data import apply_vectorisation, view_sample_vectorisation, preprocess_dataset
from train_model import train_model
from evaluate_model import visualise_training
from export_model import export_model

batch_size = 32
seed = 42


def main():
    dataset_dir = get_data()
    raw_train_dataset, raw_validation_dataset, raw_test_dataset = build_raw_datasets(dataset_dir, batch_size, seed)
    view_dataset(raw_validation_dataset)
    vectorised_layer = apply_vectorisation(raw_train_dataset)
    max_features = len(vectorised_layer.get_vocabulary())
    view_sample_vectorisation(raw_train_dataset, vectorised_layer)
    train_dataset, validation_dataset, test_dataset = \
        preprocess_dataset(raw_train_dataset, vectorised_layer), \
        preprocess_dataset(raw_validation_dataset, vectorised_layer), \
        preprocess_dataset(raw_test_dataset, vectorised_layer)
    model, history = train_model(train_dataset, validation_dataset, max_features)
    visualise_training(history, model, test_dataset)
    model_to_export = export_model(vectorised_layer, model)


if __name__ == "__main__":
    main()
