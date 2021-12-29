import sys

sys.path.append('../../')

from imdb.download_and_view import get_data
from imdb.load_raw_datasets import build_raw_datasets, view_dataset
from performance import presets
from explore_embedding import explore

batch_size = 1024
seed = 132


def main():
    dataset_dir = get_data()
    raw_train_dataset, raw_validation_dataset, _ = build_raw_datasets(dataset_dir, batch_size, seed)
    view_dataset(raw_validation_dataset)
    raw_train_dataset, raw_validation_dataset = presets(raw_train_dataset), presets(raw_validation_dataset)
    explore()




if __name__ == "__main__":
    main()
