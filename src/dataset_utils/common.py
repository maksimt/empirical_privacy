from dataset_utils import recsys_datasets, regression_datasets, text_datasets


def load_dataset(dataset_name):
    loaders = [recsys_datasets.load_dataset, regression_datasets.load_dataset,
               text_datasets.load_dataset]
    for loader in loaders:
        try:
            return loader(dataset_name)
        except ValueError:
            pass

    return None
