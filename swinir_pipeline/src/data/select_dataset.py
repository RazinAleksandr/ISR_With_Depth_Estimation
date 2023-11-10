from src.data.dataset_sr import DatasetSR as D


def define_Dataset(dataset_opt, depth=False):
    dataset_type = dataset_opt['dataset_type'].lower()


    # -----------------------------------------
    # super-resolution
    # -----------------------------------------
    if dataset_type in ['sr', 'super-resolution']:
        dataset = D(dataset_opt, depth)
        print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
        return dataset
