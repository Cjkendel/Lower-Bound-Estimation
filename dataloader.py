import torch
from dataset_chocolate import DataSet


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, full_batch=False, batch_size=False, shuffle=True,
                 n_batch=False, sampler=None, standardize=False,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        assert sum([full_batch, bool(batch_size), bool(n_batch)]) <=1, "Choose only one batch method"
        assert sum([full_batch, bool(batch_size), bool(n_batch)]) > 0, "Please choose a batch method"

        if isinstance(dataset, str):
            dataset = DataSet(dataset, standardize=standardize)

        self.filename = dataset
        self.input_dims = dataset.inputs.shape[1]
        self.output_dims = 1
        self.total_data_len = dataset.total_data_len
        self.full_batch = (self.total_data_len if full_batch is not False else False)
        self.n_batch = (int(self.total_data_len / n_batch) if n_batch is not False else False)
        self.batch_size = (batch_size if self.full_batch is not True and self.n_batch is not True else False)
        self.batch_index = int([i for i, x in enumerate([full_batch, batch_size, n_batch]) if x is not False][0])
        self.batch_size = int([self.full_batch, self.batch_size, self.n_batch][self.batch_index])
        self.num_batches = int(self.total_data_len / self.batch_size) + 1

        super().__init__(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler,
                         batch_sampler=batch_sampler, num_workers=num_workers,
                         pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context)
