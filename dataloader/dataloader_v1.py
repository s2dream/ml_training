import numpy as np
import pickle
import torch
import traceback
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class CustomDataset(Dataset):
    def __init__(self, bin_file_path, idx_file_path):
        self.list_offset = []
        with open(idx_file_path, "rb") as fread:
            self.list_offset = pickle.load(fread)
        self.data_file_pointer = None
        try:
            self.data_file_pointer = open(bin_file_path, "rb")
        except Exception as e:
            print(e)
            traceback.print_stack()

    def __len__(self):
        return len(self.list_offset)

    def __del__(self):
        if self.data_file_pointer is not None:
            try:
                self.data_file_pointer.cloes
            except Exception as e:
                print(e)
                traceback.print_stack()

    def __getitem__(self, idx):
        target_offset = self.list_offset[idx]
        self.data_file_pointer.tell(target_offset)
        ret_item = pickle.load(self.data_file_pointer)
        return ret_item


def transformer_collate_fn(batch, dict_max_size: dict, pad_token: int = 0):
    if len(dict_max_size) == 0:
        raise Exception("error in transformer_collate_fn: len(dict_max_size) == 0")

    main_seq_max_len = dict_max_size["main_seq_max_len"]
    metro_set_max_len = dict_max_size["metro_set_max_len"]
    metro_item_set_max_len = dict_max_size["metro_item_set_max_len"]

    batch_size = len(batch)
    x_main_seq_id = torch.zeros((batch_size, main_seq_max_len), dtype=torch.long)
    x_metro_set_id = torch.zeros((batch_size, main_seq_max_len, metro_set_max_len), dtype=torch.long)
    x_metro_item_set_id = torch.zeros((batch_size, main_seq_max_len, metro_set_max_len, metro_item_set_max_len),
                                      dtype=torch.long)
    x_metro_item_value = torch.zeros((batch_size, main_seq_max_len, metro_set_max_len, metro_item_set_max_len),
                                     dtype=torch.float)
    y_et = torch.zeros((batch_size, 1), dtype=torch.float)

    for idx_b_sample, sample in enumerate(batch):
        # main step
        main_len = min(len(sample["x_main_seq_id"]), main_seq_max_len)
        x_main_seq_id[idx_b_sample, :main_len] = torch.tensor(sample["x_main_seq_id"][:main_len], dtype=torch.long)

        # metro step
        for idx_main in range(main_len):
            metro_len = min(len(sample["x_metro_set_id"][idx_main]), metro_set_max_len)
            x_metro_set_id[idx_b_sample, idx_main, :metro_len] = torch.tensor(
                sample["x_metro_set_id"][idx_main][:metro_len], dtype=torch.long)

            # metro_item
            for idx_metro in range(metro_len):
                metro_item_len = min(len(sample["x_metro_item_set_id"][idx_main][idx_metro]), metro_item_set_max_len)
                x_metro_item_set_id[idx_b_sample, idx_main, idx_metro, :metro_item_len] = torch.tensor(
                    sample["x_metro_item_set_id"][idx_main][idx_metro][:metro_item_len],
                    dtype=torch.long)
                x_metro_item_value[idx_b_sample, idx_main, idx_metro, :metro_item_len] = torch.tensor(
                    sample["x_metro_item_value"][idx_main][idx_metro][:metro_item_len],
                    dtype=torch.float)

        # ET
        if isinstance(sample["y"], list) or isinstance(sample["y"], np.ndarray):
            y_et[idx_b_sample, 0] = torch.tensor(sample["y"][0], dtype=torch.float)
        else:
            y_et[idx_b_sample, 0] = torch.tensor(sample["y"], dtype=torch.float)

    padding_mask_main_steps = x_main_seq_id.eq(pad_token)
    padding_mask_metro_items = x_metro_item_set_id.eq(pad_token)

    return {
        "x_main_seq_id": x_main_seq_id,
        "x_metro_set_id": x_metro_set_id,
        "x_metro_item_set_id": x_metro_item_set_id,
        "x_metro_item_value": x_metro_item_value,
        "padding_mask_main_steps": padding_mask_main_steps,
        "y": y
    }

def create_dataloader(dataset,
                      batch_size,
                      num_workers=4,
                      pin_memory=True,
                      collate_fn=None):
    world_size = torch.cuda.device_count()
    if world_size > 1:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    data = [
        {
            'x': [{'input_ids': 10}, {'input_ids': 11}, {'input_ids': 12}],
            'y': {'taskA': 0, 'taskB': 1}
        },
        {
            'x': [{'input_ids': 20}, {'input_ids': 21}],
            'y': {'taskA': 1, 'taskB': 0}
        },
        # ...
    ]

    dataset = CustomDataset(data)
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        num_workers=2,
        collate_fn=transformer_collate_fn
    )

    for batch in dataloader:
        x_feats = batch['x']  # {'input_ids': Tensor[B,L], ...}
        mask = batch['attention_mask']  # Tensor[B,L]
        y = batch['y']  # {'taskA': Tensor[B], ...}
        # model 호출 예:
        # outputs = model(input_ids=x_feats['input_ids'], attention_mask=mask, labels=y)
        pass
