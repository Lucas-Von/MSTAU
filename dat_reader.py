import random

import numpy as np
import torch
from torch.utils.data import Dataset

def get_spike_matrix(
        filename,
        flipud=True,
        height=256,
        width=448,
        frame_num=91
):
    file_reader = open(filename, 'rb')
    spk_seq = file_reader.read()
    spk_seq = np.frombuffer(spk_seq, 'b')
    spk_seq = np.array(spk_seq).astype(np.byte)

    pix_id = np.arange(0, frame_num * height * width)
    pix_id = np.reshape(pix_id, (frame_num, height, width))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    data = spk_seq[byte_id]
    result = np.bitwise_and(data, comparator)
    SpikeMatrix = (result == comparator)

    if flipud:
        SpikeMatrix = SpikeMatrix[:, ::-1, :]
    file_reader.close()

    return SpikeMatrix

class spk_provider(Dataset):
    def __init__(self, spk_path_file, half_win=20, total_len=10, random_start=True):
        super(spk_provider, self).__init__()

        self.spk_path_file = spk_path_file
        self.half_win = half_win
        self.total_len = total_len
        self.random_start = random_start

        self.spk_path_list = []
        with open(spk_path_file) as f:
            for l in f.readlines:
                self.spk_path_list.append(l)

    def __getitem__(self, idx):
        spk_path = self.spk_path_list[idx]

        spk_voxel = get_spike_matrix(spk_path, flipud=False)
        win_len = 2 * self.half_win + 1
        if self.total_len == -1:
            spk = spk_voxel
        else:
            if self.random_start:
                start_idx = random.randint(0, spk_voxel.shape[0] - win_len - self.total_len)
            else:
                start_idx = 0
            spk = spk_voxel[start_idx: start_idx + win_len + self.total_len]

        spk = torch.from_numpy(spk).float()
        return spk

    def __len__(self):
        return len(self.spk_path_list)
