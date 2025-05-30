# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from typing import Tuple, Union, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X: Union[List[np.ndarray], np.ndarray], Y: np.ndarray):
        if isinstance(X, list):
            self.X = [torch.from_numpy(x).float() for x in X]
        else:
            self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def collate_fn(batch, padding_value=0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)  # Unzip the batch into sequences and labels
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=padding_value)  # (batch_size, max_length, embedding_dim)
    pad_mask = (padded_sequences.sum(dim=-1) != padding_value).float()  # (batch_size, max_length)
    return padded_sequences, pad_mask, torch.stack(labels)

