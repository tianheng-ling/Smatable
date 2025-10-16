from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, x_data, y_data, downsampling_rate=None):
        self.x = x_data
        self.y = y_data
        self.downsampling_rate = downsampling_rate

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.downsampling_rate is not None:
            x = self.x[idx][:: self.downsampling_rate]
        else:
            x = self.x[idx]
        return x, self.y[idx]


class AugumentedAudioDataset(Dataset):
    def __init__(self, x_data, y_data, downsampling_rate=None):
        self.x = x_data
        self.y = y_data
        self.downsampling_rate = downsampling_rate or 1

        self.original_len = len(self.y)
        self.total_len = self.original_len * self.downsampling_rate

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        original_idx = idx // self.downsampling_rate
        start_offset = idx % self.downsampling_rate

        x_original = self.x[original_idx]
        x_downsampled = x_original[start_offset :: self.downsampling_rate]
        y = self.y[original_idx]

        return x_downsampled, y
