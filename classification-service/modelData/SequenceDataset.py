from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, features, delta_t, labels, seq_len):
        self.features = features
        self.delta_t = delta_t
        self.labels = labels
        self.seq_len = seq_len
        self.num_samples = len(features) - seq_len + 1
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq_features = self.features[idx:idx+self.seq_len]
        seq_delta_t = self.delta_t[idx:idx+self.seq_len]
        seq_labels = self.labels[idx:idx+self.seq_len]
        # Ensure delta_t has shape (seq_len, 1)
        seq_delta_t = seq_delta_t.unsqueeze(-1) if seq_delta_t.dim() == 1 else seq_delta_t
        return seq_features, seq_delta_t, seq_labels
    def get_labels_for_stratification(self):
        # Trả về nhãn phân lớp cho mỗi sample (thường là nhãn cuối cùng trong sequence)
        return [self.labels[idx+self.seq_len-1] for idx in range(self.num_samples)]