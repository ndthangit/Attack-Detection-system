import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

class GraphGenerator:
    def __init__(self, threshold=0.8, time_window=2):
        """
        threshold: Ngưỡng cosine similarity để tạo cạnh
        time_window: Khoảng cách thời gian tối đa để tạo cạnh
        """
        self.threshold = threshold
        self.time_window = time_window
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate(self, features, labels, time_steps):
        """
        features: numpy array (n_samples, 768) - Đặc trưng từ DistilBERT
        labels: numpy array (n_samples,) - Nhãn (0: benign, 1-5: tấn công)
        time_steps: numpy array (n_samples,) - Thời gian của mỗi mẫu
        """
        num_nodes = features.shape[0]
        # Chuyển đặc trưng thành tensor
        x = torch.tensor(features, dtype=torch.float).to(self.device)

        # Tính cosine similarity để tạo cạnh
        sim_matrix = cosine_similarity(features)
        edge_index = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if (sim_matrix[i, j] > self.threshold or
                        abs(time_steps[i] - time_steps[j]) < self.time_window):
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)

        # Tạo đối tượng Data của PyG
        data = Data(x=x, edge_index=edge_index, y=y)
        return data