import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from typing import Optional
from torch_geometric.data import Data

class AttackGraphGenerator:
    def __init__(self,
                 feature_sim_threshold: float = 0.7,
                 time_threshold: int = 60,
                 k_nearest: int = 5,
                 device: str = 'cuda'):
        """
        Initialize the Attack Graph Generator

        Args:
            feature_sim_threshold: Minimum cosine similarity for connection
            time_threshold: Maximum time difference (seconds) for connection
            k_nearest: Number of fallback nearest neighbors if no connections
            device: Device for tensor operations ('cuda' or 'cpu')
        """
        self.feature_sim_threshold = feature_sim_threshold
        self.time_threshold = time_threshold
        self.k_nearest = k_nearest
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()

    def _compute_similarities(self, features: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix"""
        return cosine_similarity(features)

    def _create_edges(self, timestamps: np.ndarray, feature_sim: np.ndarray) -> tuple:
        """
        Create edges based on temporal and feature similarity

        Args:
            timestamps: Array of event timestamps
            feature_sim: Precomputed feature similarity matrix

        Returns:
            Tuple of (edges, edge_attributes)
        """
        edges = []
        edge_attrs = []
        num_nodes = len(timestamps)

        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                time_diff = abs(timestamps[i] - timestamps[j])

                if (time_diff < self.time_threshold and
                    feature_sim[i,j] > self.feature_sim_threshold):
                    # Add bidirectional edges
                    edges.extend([[i, j], [j, i]])
                    # Edge attributes: [feature_sim, time_weight]
                    edge_attrs.extend([
                        [feature_sim[i,j], 1/(1+time_diff)],
                        [feature_sim[i,j], 1/(1+time_diff)]
                    ])

        return edges, edge_attrs

    def _fallback_connections(self, timestamps: np.ndarray, feature_sim: np.ndarray) -> tuple:
        """
        Create fallback connections using k-nearest neighbors

        Args:
            timestamps: Array of event timestamps
            feature_sim: Precomputed feature similarity matrix

        Returns:
            Tuple of (edges, edge_attributes)
        """
        edges = []
        edge_attrs = []
        num_nodes = len(timestamps)

        for i in range(num_nodes):
            # Get indices of k most similar nodes (excluding self)
            sims = feature_sim[i]
            top_k = np.argpartition(sims, -self.k_nearest-1)[-self.k_nearest-1:-1]

            for j in top_k:
                if i != j:
                    edges.append([i, j])
                    time_diff = abs(timestamps[i] - timestamps[j])
                    edge_attrs.append([feature_sim[i,j], 1/(1+time_diff)])

        return edges, edge_attrs

    def generate_graph(self,
                      features: np.ndarray,
                      timestamps: np.ndarray,
                      labels: Optional[np.ndarray] = None) -> Data:
        """
        Generate PyG Data object from raw features

        Args:
            features: Node features array (n_nodes x n_features)
            timestamps: Timestamps for each node
            labels: Optional labels for each node (can be string or numeric)

        Returns:
            PyTorch Geometric Data object
        """
        # Compute feature similarities
        feature_sim = self._compute_similarities(features)

        # Create edges based on similarity and temporal proximity
        edges, edge_attrs = self._create_edges(timestamps, feature_sim)

        # Fallback to k-nearest if no connections were made
        if not edges:
            edges, edge_attrs = self._fallback_connections(timestamps, feature_sim)

        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).to(self.device)
        x = torch.tensor(features, dtype=torch.float).to(self.device)

        # Create PyG Data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Add labels if provided
        if labels is not None:
            # Convert string labels to numeric if needed
            if labels.dtype == np.object_:
                y = self.label_encoder.fit_transform(labels)
            else:
                y = labels
            y = torch.tensor(y, dtype=torch.long).to(self.device)
            graph_data.y = y

        return graph_data