import torch
from torch_geometric.data import Data, Dataset


def build_graph(
    boxes: torch.Tensor,
    color_logits: torch.Tensor,
    scores: torch.Tensor,
    img_w: int,
    img_h: int,
    k: int = 6,
    color_labels: torch.Tensor = None,
) -> Data:
    n = len(boxes)
    if n == 0:
        return None

    cx = ((boxes[:, 0] + boxes[:, 2]) / 2) / img_w
    cy = ((boxes[:, 1] + boxes[:, 3]) / 2) / img_h
    bw = (boxes[:, 2] - boxes[:, 0]) / img_w
    bh = (boxes[:, 3] - boxes[:, 1]) / img_h

    spatial = torch.stack([cx, cy, bw, bh, scores], dim=1)
    node_features = torch.cat([color_logits, spatial], dim=1)

    centers = torch.stack([cx, cy], dim=1)
    dists = torch.cdist(centers, centers)
    dists.fill_diagonal_(float("inf"))

    actual_k = min(k, n - 1)
    if actual_k < 1:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        _, knn_idx = dists.topk(actual_k, largest=False, dim=1)
        src = torch.arange(n).unsqueeze(1).expand_as(knn_idx).reshape(-1)
        dst = knn_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)

    data = Data(x=node_features, edge_index=edge_index)
    data.boxes = boxes
    data.num_nodes_graph = n

    if color_labels is not None:
        data.y = color_labels
        src_labels = color_labels[edge_index[0]]
        dst_labels = color_labels[edge_index[1]]
        data.edge_labels = (src_labels == dst_labels).float()

    return data


class GNNGraphDataset(Dataset):
    def __init__(self, graphs_path: str):
        super().__init__()
        self.graphs = torch.load(graphs_path, weights_only=False)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]
