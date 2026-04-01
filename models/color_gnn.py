import torch
from torch import nn
from torch_geometric.nn import GATConv

from common.config import cfg
from data.crop_dataset import get_dataset_info
from models import register


class EdgeMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)
        return self.net(edge_features).squeeze(-1)


class ColorRouteGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_colors: int, num_layers: int = 2):
        super().__init__()
        self.num_colors = num_colors
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=4, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))

        self.color_head = nn.Linear(hidden_dim, num_colors)
        self.route_head = EdgeMLP(hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        input_color_probs = x[:, :self.num_colors]

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)

        color_logits = self.color_head(x) + input_color_probs
        route_logits = self.route_head(x, edge_index)

        return color_logits, route_logits


def _build_color_gnn(model_name: str):
    mcfg = cfg.model_cfg(model_name)
    hidden_dim = mcfg.get("hidden_dim", 128)
    num_layers = mcfg.get("num_layers", 2)
    color_model_type = mcfg.get("color_model_type", "cnn")

    if color_model_type == "catboost":
        from models.color_handcrafted import HandcraftedColorClassifier
        hc = HandcraftedColorClassifier.load(mcfg["color_weights"])
        num_colors = len(hc.class_names)
    else:
        color_model = mcfg.get("color_model", "hold_color_classifier")
        info = get_dataset_info(color_model)
        num_colors = info.num_classes

    in_dim = num_colors + 5
    return ColorRouteGNN(in_dim, hidden_dim, num_colors, num_layers)


@register("color_gnn")
class ColorGNN:
    def __new__(cls, **kwargs):
        return _build_color_gnn("color_gnn")
