"""Standard supervised CPU baselines for the common node-classification protocol.

All networks have two trainable layers. Graph methods consume the identical
input adjacency; only their standard operators add self loops or normalize it.
HGNN uses one label-free structural hypergraph, without the PRISM filter, from
the repository's capped k-hop construction. Its propagation is the symmetric
operator Dv^-1/2 H De^-1 H^T Dv^-1/2; this explicitly specifies the hypergraph
construction instead of treating HGNN as a uniquely defined data transform.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GATConv, GCNConv, SAGEConv

from models.hypergraph import MultiViewHypergraphBuilder


METHODS = ("mlp", "gcn", "gat", "sage", "appnp", "hgnn")


def _sparse_tensor(matrix):
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, check_invariants=True).coalesce()


def structural_operator(graph, settings):
    builder = MultiViewHypergraphBuilder(
        hop_k=int(settings.get("hop_k", 2)),
        structural_cap=int(settings.get("structural_cap", 32)),
        structure_filter=False,
        random_state=int(settings.get("seed", 0)),
    )
    views = builder.build(
        graph.x.cpu().numpy(), graph.adj, enabled_views=["structural"],
        include_elastic=False, include_knn_pairs=False,
    )
    return {
        "h": _sparse_tensor(views.incidences[0]),
        "ht": _sparse_tensor(views.incidences[0].T.tocsr()),
        "dv_inv_sqrt": torch.from_numpy(np.sqrt(views.node_degree_inv[0])).float(),
        "de_inv": torch.from_numpy(views.edge_degree_inv[0]).float(),
    }


class HypergraphLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, operator):
        h = self.linear(x) * operator["dv_inv_sqrt"].unsqueeze(-1)
        h = torch.sparse.mm(operator["ht"], h)
        h = h * operator["de_inv"].unsqueeze(-1)
        h = torch.sparse.mm(operator["h"], h)
        return h * operator["dv_inv_sqrt"].unsqueeze(-1) + self.bias


class StandardBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, method, dropout, settings, operator=None):
        super().__init__()
        self.method = method
        self.dropout = float(dropout)
        self.operator = operator
        if method in {"mlp", "appnp"}:
            self.first = nn.Linear(input_dim, hidden_dim)
            self.second = nn.Linear(hidden_dim, output_dim)
            if method == "appnp":
                self.propagation = APPNP(K=int(settings.get("appnp_K", 10)),
                                         alpha=float(settings.get("appnp_alpha", .1)),
                                         dropout=0., cached=True)
        elif method == "gcn":
            self.first = GCNConv(input_dim, hidden_dim, cached=True)
            self.second = GCNConv(hidden_dim, output_dim, cached=True)
        elif method == "gat":
            heads = int(settings.get("gat_heads", 4))
            if hidden_dim % heads:
                raise ValueError("GAT hidden_dim must be divisible by gat_heads")
            self.first = GATConv(input_dim, hidden_dim // heads, heads=heads,
                                 concat=True, dropout=self.dropout)
            self.second = GATConv(hidden_dim, output_dim, heads=1,
                                  concat=False, dropout=self.dropout)
        elif method == "sage":
            self.first = SAGEConv(input_dim, hidden_dim, aggr="mean")
            self.second = SAGEConv(hidden_dim, output_dim, aggr="mean")
        elif method == "hgnn":
            self.first = HypergraphLayer(input_dim, hidden_dim)
            self.second = HypergraphLayer(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported standard baseline: {method}")

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.method in {"mlp", "appnp"}:
            h = F.relu(self.first(x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            logits = self.second(h)
            if self.method == "appnp":
                logits = self.propagation(logits, edge_index)
            return logits
        if self.method == "hgnn":
            h = F.relu(self.first(x, self.operator))
            h = F.dropout(h, p=self.dropout, training=self.training)
            return self.second(h, self.operator)
        h = self.first(x, edge_index)
        h = F.elu(h) if self.method == "gat" else F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.second(h, edge_index)


def make_baseline(graph, method, settings):
    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    coo = graph.adj.tocoo()
    edge_index = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    operator = structural_operator(graph, settings) if method == "hgnn" else None
    model = StandardBaseline(
        graph.num_features, int(settings.get("hidden_dim", 64)), graph.num_classes,
        method, settings.get("dropout", .5), settings, operator=operator,
    )
    return model, edge_index


def _accuracy(logits, labels, mask):
    if not bool(mask.any()):
        raise ValueError("Cannot evaluate an empty node mask")
    return float((logits[mask].argmax(-1) == labels[mask]).float().mean().item())


def train_baseline(graph, method, settings, evaluate_test=False):
    """Train using training labels, select only on validation, test once.

    Caller passes the run seed in settings['seed']. Training histories never
    contain test scores. A final run evaluates the held-out test mask only after
    restoring the earliest checkpoint with maximal validation accuracy.
    """
    preparation_started = time.perf_counter()
    torch.set_num_threads(1)
    seed = int(settings.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if graph.x.device.type != "cpu":
        raise ValueError("This benchmark runner expects CPU GraphData")
    if not bool(graph.train_mask.any()) or not bool(graph.val_mask.any()):
        raise ValueError("Training and validation masks must both be nonempty")
    if bool((graph.train_mask & graph.val_mask).any()):
        raise ValueError("Training and validation masks overlap")
    epochs = int(settings.get("epochs", 100))
    patience = int(settings.get("patience", 25))
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    if not 0 <= float(settings.get("dropout", .5)) < 1:
        raise ValueError("dropout must be in [0, 1)")
    model, edge_index = make_baseline(graph, method, settings)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(settings.get("lr", .01)),
        weight_decay=float(settings.get("weight_decay", .0005)),
    )
    preparation_seconds = time.perf_counter() - preparation_started
    training_started = time.perf_counter()
    best_score, best_epoch, best_state, bad = -1., 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(graph.x, edge_index)
        loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask],
                               label_smoothing=float(settings.get("label_smoothing", 0.)))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {method} loss at epoch {epoch}")
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("grad_clip", 5.)))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Nonfinite {method} gradient at epoch {epoch}")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(graph.x, edge_index)
            validation = _accuracy(logits, graph.y, graph.val_mask)
        if not math.isfinite(validation):
            raise FloatingPointError("Nonfinite validation score")
        history.append({"epoch": epoch, "cls": float(loss.item()), "val_score": validation})
        if validation > best_score:
            best_score, best_epoch, bad = validation, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    result = {
        "model_name": method.upper(), "method": method, "selection": "val",
        "best_val_score": best_score, "best_select_score": best_score,
        "best_epoch": best_epoch, "epochs_run": len(history),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "preparation_seconds": round(preparation_seconds, 4),
        "seconds": round(time.perf_counter() - training_started, 4),
        "history": history, "device": "cpu", "seed": seed,
    }
    if evaluate_test:
        with torch.no_grad():
            logits = model(graph.x, edge_index)
            result["best_test_score"] = _accuracy(logits, graph.y, graph.test_mask)
    if method == "hgnn":
        result["hypergraph_construction"] = {
            "view": "structural", "hop_k": int(settings.get("hop_k", 2)),
            "structural_cap": int(settings.get("structural_cap", 32)),
            "normalization": "Dv^-1/2 H De^-1 H^T Dv^-1/2",
            "structure_filter": False,
        }
    return result
