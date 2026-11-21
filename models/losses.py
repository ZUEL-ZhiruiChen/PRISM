from __future__ import annotations

from itertools import combinations
import math

import torch
import torch.nn.functional as F


def bootstrap_alignment(z_list, p_list):
    if len(z_list) < 2:
        return z_list[0].new_tensor(0.0)
    losses = []
    for i, j in combinations(range(len(z_list)), 2):
        pi = F.normalize(p_list[i], dim=-1)
        pj = F.normalize(p_list[j], dim=-1)
        zi = F.normalize(z_list[i].detach(), dim=-1)
        zj = F.normalize(z_list[j].detach(), dim=-1)
        losses.append(2 - 2 * (pi * zj).sum(dim=-1).mean())
        losses.append(2 - 2 * (pj * zi).sum(dim=-1).mean())
    return sum(losses) / len(losses)


def dominant_view_distill(z_list, gate, temperature=1.0, hard=False):
    if len(z_list) < 2:
        return z_list[0].new_tensor(0.0)
    if hard:
        with torch.no_grad():
            dom = gate.argmax(dim=-1)
        losses = []
        for v, z in enumerate(z_list):
            teacher = torch.stack([z_list[int(d)][i] for i, d in enumerate(dom)], dim=0).detach()
            mask = (dom != v).float()
            if mask.sum() > 0:
                loss = 2 - 2 * (F.normalize(z, dim=-1) * F.normalize(teacher, dim=-1)).sum(dim=-1)
                losses.append((loss * mask).sum() / mask.sum())
        return sum(losses) / len(losses) if losses else gate.new_tensor(0.0)
    with torch.no_grad():
        weights = F.softmax(torch.log(gate.clamp_min(1e-8)) / temperature, dim=-1)
        teacher_sum = sum(weights[:, i : i + 1] * z_list[i].detach() for i in range(len(z_list)))
    losses = []
    for v, z in enumerate(z_list):
        other_weight = (1.0 - weights[:, v]).clamp_min(1e-6)
        teacher = (teacher_sum - weights[:, v : v + 1] * z_list[v].detach()) / other_weight.unsqueeze(-1)
        loss = 2 - 2 * (F.normalize(z, dim=-1) * F.normalize(teacher, dim=-1)).sum(dim=-1)
        losses.append((loss * other_weight).sum() / other_weight.sum().clamp_min(1e-6))
    return sum(losses) / len(losses)


def gated_leave_one_out_distill(shared_list, p_list, gate, temperature=1.0):
    """Align shared predictions to detached gate-weighted other-view targets.

    A teacher never contains the student's representation. We normalize only
    the other views directly, avoiding cancellation when one gate is near 1.
    Nodes with no positively weighted other view (e.g. a hard top-1 gate), or
    an exactly zero teacher, do not receive this auxiliary loss. Gate weights
    and teacher features receive no gradients from distillation. Classification
    still trains the gate and both representation branches.
    """
    if not shared_list or len(shared_list) != len(p_list):
        raise ValueError("shared_list and p_list must have equal nonzero lengths")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("distillation temperature must be finite and positive")
    if gate.ndim != 2 or gate.shape != (shared_list[0].shape[0], len(shared_list)):
        raise ValueError("gate must have shape [num_nodes, num_views]")
    if not torch.isfinite(gate).all() or (gate < 0).any():
        raise ValueError("gate weights must be finite and nonnegative")
    if len(shared_list) < 2:
        return p_list[0].sum() * 0.0
    shape = shared_list[0].shape
    if any(z.shape != shape for z in shared_list) or any(p.shape != shape for p in p_list):
        raise ValueError("all shared representations and predictions must have the same shape")
    losses = []
    eps = torch.finfo(gate.dtype).eps
    for student, prediction in enumerate(p_list):
        others = [idx for idx in range(len(shared_list)) if idx != student]
        with torch.no_grad():
            raw_weights = gate.detach()[:, others]
            valid = raw_weights.gt(0).any(dim=-1)
            safe_weights = torch.where(valid.unsqueeze(-1), raw_weights, torch.ones_like(raw_weights))
            # Log-space temperature scaling preserves exact zero weights.
            logits = safe_weights.clamp_min(torch.finfo(gate.dtype).tiny).log()
            logits = logits.masked_fill(safe_weights <= 0, float("-inf"))
            weights = F.softmax(logits / max(temperature, eps), dim=-1)
            representations = torch.stack([shared_list[idx].detach() for idx in others], dim=1)
            teacher = (weights.unsqueeze(-1) * representations).sum(dim=1)
            valid = valid & (teacher.norm(dim=-1) > eps)
        per_node = 2.0 - 2.0 * (
            F.normalize(prediction, dim=-1) * F.normalize(teacher, dim=-1)
        ).sum(dim=-1)
        losses.append((per_node * valid).sum() / valid.sum().clamp_min(1))
    return torch.stack(losses).mean()


def shared_private_decorrelation_loss(shared_list, private_list, eps=1e-6):
    """Penalize centered cross-correlation within each shared/private pair.

    Standardization prevents arbitrary feature scaling from reducing this
    penalty. Variance regularization and task supervision remain necessary:
    decorrelation alone does not identify semantic shared/private factors or
    prevent a constant branch.
    """
    if not shared_list:
        raise ValueError("shared_list must not be empty")
    if not private_list:
        return shared_list[0].sum() * 0.0
    if len(shared_list) != len(private_list):
        raise ValueError("shared_list and private_list must have equal lengths")
    losses = []
    for shared, private in zip(shared_list, private_list):
        if shared.ndim != 2 or private.ndim != 2 or shared.shape[0] != private.shape[0]:
            raise ValueError("each branch pair must be matrices with the same node count")
        if shared.shape[0] < 2:
            losses.append((shared.sum() + private.sum()) * 0.0)
            continue
        centered_shared = shared - shared.mean(dim=0, keepdim=True)
        centered_private = private - private.mean(dim=0, keepdim=True)
        shared_std = centered_shared.square().mean(dim=0, keepdim=True).clamp_min(eps ** 2).sqrt()
        private_std = centered_private.square().mean(dim=0, keepdim=True).clamp_min(eps ** 2).sqrt()
        shared_unit = centered_shared / shared_std
        private_unit = centered_private / private_std
        cross_correlation = shared_unit.T @ private_unit / shared.shape[0]
        losses.append(cross_correlation.square().mean())
    return torch.stack(losses).mean()


def variance_loss(z_list, gamma=1.0, eps=1e-4):
    if not z_list:
        return torch.tensor(0.0)
    vals = []
    for z in z_list:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        vals.append(F.relu(gamma - std).mean())
    return sum(vals) / len(vals)


def elastic_interval_loss(embedding, pair_index, lower, upper):
    if pair_index.numel() == 0:
        return embedding.new_tensor(0.0)
    src, dst = pair_index[0], pair_index[1]
    e1 = F.normalize(embedding[src], dim=-1)
    e2 = F.normalize(embedding[dst], dim=-1)
    d = torch.clamp(1.0 - (e1 * e2).sum(dim=-1), 0.0, 2.0)
    return (F.relu(lower - d) + F.relu(d - upper)).mean()


def discriminative_elastic_loss(embedding, pair_index, lower, upper, y, train_mask, neg_margin=0.8):
    if pair_index.numel() == 0:
        return embedding.new_tensor(0.0)
    src, dst = pair_index[0], pair_index[1]
    e1 = F.normalize(embedding[src], dim=-1)
    e2 = F.normalize(embedding[dst], dim=-1)
    d = torch.clamp(1.0 - (e1 * e2).sum(dim=-1), 0.0, 2.0)
    pos = (F.relu(lower - d) + F.relu(d - upper)).mean()
    mask = train_mask[src] & train_mask[dst] & (y[src] != y[dst])
    neg = F.relu(neg_margin - d[mask]).mean() if mask.any() else embedding.new_tensor(0.0)
    return pos + neg


def load_balance_loss(gate):
    target = torch.full((gate.size(1),), 1.0 / gate.size(1), device=gate.device)
    avg = gate.mean(dim=0)
    return ((avg - target) ** 2).mean()
