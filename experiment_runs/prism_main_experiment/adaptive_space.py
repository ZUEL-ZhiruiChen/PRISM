"""Deterministic population search over PRISM hyperparameters; no data or labels."""
from __future__ import annotations
import copy
import hashlib
import json
import random

GRAPH_PROFILES = [
    dict(hop_k=2, structural_cap=32, structural_candidate_cap=32, knn_k=12,
         n_clusters=64, densify_k=1, densify_min_similarity=.3, hyperedge_keep_ratio=.95),
    dict(hop_k=2, structural_cap=48, structural_candidate_cap=48, knn_k=16,
         n_clusters=96, densify_k=2, densify_min_similarity=.2, hyperedge_keep_ratio=.85),
    dict(hop_k=1, structural_cap=16, structural_candidate_cap=24, knn_k=8,
         n_clusters=32, densify_k=1, densify_min_similarity=.5, hyperedge_keep_ratio=.7),
    dict(hop_k=2, structural_cap=32, structural_candidate_cap=32, knn_k=24,
         n_clusters=128, densify_k=4, densify_min_similarity=.2, hyperedge_keep_ratio=.8),
]
CHOICES = {
    'model.hidden_dim':[64,128,256,512], 'model.proj_dim':[32,64,128],
    'model.K':[1,2,3,4], 'model.dropout':[.15,.3,.4,.5,.6,.7],
    'model.gate_temperature':[.7,1.,1.5,2.,3.],
    'model.gate_residual':[0.,.05,.15,.3],
    'model.private_gate_strength':[0.,.5,1.],
    'model.gate_mode':['soft','reliability'],
    'loss.lambda_align':[.003,.01,.03,.1,.2],
    'loss.lambda_var':[.001,.005,.02,.05],
    'loss.lambda_decorrelation':[.001,.005,.02,.05],
    'loss.lambda_view_ce':[0.,.05,.2,.5],
    'loss.distill_warmup_epochs':[5,15,25],
    'loss.distill_temperature':[.7,1.,1.5,2.],
    'loss.label_smoothing':[0.,.05,.1],
    'optim.lr':[.0003,.0007,.001,.002,.003,.005,.01],
    'optim.weight_decay':[.00001,.0001,.0005,.002,.01],
}
PRIORS = [dict(structural=.34, feature_knn=.33, prototype=.33),
          dict(structural=.7, feature_knn=.2, prototype=.1),
          dict(structural=.1, feature_knn=.7, prototype=.2),
          dict(structural=.15, feature_knn=.35, prototype=.5)]

def normalize(config):
    cfg = copy.deepcopy(config)
    m, loss = cfg['model'], cfg['loss']
    m.update(model_type='prism',dual_branch=True,structure_filter=True,
             enabled_views=['structural','feature_knn','prototype'],fusion_mode='embedding')
    m.setdefault('gate_mode','soft')
    m.setdefault('refine',False)
    loss.update(lambda_elastic=0.,distill_enabled=True)
    loss.setdefault('lambda_view_ce',0.)
    loss.setdefault('label_smoothing',0.)
    if m['gate_mode'] == 'reliability':
        loss['lambda_view_ce'] = max(.05,loss['lambda_view_ce'])
    m['proj_dim'] = min(m['proj_dim'],m['hidden_dim'])
    m['preserve_node_coverage'] = True
    cfg.setdefault('train',{}).update(epochs=400,patience=60,selection='val')
    assert m['hidden_dim'] > 0 and m['proj_dim'] > 0 and 1 <= m['K'] <= 4
    assert 0 <= m['dropout'] < 1 and m['gate_temperature'] > 0
    assert 0 <= m['private_gate_strength'] <= 1 and 0 < m['knn_k'] <= 32
    assert 0 < m['hyperedge_keep_ratio'] <= 1
    return cfg

def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]

def mutate(config, rng, global_search=False):
    cfg = copy.deepcopy(config)
    keys = list(CHOICES)
    selected = keys if global_search else rng.sample(keys,rng.randint(1,4))
    for key in selected:
        section,name = key.split('.')
        cfg[section][name] = rng.choice(CHOICES[key])
    if global_search or rng.random() < .3:
        cfg['model'].update(copy.deepcopy(rng.choice(GRAPH_PROFILES)))
    if global_search or rng.random() < .25:
        cfg['model']['view_gate_prior'] = copy.deepcopy(rng.choice(PRIORS))
    return normalize(cfg)

def candidates(dataset, round_index, parents, seen, batch_size=8):
    """Two global proposals per batch keep search from using only local changes."""
    assert parents and round_index >= 0
    seed = int(hashlib.sha256(f'{dataset}:{round_index}:prism-adaptive-v1'.encode()).hexdigest()[:12],16)
    rng = random.Random(seed)
    result,used = [],set(seen)
    if round_index == 0:
        for parent in parents:
            cfg = normalize(parent)
            key = fingerprint(cfg)
            if key not in used:
                used.add(key)
                result.append(cfg)
            if len(result) >= batch_size:
                return result
    for attempt in range(10000):
        cfg = mutate(rng.choice(parents[:3]),rng,
                     global_search=len(result)%4 == 3 or attempt%13 == 12)
        key = fingerprint(cfg)
        if key in used:
            continue
        used.add(key)
        result.append(cfg)
        if len(result) >= batch_size:
            break
    assert len(result) == batch_size, 'Search space exhausted'
    return result

def update_plateau(previous_best, new_best, stale_rounds, min_delta=.0005):
    gain = new_best-previous_best
    return 0 if gain >= min_delta else stale_rounds+1
