"""Download official sources and prepare full-size, common benchmark inputs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import yaml
from utils.data_node import load_node_dataset, _normalize_features, _finalize_adj, _build_masks

DATASETS = ['actor', 'amazon_ratings', 'coauthor_cs', 'coauthor_physics', 'dblp',
            'ogbn_arxiv', 'wikics', 'questions', 'tolokers']
RAW = ROOT / 'data/raw/full_suite'
PREP = HERE / 'data'

def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    print('Downloading ' + url, flush=True)
    part = target.with_suffix(target.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'PRISM-research-benchmark/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response, part.open('wb') as output:
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
    part.replace(target)
    return target

def fetch_special(name):
    if name == 'wikics':
        return download('https://raw.githubusercontent.com/pmernyei/wiki-cs-dataset/master/dataset/data.json', RAW / 'wikics/data.json')
    if name == 'dblp':
        return download('https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1', RAW / 'dblp.zip')
    return download('https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip', RAW / 'arxiv.zip')

def indices_masks(splits, n):
    return {k: np.isin(np.arange(n), np.asarray(v).reshape(-1)) for k, v in splits.items()}

def save(name, x, y, adj, masks, meta):
    PREP.mkdir(parents=True, exist_ok=True)
    x = _normalize_features(x)
    adj = _finalize_adj(adj, undirected=True)
    np.savez_compressed(PREP / (name + '.npz'), x=x, y=y, indptr=adj.indptr, indices=adj.indices,
                        **masks)
    meta.update(name=name, nodes=len(y), features=x.shape[1], classes=int(y.max())+1,
                labeled_nodes=int((y >= 0).sum()), adjacency_entries=adj.nnz,
                normalization='row L2; all edges undirected; no node subsampling or feature projection',
                sha256=hashlib.sha256((PREP / (name + '.npz')).read_bytes()).hexdigest())
    (PREP / (name + '.json')).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False), flush=True)

def prepare(name):
    if (PREP / (name + '.npz')).exists():
        print('Prepared input exists: ' + name, flush=True)
        return
    if name not in {'dblp', 'wikics', 'ogbn_arxiv'}:
        cfg = yaml.safe_load((ROOT / 'configs/node' / (name + '.yaml')).read_text(encoding='utf-8'))['dataset']
        cfg.pop('max_nodes', None)
        cfg.pop('feature_projection_dim', None)
        graph = load_node_dataset(cfg, 0)
        masks = {}
        split_ids = list(range(10))
        # Coauthor has no official split; ten independent class-stratified splits.
        for split in split_ids:
            g = graph if split == 0 else load_node_dataset(cfg, split)
            for key in ('train_mask', 'val_mask', 'test_mask'):
                masks.setdefault(key, []).append(getattr(g, key).numpy())
        masks = {k: np.stack(v) for k, v in masks.items()}
        save(name, graph.x.numpy(), graph.y.numpy(), graph.adj, masks,
             {'split_policy': '20 train labels per class; 1000 validation; 1000 test; 10 seeds 0-9' if 'coauthor' in name else 'official splits 0-9',
              'metric': 'auroc' if name in ('questions', 'tolokers') else 'accuracy',
              'input_config': cfg})
    elif name == 'wikics':
        path = fetch_special(name)
        raw = json.loads(path.read_text(encoding='utf-8'))
        x, y = np.asarray(raw['features'], dtype=np.float32), np.asarray(raw['labels'], dtype=np.int64)
        rows = np.repeat(np.arange(len(y)), [len(v) for v in raw['links']])
        cols = np.concatenate([np.asarray(v, dtype=np.int64) for v in raw['links']])
        adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(y), len(y)))
        masks = {k: np.asarray(raw[v], dtype=bool) for k, v in [('train_mask','train_masks'), ('val_mask','val_masks'), ('stopping_mask','stopping_masks')]}
        masks['test_mask'] = np.repeat(np.asarray(raw['test_mask'], dtype=bool)[None], 20, axis=0)
        save(name, x, y, adj, masks, {'metric':'accuracy', 'split_policy':'all 20 official splits; val selects configurations, stopping selects checkpoints; shared test', 'source_url':str(path)})
    elif name == 'dblp':
        path = fetch_special(name)
        folder = RAW / 'dblp'
        if not (folder / 'adjM.npz').exists():
            folder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as z:
                # Extract only known raw basenames, never execute downloaded code.
                for info in z.infolist():
                    base = Path(info.filename).name
                    if base in {'adjM.npz','features_0.npz','features_1.npz','features_2.npy','labels.npy','node_types.npy','train_val_test_idx.npz'}:
                        (folder / base).write_bytes(z.read(info))
        adj = sp.load_npz(folder / 'adjM.npz')
        types = np.load(folder / 'node_types.npy')
        arrays = [sp.load_npz(folder / 'features_0.npz').toarray(), sp.load_npz(folder / 'features_1.npz').toarray(), np.load(folder / 'features_2.npy')]
        dims = [a.shape[1] for a in arrays]
        x = np.zeros((len(types), sum(dims) + 4), dtype=np.float32)
        offset = 0
        for t, array in enumerate(arrays):
            x[types == t, offset:offset + dims[t]] = array
            offset += dims[t]
        x[np.arange(len(types)), sum(dims) + types] = 1
        y = np.full(len(types), -1, dtype=np.int64)
        y[types == 0] = np.load(folder / 'labels.npy')
        raw = np.load(folder / 'train_val_test_idx.npz')
        masks = {k + '_mask':v[None] for k,v in indices_masks({k:raw[k + '_idx'] for k in ('train','val','test')}, len(y)).items()}
        save(name, x, y, adj, masks, {'metric':'accuracy', 'split_policy':'official MAGNN/PyG author split; 10 model initializations',
             'conversion':'all 26128 nodes and all relation edges retained; disjoint native feature blocks 334+4231+50 plus 4 type indicators; only 4057 authors labeled',
             'note':'This common homogeneous conversion is a local protocol, not directly comparable to published heterogeneous DBLP scores.'})
    else:
        path = fetch_special(name)
        folder = RAW / 'arxiv'
        if not (folder / 'raw/node-feat.csv.gz').exists():
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    parts = Path(info.filename).parts
                    if info.is_dir() or '..' in parts:
                        continue
                    target = RAW.joinpath(*parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as source, target.open('wb') as output:
                        import shutil
                        shutil.copyfileobj(source, output)
        x = pd.read_csv(folder/'raw/node-feat.csv.gz', header=None).to_numpy(dtype=np.float32)
        y = pd.read_csv(folder/'raw/node-label.csv.gz', header=None).to_numpy(dtype=np.int64).reshape(-1)
        edges = pd.read_csv(folder/'raw/edge.csv.gz', header=None).to_numpy(dtype=np.int64)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:,0],edges[:,1])), shape=(len(y),len(y)))
        splits = {k:pd.read_csv(folder/'split/time'/v,header=None).to_numpy(dtype=np.int64).reshape(-1) for k,v in [('train','train.csv.gz'),('val','valid.csv.gz'),('test','test.csv.gz')]}
        masks = {k+'_mask':v[None] for k,v in indices_masks(splits,len(y)).items()}
        save(name,x,y,adj,masks,{'metric':'accuracy','split_policy':'official OGB time split; 10 model initializations','source':'official OGB arxiv.zip'})

if __name__ == '__main__':
    requested = sys.argv[1:] or DATASETS
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {name:pool.submit(fetch_special,name) for name in requested if name in ('dblp','wikics','ogbn_arxiv')}
        for name, future in futures.items():
            try:
                future.result()
            except Exception as error:
                print(f'DOWNLOAD ERROR {name}: {error}', flush=True)
    for name in requested:
        prepare(name)
