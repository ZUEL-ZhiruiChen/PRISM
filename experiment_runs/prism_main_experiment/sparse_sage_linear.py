"""Equivalent GraphSAGE mean aggregation without an E x input_dim tensor.

This separately frozen execution adaptation replaces only GraphSAGE's message
materialization. It retains the two native PyG SAGEConv parameter sets, input
and hidden dropout, root contributions, biases and mean aggregation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import zipfile
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import suite

HERE=Path(__file__).resolve().parent
OUT=HERE/'results_sparse_sage_linear'
ORIGINAL_MAKE_MODEL=suite.make_model

class SparseMeanSAGE(suite.StandardBaseline):
    def __init__(self,inputs,spec):
        super().__init__(inputs.x.shape[1],spec['hidden_dim'],int(inputs.y.max())+1,'sage',spec['dropout'],{})
        # Keep row summation followed by division, as in the native mean op.
        self.mean_adj=suite.tensor_sparse(inputs.adj).coalesce()
        self.degree=torch.from_numpy(np.maximum(1.,np.asarray(inputs.adj.sum(1),dtype=np.float32).reshape(-1))).to(suite.DEVICE).unsqueeze(-1)
    def layer(self,layer,x):
        # Linearity permits the feature projection before aggregation. The
        # bias is added afterwards, including for zero-degree nodes.
        projected=F.linear(x,layer.lin_l.weight,None)
        neighbors=torch.sparse.mm(self.mean_adj,projected)/self.degree
        if layer.lin_l.bias is not None:neighbors=neighbors+layer.lin_l.bias
        return neighbors+layer.lin_r(x)
    def forward(self,x,edge_index=None):
        x=F.dropout(x,p=self.dropout,training=self.training)
        h=F.relu(self.layer(self.first,x))
        h=F.dropout(h,p=self.dropout,training=self.training)
        return self.layer(self.second,h)

def make_model(inputs,method,spec):
    if method!='sage':return ORIGINAL_MAKE_MODEL(inputs,method,spec)
    return SparseMeanSAGE(inputs,spec).to(suite.DEVICE),None

def check():
    class Input:pass
    data=Input();rng=np.random.default_rng(17)
    data.x=torch.from_numpy(rng.normal(size=(61,23)).astype(np.float32))
    data.y=torch.arange(61)%3
    a=sp.random(61,61,density=.12,random_state=17,format='csr');a=a.maximum(a.T);a.setdiag(0);a.eliminate_zeros();a.data[:]=1.
    data.adj=a
    coo=a.tocoo();edges=torch.from_numpy(np.vstack([coo.row,coo.col]).astype(np.int64)).to(suite.DEVICE)
    x=data.x.to(suite.DEVICE)
    spec={'hidden_dim':32,'dropout':.35}
    suite.seed_all(41)
    native=suite.StandardBaseline(23,32,3,'sage',.35,{}).to(suite.DEVICE)
    equivalent=SparseMeanSAGE(data,spec).to(suite.DEVICE)
    equivalent.load_state_dict(native.state_dict())
    for training in [False,True]:
        native.train(training);equivalent.train(training)
        suite.seed_all(55);z=native(x,edges)
        suite.seed_all(55);q=equivalent(x)
        assert torch.allclose(z,q,atol=3e-6,rtol=3e-5),(z-q).abs().max()
        z.square().sum().backward();q.square().sum().backward()
        for (name,p),(other,v) in zip(native.named_parameters(),equivalent.named_parameters()):
            assert name==other and torch.allclose(p.grad,v.grad,atol=5e-5,rtol=1e-4),name
        native.zero_grad();equivalent.zero_grad()
    print('PASS: native and sparse GraphSAGE logits, dropout and parameter gradients agree',flush=True)

def protocol():
    parent=suite.protocol()
    hashes={**parent['source_hashes'],str(Path(__file__).resolve().relative_to(suite.ROOT)):suite.digest(Path(__file__))}
    payload={**{k:v for k,v in parent.items() if k!='frozen_at_utc'},
             'version':'sparse-mean-sage-linear-v2','methods':['sage'],
             'parent_source_signature':parent['source_signature'],'source_hashes':hashes,
             'source_signature':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
             'adaptation':'same PyG GraphSAGE weights, dropout, roots and mean aggregation; linearity allows A @ (X @ W) / degree with bias added after aggregation, including isolated nodes'}
    path=OUT/'protocol.json'
    if path.exists():
        assert suite.read(path)==payload
    else:
        suite.write(payload,path)
        with zipfile.ZipFile(OUT/'frozen_sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in hashes:archive.write(suite.ROOT/name,name)
    return payload

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--datasets',nargs='+',default=suite.DATASETS);args=ap.parse_args()
    if args.check:check()
    else:
        p=protocol();suite.OUT=OUT;suite.make_model=make_model
        for name in args.datasets:
            inputs=suite.Inputs(name)
            suite.run_method(inputs,'sage',p)
            del inputs
            torch.cuda.empty_cache()
