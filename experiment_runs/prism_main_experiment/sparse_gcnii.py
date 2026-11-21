"""Equivalent GCNII sparse propagation for full-size OGBN-Arxiv.

Retains the native PyG GCN2Conv parameters and update operations. Only its
weighted edge-message sum is evaluated as a normalized sparse matrix product.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import suite

HERE=Path(__file__).resolve().parent
OUT=HERE/'results_sparse_gcnii'
ORIGINAL_MAKE_MODEL=suite.make_model

class SparseGCNII(suite.GCNII):
    def __init__(self,fin,hidden,classes,dropout,operator):
        super().__init__(fin,hidden,classes,dropout)
        self.operator=operator
    def forward(self,x,edges=None):
        x=F.dropout(x,p=self.dropout,training=self.training)
        x0=F.relu(self.first(x));x=x0
        for layer in self.layers:
            x=F.dropout(x,p=self.dropout,training=self.training)
            propagated=torch.sparse.mm(self.operator,x)
            propagated.mul_(1-layer.alpha)
            out=propagated.add_(layer.alpha*x0)
            x=F.relu(torch.addmm(out,out,layer.weight1,beta=1-layer.beta,alpha=layer.beta))
        return self.second(F.dropout(x,p=self.dropout,training=self.training))

def make_model(inputs,method,spec):
    if method!='gcnii':return ORIGINAL_MAKE_MODEL(inputs,method,spec)
    model=SparseGCNII(inputs.x.shape[1],spec['hidden_dim'],int(inputs.y.max())+1,
                      spec['dropout'],inputs.normalization())
    return model.to(suite.DEVICE),None

def check():
    suite.seed_all(51)
    rng=np.random.default_rng(51)
    n,fin,hidden,classes=71,17,32,3
    a=sp.random(n,n,density=.1,format='csr',random_state=51,dtype=np.float32)
    a=a.maximum(a.T);a.setdiag(0);a.eliminate_zeros();a.data[:]=1.
    # Include an isolated node to exercise self-loop normalization.
    a=a.tolil();a[-1,:]=0;a[:,-1]=0;a=a.tocsr()
    coo=a.tocoo();edges=torch.from_numpy(np.vstack([coo.row,coo.col]).astype(np.int64)).to(suite.DEVICE)
    ei,ew=gcn_norm(edges,num_nodes=n,add_self_loops=True,dtype=torch.float32)
    operator=torch.sparse_coo_tensor(ei.flip(0),ew,(n,n)).coalesce()
    x=torch.tensor(rng.normal(size=(n,fin)).astype(np.float32),device=suite.DEVICE,requires_grad=True)
    a=suite.GCNII(fin,hidden,classes,.35).to(suite.DEVICE)
    b=SparseGCNII(fin,hidden,classes,.35,operator).to(suite.DEVICE)
    b.load_state_dict(a.state_dict())
    for training in [False,True]:
        a.train(training);b.train(training)
        xa=x.detach().clone().requires_grad_(True);xb=x.detach().clone().requires_grad_(True)
        suite.seed_all(67);za=a(xa,edges)
        suite.seed_all(67);zb=b(xb)
        assert torch.allclose(za,zb,atol=3e-6,rtol=3e-5),(za-zb).abs().max()
        za.square().sum().backward();zb.square().sum().backward()
        assert torch.allclose(xa.grad,xb.grad,atol=5e-5,rtol=2e-3)
        for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
            assert name==other and torch.allclose(p.grad,q.grad,atol=5e-5,rtol=2e-3),(name,(p.grad-q.grad).abs().max())
        a.zero_grad();b.zero_grad()
    print('PASS: native PyG GCNII and sparse GCNII logits/dropout/input and parameter gradients agree',flush=True)

def protocol():
    parent=suite.protocol()
    hashes={**parent['source_hashes'],str(Path(__file__).resolve().relative_to(suite.ROOT)):suite.digest(Path(__file__))}
    payload={**{k:v for k,v in parent.items() if k!='frozen_at_utc'},'version':'sparse-gcnii-arxiv-v1',
             'methods':['gcnii'],'datasets':['ogbn_arxiv'],'source_hashes':hashes,
             'parent_source_signature':parent['source_signature'],
             'source_signature':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
             'adaptation':'same 16 PyG GCN2Conv layers, weights, dropout, normalization, alpha/beta and update operations; normalized sparse multiplication replaces explicit weighted message materialization'}
    path=OUT/'protocol.json'
    if path.exists():assert suite.read(path)==payload
    else:
        suite.write(payload,path)
        with zipfile.ZipFile(OUT/'frozen_sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in hashes:archive.write(suite.ROOT/name,name)
    return payload

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    if args.check:check()
    else:
        p=protocol();suite.OUT=OUT;suite.make_model=make_model
        suite.run_method(suite.Inputs('ogbn_arxiv'),'gcnii',p)
