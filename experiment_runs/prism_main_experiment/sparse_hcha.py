"""Memory-efficient, algebraically equivalent PyG HCHA for OGBN-Arxiv.

Attention logits are sums of node and edge scalar projections. Weighted
incidence propagation uses sparse mm; edge-weight gradients use bounded
chunks instead of allocating membership_count by hidden_dim tensors.
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
from torch_geometric.utils import softmax
import suite
HERE=Path(__file__).resolve().parent
OUT=HERE/'results_sparse_hcha'
ORIGINAL_MAKE_MODEL=suite.make_model

class WeightedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,indices,values,x,rows,cols,transpose_order):
        ctx.save_for_backward(indices,values,x,transpose_order)
        ctx.shape=(rows,cols)
        matrix=torch.sparse_coo_tensor(indices,values,(rows,cols),is_coalesced=True)
        return torch.sparse.mm(matrix,x)
    @staticmethod
    def backward(ctx,grad):
        indices,values,x,order=ctx.saved_tensors
        rows,cols=ctx.shape
        grad_values=None
        if ctx.needs_input_grad[1]:
            grad_values=torch.empty_like(values)
            for start in range(0,len(values),16384):
                idx=indices[:,start:start+16384]
                grad_values[start:start+16384]=(grad[idx[0]]*x[idx[1]]).sum(-1)
        grad_x=None
        if ctx.needs_input_grad[2]:
            transposed=torch.sparse_coo_tensor(indices.flip(0)[:,order],values[order],(cols,rows),is_coalesced=True)
            grad_x=torch.sparse.mm(transposed,grad)
        return None,grad_values,grad_x,None,None,None

def efficient_layer(layer,x,attrs,operator,training):
    assert layer.heads==1 and layer.use_attention and layer.attention_mode=='node'
    indices=operator['h'].indices()
    n,m=operator['h'].shape
    if 'transpose_order' not in operator:
        operator['transpose_order']=torch.argsort(indices[1]*n+indices[0],stable=True)
        operator['reverse_order']=torch.argsort(operator['transpose_order'])
    order=operator['transpose_order'];reverse=operator['reverse_order']
    xp=layer.lin(x);ep=layer.lin(attrs)
    left,right=layer.att[0,0,:layer.out_channels],layer.att[0,0,layer.out_channels:]
    alpha=(xp@left)[indices[0]]+(ep@right)[indices[1]]
    alpha=F.leaky_relu(alpha,layer.negative_slope)
    alpha=softmax(alpha,indices[1],num_nodes=m)
    alpha=F.dropout(alpha,p=layer.dropout,training=training)
    edge=WeightedMM.apply(indices.flip(0)[:,order],alpha[order],xp,m,n,reverse)
    edge=edge*operator['de_inv'].unsqueeze(-1)
    result=WeightedMM.apply(indices,alpha,edge,n,m,order)
    result=result*operator['dv_inv_sqrt'].square().unsqueeze(-1)
    if layer.bias is not None:result=result+layer.bias
    return result

class SparseHCHA(suite.HCHA):
    def forward(self,x,edges=None):
        if self.attr0 is None:self.attr0=self.attrs(x).detach()
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=F.elu(efficient_layer(self.first,x,self.attr0,self.operator,self.training))
        x=F.dropout(x,p=self.dropout,training=self.training)
        return efficient_layer(self.second,x,self.attrs(x),self.operator,self.training)

def make_model(inputs,method,spec):
    if method!='hcha':return ORIGINAL_MAKE_MODEL(inputs,method,spec)
    context=suite.make_context(inputs,method,spec)
    model=SparseHCHA(inputs.x.shape[1],spec['hidden_dim'],int(inputs.y.max())+1,spec['dropout'],context)
    return model.to(suite.DEVICE),context

def check():
    suite.seed_all(51)
    n,m,fin,hidden,classes=71,43,17,32,3
    rng=np.random.default_rng(51)
    h=sp.random(n,m,density=.14,format='csr',random_state=51,dtype=np.float32);h.data[:]=1.
    dv=np.asarray(h.sum(1)).reshape(-1);de=np.asarray(h.sum(0)).reshape(-1)
    op={'h':suite.tensor_sparse(h),'ht':suite.tensor_sparse(h.T.tocsr()),
        'dv_inv_sqrt':torch.tensor(1/np.sqrt(np.maximum(dv,1)),device=suite.DEVICE),
        'de_inv':torch.tensor(1/np.maximum(de,1),device=suite.DEVICE)}
    x=torch.tensor(rng.normal(size=(n,fin)).astype(np.float32),device=suite.DEVICE)
    a=suite.HCHA(fin,hidden,classes,.25,op).to(suite.DEVICE)
    b=SparseHCHA(fin,hidden,classes,.25,op).to(suite.DEVICE);b.load_state_dict(a.state_dict())
    for training in [False,True]:
        a.train(training);b.train(training)
        suite.seed_all(67);za=a(x)
        suite.seed_all(67);zb=b(x)
        assert torch.allclose(za,zb,atol=2e-6,rtol=2e-4),(za-zb).abs().max()
        za.square().sum().backward();zb.square().sum().backward()
        for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
            assert name==other and torch.allclose(p.grad,q.grad,atol=5e-5,rtol=2e-3),(name,(p.grad-q.grad).abs().max())
        a.zero_grad();b.zero_grad()
    print('PASS: native PyG HCHA and sparse HCHA logits/dropout/parameter gradients agree',flush=True)

def protocol():
    parent=suite.protocol()
    hashes={**parent['source_hashes'],str(Path(__file__).resolve().relative_to(suite.ROOT)):suite.digest(Path(__file__))}
    payload={**{k:v for k,v in parent.items() if k!='frozen_at_utc'},'version':'sparse-hcha-arxiv-v1',
             'methods':['hcha'],'datasets':['ogbn_arxiv'],'source_hashes':hashes,
             'parent_source_signature':parent['source_signature'],
             'source_signature':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
             'adaptation':'same PyG HCHA scalar attention, node-mode softmax, dropout, normalizations and biases; sparse incidence propagation with chunked weight gradients'}
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
        suite.run_method(suite.Inputs('ogbn_arxiv'),'hcha',p)
