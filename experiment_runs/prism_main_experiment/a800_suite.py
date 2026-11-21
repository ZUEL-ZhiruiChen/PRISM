"""Full-size benchmark, validation-only tuning and independently recorded tests.

Core repository files are read unchanged. Local PyG baseline implementations
and the CUDA adaptation live here, separate from the earlier frozen CPU study.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from statistics import mean, stdev
import sys
import time
import zipfile

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiment_runs/full_benchmark'))
os.chdir(ROOT)
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from threadpoolctl import threadpool_limits
from torch_geometric.nn import GCN2Conv, HypergraphConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import run_node
from models.hypergraph import MultiViewHypergraphBuilder
from models.model import PRISM, MVHGCL
from models.losses import shared_private_decorrelation_loss
from baselines import StandardBaseline
from prepare_data import DATASETS, PREP
from utils.data_node import GraphData

METHODS = ['mlp','gcn','gat','sage','appnp','gprgnn','gcnii','hgnn','hcha','legacy_ce','legacy_full','prism']
LABELS = {'mlp':'MLP','gcn':'GCN','gat':'GAT','sage':'GraphSAGE','appnp':'APPNP',
          'gprgnn':'GPR-GNN','gcnii':'GCNII','hgnn':'HGNN','hcha':'HCHA',
          'legacy_ce':'原始框架（仅CE）','legacy_full':'原始完整框架','prism':'PRISM'}
OUT = HERE / 'results'
CACHE = HERE / 'cache'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS, PATIENCE = 200, 40
CONSTRUCTION_SEED = 0
VERSION = 'full-size-nine-datasets-v2'
EPOCH_CALLBACK = None
SOURCES = ['run_node.py','models/model.py','models/losses.py','models/hypergraph.py',
           'utils/data_node.py','experiment_runs/full_benchmark/baselines.py',
           str(Path(__file__).resolve().relative_to(ROOT)),
           str((HERE/'prepare_data.py').relative_to(ROOT)),
           str((HERE/'check_protocol.py').relative_to(ROOT))]
SOURCES += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'configs/node').glob('*.yaml'))]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()

def write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    # All neighbor similarities and network matrix products use float32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def tensor_sparse(matrix):
    return run_node.scipy_to_torch_sparse(matrix).to(DEVICE)

def input_fingerprint(graph):
    def hasharray(value):
        return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
    return {'nodes':len(graph.y),'features':graph.num_features,'classes':graph.num_classes,
            'adjacency_entries':graph.adj.nnz,'x':hasharray(graph.x.numpy()),'y':hasharray(graph.y.numpy()),
            'adj':hasharray(graph.adj.indptr)+hasharray(graph.adj.indices),
            'masks':{key:hasharray(getattr(graph,key).numpy()) for key in ['train_mask','val_mask','test_mask']},
            'counts':{key:int(getattr(graph,key).sum()) for key in ['train_mask','val_mask','test_mask']}}

class Inputs:
    def __init__(self,name):
        self.name = name
        self.raw = np.load(PREP/(name+'.npz'),allow_pickle=False)
        self.meta = read(PREP/(name+'.json'))
        self.x = torch.from_numpy(self.raw['x'])
        self.y = torch.from_numpy(self.raw['y'])
        n = len(self.y)
        idx,ptr = self.raw['indices'],self.raw['indptr']
        self.adj = sp.csr_matrix((np.ones(len(idx),dtype=np.float32),idx,ptr),shape=(n,n))
        coo = self.adj.tocoo()
        self.edges = torch.from_numpy(np.vstack([coo.row,coo.col]).astype(np.int64)).to(DEVICE)
        self.x_gpu,self.y_gpu = self.x.to(DEVICE),self.y.to(DEVICE)
        self.num_splits = self.raw['train_mask'].shape[0]
        self.contexts = {}
        self.norm = None

    def graph(self, split):
        masks = [torch.from_numpy(self.raw[key][split].copy()).bool() for key in ['train_mask','val_mask','test_mask']]
        graph = GraphData(self.name,self.x,self.y,self.adj,*masks,int(self.y.max())+1,self.x.shape[1],[],split)
        for mask in masks:
            assert bool(mask.any()) and not bool((mask & (self.y < 0)).any())
        assert not any(bool((masks[i]&masks[j]).any()) for i,j in [(0,1),(0,2),(1,2)])
        return graph

    def mask(self,key,split):
        return torch.from_numpy(self.raw[key][split].copy()).to(DEVICE)

    def normalization(self):
        if self.norm is None:
            ei,ew = gcn_norm(self.edges,num_nodes=len(self.y),add_self_loops=True,dtype=torch.float32)
            self.norm = torch.sparse_coo_tensor(ei.flip(0),ew,(len(self.y),len(self.y))).coalesce()
        return self.norm

class FastBuilder(MultiViewHypergraphBuilder):
    """Original construction, with a blockwise exact GPU cosine neighbor search.

    Like sklearn kneighbors(X=None), self is excluded and k+1 neighbors are
    returned. Ties have a deterministic ascending-node convention in this suite.
    No N by N similarity matrix is retained.
    """
    def _knn_hyperedges(self,x):
        CACHE.mkdir(parents=True,exist_ok=True)
        key = hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
        path = CACHE/('knn_'+key+'.npz')
        count = min(33,len(x)-1)
        if path.exists():
            raw = np.load(path)
            idx,vals = raw['idx'],raw['sims']
        else:
            started = time.perf_counter()
            xp = F.normalize(torch.from_numpy(np.asarray(x)).to(DEVICE),dim=-1)
            ids,values = [],[]
            with torch.no_grad():
                for start in range(0,len(x),512):
                    sims = xp[start:start+512] @ xp.T
                    local = torch.arange(sims.shape[0],device=DEVICE)
                    sims[local,start+local] = -torch.inf
                    # Stable sorting is used only for exact ties among selected
                    # candidates; topk candidate ties at the cutoff are checked.
                    v,i = sims.topk(count,dim=1,largest=True,sorted=True)
                    # Resolve cutoff ties by explicit stable order on that row.
                    boundary = v[:,-1:]
                    tied = (sims == boundary).sum(1) > (v == boundary).sum(1)
                    for row in tied.nonzero().flatten().tolist():
                        sel = torch.argsort(sims[row],descending=True,stable=True)[:count]
                        i[row],v[row] = sel,sims[row,sel]
                    ni,nv = i.cpu().numpy(),v.cpu().numpy()
                    for row in range(len(ni)):
                        order = np.lexsort((ni[row],-nv[row]))
                        ni[row],nv[row] = ni[row,order],nv[row,order]
                    ids.append(ni); values.append(nv)
            idx,vals = np.concatenate(ids),np.concatenate(values)
            np.savez_compressed(path,idx=idx,sims=vals)
            del xp
            print(json.dumps({'stage':'exact_knn','nodes':len(x),'seconds':time.perf_counter()-started}),flush=True)
        count = min(self.knn_k+1,len(x)-1)
        idx,vals = idx[:,:count],vals[:,:count]
        hes = [[(int(j),float(max(1e-3,v)) if self.soft_membership else 1.) for j,v in zip(js,vs)] for js,vs in zip(idx,vals)]
        return hes,idx

    def build(self,x,adj,**kwargs):
        CACHE.mkdir(parents=True,exist_ok=True)
        keydata = {'impl':digest(Path(__file__)),'base':digest(ROOT/'models/hypergraph.py'),
                   'kwargs':kwargs,'settings':self.__dict__,
                   'x':hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest(),
                   'adj':hashlib.sha256(adj.indptr.tobytes()+adj.indices.tobytes()).hexdigest()}
        path = CACHE/(hashlib.sha256(json.dumps(keydata,sort_keys=True).encode()).hexdigest()+'.pkl')
        if path.exists():
            with path.open('rb') as stream:
                return pickle.load(stream)
        print(json.dumps({'stage':'hypergraph_build','nodes':len(x),'settings':self.__dict__}),flush=True)
        with threadpool_limits(limits=1):
            result = super().build(x,adj,**kwargs)
        temp = path.with_suffix('.'+str(os.getpid())+'.tmp')
        with temp.open('wb') as stream:
            pickle.dump(result,stream,protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
        return result

class GPRGNN(nn.Module):
    """GPR-GNN: two-layer MLP and trainable generalized PageRank coefficients."""
    def __init__(self,fin,hidden,classes,dropout,operator):
        super().__init__()
        self.first,self.second = nn.Linear(fin,hidden),nn.Linear(hidden,classes)
        self.dropout,self.operator = dropout,operator
        alpha,K = .1,10
        coefficients = [alpha*(1-alpha)**k for k in range(K)]+[(1-alpha)**K]
        self.coefficients = nn.Parameter(torch.tensor(coefficients))
    def forward(self,x,edges=None):
        x = F.dropout(x,p=self.dropout,training=self.training)
        x = F.relu(self.first(x))
        x = F.dropout(x,p=self.dropout,training=self.training)
        x = self.second(x)
        result = self.coefficients[0]*x
        for k in range(1,len(self.coefficients)):
            x = torch.sparse.mm(self.operator,x)
            result = result+self.coefficients[k]*x
        return result

class GCNII(nn.Module):
    """GCNII: 16 layers, alpha=.1, theta=.5, shared identity mapping weights."""
    def __init__(self,fin,hidden,classes,dropout):
        super().__init__()
        self.first,self.second = nn.Linear(fin,hidden),nn.Linear(hidden,classes)
        self.layers = nn.ModuleList([GCN2Conv(hidden,alpha=.1,theta=.5,layer=i+1,shared_weights=True,cached=True) for i in range(16)])
        self.dropout = dropout
    def forward(self,x,edges):
        x = F.dropout(x,p=self.dropout,training=self.training)
        x0 = F.relu(self.first(x)); x=x0
        for layer in self.layers:
            x=F.dropout(x,p=self.dropout,training=self.training)
            x=F.relu(layer(x,x0,edges))
        return self.second(F.dropout(x,p=self.dropout,training=self.training))

class HCHA(nn.Module):
    """Two PyG attention HypergraphConv layers; mean hyperedge attributes."""
    def __init__(self,fin,hidden,classes,dropout,operator):
        super().__init__()
        self.first = HypergraphConv(fin,hidden,use_attention=True,heads=1,dropout=dropout)
        self.second = HypergraphConv(hidden,classes,use_attention=True,heads=1,concat=False,dropout=dropout)
        self.dropout,self.operator=dropout,operator
        self.register_buffer('incidences',operator['h'].indices())
        self.attr0 = None
    def attrs(self,x):
        return torch.sparse.mm(self.operator['ht'],x)*self.operator['de_inv'].unsqueeze(-1)
    def forward(self,x,edges=None):
        if self.attr0 is None:
            self.attr0 = self.attrs(x).detach()
        # Native input features, same input dropout as other networks.
        xd = F.dropout(x,p=self.dropout,training=self.training)
        x = F.elu(self.first(xd,self.incidences,hyperedge_attr=self.attr0,num_edges=self.operator['ht'].shape[0]))
        x = F.dropout(x,p=self.dropout,training=self.training)
        return self.second(x,self.incidences,hyperedge_attr=self.attrs(x),num_edges=self.operator['ht'].shape[0])

def candidates(name,method):
    presets = [dict(hidden_dim=64,dropout=.5,lr=.01,weight_decay=.0005),
               dict(hidden_dim=64,dropout=.5,lr=.003,weight_decay=.0005),
               dict(hidden_dim=128,dropout=.5,lr=.01,weight_decay=.0005),
               dict(hidden_dim=64,dropout=.6,lr=.01,weight_decay=.005)]
    if method not in ('prism','legacy_ce','legacy_full'):
        return [dict(id=f'c{i}',**p) for i,p in enumerate(presets)]
    source = ROOT/'configs/node'/f'{name}.yaml'
    if not source.exists():
        source = ROOT/'configs/node/coauthor_cs.yaml'
    base = yaml.safe_load(source.read_text(encoding='utf-8'))
    base.pop('dataset',None)
    base['model'].update(hidden_dim=128,proj_dim=64,enabled_views=['structural','feature_knn','prototype'],
                         structural_cap=32,structural_candidate_cap=32,hop_k=2,knn_k=12,n_clusters=64,K=2,
                         dropout=.5,gate_mode='soft',fusion_mode='embedding',private_gate_strength=.5,
                         view_gate_prior={'structural':.34,'feature_knn':.33,'prototype':.33},
                         densify_k=1,densify_min_similarity=.3,hyperedge_keep_ratio=.95,
                         structure_filter=method=='prism',dual_branch=method=='prism')
    base['loss'].update(lambda_align=.03,lambda_var=.005,lambda_elastic=0.,lambda_decorrelation=.005,
                        lambda_view_ce=0.,distill_warmup_epochs=5,distill_temperature=1.)
    if method=='legacy_full':
        base['loss'].update(lambda_align=.15,lambda_var=.05,lambda_elastic=.2,distill_warmup_epochs=20)
    base['optim'].update(lr=.003,weight_decay=.0005)
    if method == 'prism':
        base['model']['model_type']='prism'
    else:
        base['model'].pop('model_type',None)
    result=[]
    for i in range(4):
        cfg=copy.deepcopy(base)
        if i==1:
            cfg['model'].update(hidden_dim=64,proj_dim=32)
        elif i==2:
            cfg['model']['dropout']=.6
            cfg['optim']['weight_decay']=.002
            cfg['loss'].update(lambda_align=.01,lambda_var=.001,lambda_decorrelation=.001)
        elif i==3:
            cfg['model'].update(K=1,dropout=.5)
            cfg['optim']['lr']=.001
        result.append({'id':f'c{i}','config':cfg})
    return result

def make_context(inputs,method,spec):
    cfg = spec.get('config',{})
    key = 'symmetric_structural' if method in ('hgnn','hcha') else method
    if key in inputs.contexts:
        return inputs.contexts[key]
    m=cfg.get('model',{})
    # Candidate construction settings must affect the actual views. Keeping
    # them fixed here would make a tuning result misleading.
    builder=FastBuilder(hop_k=int(m.get('hop_k',2)),
                        structural_cap=int(m.get('structural_cap',32)),
                        knn_k=int(m.get('knn_k',12)),
                        n_clusters=int(m.get('n_clusters',64)),
                        random_state=CONSTRUCTION_SEED,soft_membership=method not in ('hgnn','hcha','legacy_ce'),
                        elastic_margin=cfg.get('loss',{}).get('elastic_margin',.18),
                        max_elastic_pairs=cfg.get('loss',{}).get('max_elastic_pairs',10000),
                        structure_filter=method=='prism',
                        structural_candidate_cap=int(m.get('structural_candidate_cap',32)),
                        densify_k=int(m.get('densify_k',1)),
                        densify_min_similarity=float(m.get('densify_min_similarity',.3)),
                        hyperedge_keep_ratio=float(m.get('hyperedge_keep_ratio',.95)),
                        hyperedge_score=m.get('hyperedge_score','coherence_coverage'),
                        preserve_node_coverage=bool(m.get('preserve_node_coverage',True)),
                        prototype_soft_topk=m.get('prototype_soft_topk',3))
    enabled_views=m.get('enabled_views',['structural','feature_knn','prototype'])
    views=builder.build(inputs.x.numpy(),inputs.adj,enabled_views=['structural'] if method in ('hgnn','hcha') else enabled_views,
                        include_elastic=method=='legacy_full',include_knn_pairs=method=='legacy_full')
    if method in ('hgnn','hcha'):
        # HGNN/HCHA consume the same binary, unfiltered k-hop incidence.
        h=views.incidences[0].copy();h.data[:]=1.
        ni,ei=builder._degree_stats(h)
        context={'h':tensor_sparse(h),'ht':tensor_sparse(h.T.tocsr()),
                 'dv_inv_sqrt':torch.from_numpy(np.sqrt(ni)).to(DEVICE),
                 'de_inv':torch.from_numpy(ei).to(DEVICE)}
    else:
        ops,names,prior=run_node.build_view_ops(views,{'enabled_views':enabled_views,
                                'view_gate_prior':m.get('view_gate_prior',
                                    {'structural':.34,'feature_knn':.33,'prototype':.33})})
        context={'ops':[{k:v.to(DEVICE) for k,v in op.items()} for op in ops],'prior':prior,'filter_stats':views.filter_stats,
                 'pairs':torch.from_numpy(views.elastic_pairs).long().to(DEVICE),
                 'lower':torch.from_numpy(views.elastic_lower).to(DEVICE),'upper':torch.from_numpy(views.elastic_upper).to(DEVICE)}
    inputs.contexts[key]=context
    return context

def make_model(inputs,method,spec):
    fin,classes=inputs.x.shape[1],int(inputs.y.max())+1
    if method in ('prism','legacy_ce','legacy_full'):
        context=make_context(inputs,method,spec)
        m=spec['config']['model']
        kw=dict(num_views=3,K=m['K'],dropout=m['dropout'],gate_temperature=m['gate_temperature'],
                gate_residual=m['gate_residual'],gate_prior=context['prior'])
        cls=PRISM if method=='prism' else MVHGCL
        if method=='prism':
            kw.update(dual_branch=True,private_gate_strength=m['private_gate_strength'])
        return cls(fin,m['hidden_dim'],m['proj_dim'],classes,**kw).to(DEVICE),context
    hidden,dropout=spec['hidden_dim'],spec['dropout']
    context = make_context(inputs,method,spec) if method in ('hgnn','hcha') else None
    if method=='gprgnn':
        model=GPRGNN(fin,hidden,classes,dropout,inputs.normalization())
    elif method=='gcnii':
        model=GCNII(fin,hidden,classes,dropout)
    elif method=='hcha':
        model=HCHA(fin,hidden,classes,dropout,context)
    else:
        model=StandardBaseline(fin,hidden,classes,method,dropout,dict(gat_heads=4),operator=context)
    return model.to(DEVICE),context

def fit(inputs,method,spec,split,init,final=False,epochs=None):
    seed_all(init)
    start=time.perf_counter()
    model,context=make_model(inputs,method,spec)
    x,y=inputs.x_gpu,inputs.y_gpu
    train=inputs.mask('train_mask',split)
    # WikiCS provides separate stopping and model-selection masks.
    stopping=inputs.mask('stopping_mask' if 'stopping_mask' in inputs.raw else 'val_mask',split)
    validation=inputs.mask('val_mask',split)
    metric=inputs.meta['metric']
    cfg=spec.get('config',{})
    optim=cfg.get('optim',spec)
    optimizer_cls=torch.optim.AdamW if cfg else torch.optim.Adam
    optimizer=optimizer_cls(model.parameters(),lr=optim['lr'],weight_decay=optim['weight_decay'])
    if method=='gprgnn':
        optimizer=torch.optim.Adam([{'params':list(model.first.parameters())+list(model.second.parameters()),'weight_decay':optim['weight_decay']},
                                     {'params':[model.coefficients],'weight_decay':0.}],lr=optim['lr'])
    best,best_epoch,bad,state=-1.,0,0,None
    history=[]
    variant='prism+soft_hg' if method=='prism' else ('soft_hg+dominant_distill+discrim_elastic' if method=='legacy_full' else 'baseline')
    flags=run_node.parse_variant(variant)
    empty=torch.empty((2,0),dtype=torch.long,device=DEVICE)
    bound=torch.empty(0,device=DEVICE)
    def forward():
        output=model(x,context['ops']) if cfg else model(x,inputs.edges)
        return output if cfg else {'logits':output}
    for epoch in range(1,(epochs or EPOCHS)+1):
        model.train()
        out=forward()
        loss=F.cross_entropy(out['logits'][train],y[train])
        components={'ce':float(loss.detach())}
        if method in ('prism','legacy_full'):
            align,var,elastic=run_node.compute_aux_losses(cfg,flags,out,context['pairs'],context['lower'],context['upper'],y,train,epoch,variant)
            lc=cfg['loss']
            loss=loss+lc['lambda_align']*align+lc['lambda_var']*var+lc['lambda_elastic']*elastic
            if method=='prism':
                decor=shared_private_decorrelation_loss(out['shared_list'],out['private_list'])
                loss=loss+lc['lambda_decorrelation']*decor
            components.update(align=float(align.detach()),variance=float(var.detach()))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{method} nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm=nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f'{method} nonfinite gradient')
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits=forward()['logits']
            score=run_node.masked_score(logits,y,stopping,metric)
        history.append({'epoch':epoch,'loss':float(loss.detach()),'stopping_score':score,**components})
        if score>best:
            best,best_epoch,bad=score,epoch,0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
        if EPOCH_CALLBACK is not None:
            EPOCH_CALLBACK({'epoch':epoch,'loss':float(loss.detach()),
                            'stopping_score':score,'best_stopping_score':best,
                            'best_epoch':best_epoch,'patience_used':bad})
        if bad>=PATIENCE:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits=forward()['logits']
        val=run_node.masked_score(logits,y,validation,metric)
        result={'best_val_score':val,'best_stopping_score':best,'best_epoch':best_epoch,'epochs_run':len(history),
                'history':history,'test_evaluations':0,'metric':metric,
                'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start}
        if final:
            result['best_test_score']=run_node.masked_score(logits,y,inputs.mask('test_mask',split),metric)
            result['test_evaluations']=1
    del model,optimizer,state,logits,out,loss
    if DEVICE.type=='cuda':
        torch.cuda.empty_cache()
    return result

def protocol():
    hashes={name:digest(ROOT/name) for name in SOURCES}
    signatures=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    payload={'version':VERSION,'source_hashes':hashes,'source_signature':signatures,
             'datasets':DATASETS,'methods':METHODS,'epochs':EPOCHS,'patience':PATIENCE,
             'candidate_count':4,'tuning':'four candidates on split 0/init 41; best two confirmed on split 1/init 42 (fixed official split for DBLP/Arxiv); mean validation selects; ties use candidate id',
             'final':'all ten official splits, all twenty WikiCS splits; DBLP and Arxiv fixed official split with ten initializations 201-210',
             'construction_seed':CONSTRUCTION_SEED,'device':str(DEVICE),
             'checkpoint':'validation-only; WikiCS stopping masks; never test',
             'test':'exactly once per final fit after restoring selected checkpoint; zero test scores in tuning records or histories',
             'precision':'float32, TF32 disabled; CUDA sparse scatter may be nondeterministic',
             'versions':{'torch':torch.__version__,'numpy':np.__version__,'python':sys.version.split()[0]},
             'hypergraph_baselines':'common binary unfiltered capped 2-hop hypergraph, cap32; HGNN symmetric normalization; HCHA PyG attention HypergraphConv with mean edge attributes',
             'scope':'local implementations under a common modest search budget; does not reproduce the entire literature leaderboard'}
    path=OUT/'protocol.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='frozen_at_utc'}!=payload:
            raise AssertionError('Frozen protocol changed. Use a new version/directory.')
        return old
    payload['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
    write(payload,path)
    with zipfile.ZipFile(OUT/'frozen_sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT/name,name)
    return payload

def run_method(inputs,method,manifest):
    folder=OUT/inputs.name/method
    specs=candidates(inputs.name,method)
    selected_path=folder/'selected.json'
    if not selected_path.exists():
        first=[]
        for spec in specs:
            path=folder/'tuning'/f"{spec['id']}_0_41.json"
            if not path.exists():
                result=fit(inputs,method,spec,0,41)
                result.update(spec=spec,source_signature=manifest['source_signature'],split=0,init=41)
                write(result,path)
                print(json.dumps({'dataset':inputs.name,'method':method,'stage':'tuning','candidate':spec['id'],'val':result['best_val_score'],'seconds':result['seconds']}),flush=True)
            first.append(read(path))
        finalists=sorted(first,key=lambda r:(-r['best_val_score'],r['spec']['id']))[:2]
        second=[]
        for initial in finalists:
            spec=initial['spec']
            split=min(1,inputs.num_splits-1)
            path=folder/'tuning'/f"{spec['id']}_{split}_42.json"
            if not path.exists():
                result=fit(inputs,method,spec,split,42)
                result.update(spec=spec,source_signature=manifest['source_signature'],split=split,init=42)
                write(result,path)
            check=read(path)
            second.append({'spec':spec,'val_mean':mean([initial['best_val_score'],check['best_val_score']])})
        winner=min(second,key=lambda r:(-r['val_mean'],r['spec']['id']))
        write({'selected':winner['spec'],'confirmation':second,'source_signature':manifest['source_signature'],
               'input_sha256':inputs.meta['sha256'],'selection':'validation-only'},selected_path)
    selection=read(selected_path)
    assert selection['source_signature']==manifest['source_signature']
    spec=selection['selected']
    final_indices=list(range(inputs.num_splits)) if inputs.num_splits>1 else list(range(10))
    for number in final_indices:
        split=number if inputs.num_splits>1 else 0
        init=201+number
        path=folder/'final'/f'split_{split}_init_{init}.json'
        if path.exists():
            continue
        graph=inputs.graph(split)
        result=fit(inputs,method,spec,split,init,final=True)
        result.update(dataset=inputs.name,method=method,split=split,init=init,
                      source_signature=manifest['source_signature'],spec=spec,fingerprint=input_fingerprint(graph))
        write(result,path)
        print(json.dumps({k:result[k] for k in ('dataset','method','split','init','best_val_score','best_test_score','best_epoch','seconds')}),flush=True)
    inputs.contexts.clear()
    if DEVICE.type=='cuda':
        torch.cuda.empty_cache()

def aggregate(manifest,strict=True):
    missing,summary=[],{}
    for name in DATASETS:
        summary[name]={}
        for method in METHODS:
            folder=OUT/name/method
            records=[read(path) for path in (folder/'final').glob('*.json')]
            expected=20 if name=='wikics' else 10
            if len(records)!=expected:
                missing.append(f'{name}/{method}: {len(records)}/{expected}')
                continue
            selection=read(folder/'selected.json')
            assert selection['selected']==min(selection['confirmation'],key=lambda r:(-r['val_mean'],r['spec']['id']))['spec']
            for record in records:
                assert record['source_signature']==manifest['source_signature']
                assert record['spec']==selection['selected']
                assert record['test_evaluations']==1
                assert all('test' not in key for row in record['history'] for key in row)
                peers=[read(path) for path in (OUT/name).glob(f"*/final/split_{record['split']}_init_{record['init']}.json")]
                assert all(p['fingerprint']==record['fingerprint'] for p in peers)
            values=[r['best_test_score'] for r in records]
            summary[name][method]={'mean':mean(values),'std':stdev(values),'n':len(values),'metric':records[0]['metric']}
    result={'complete':not missing,'missing':missing,'summary':summary,'source_signature':manifest['source_signature']}
    write(result,OUT/'summary.json')
    groups=[DATASETS[:5],DATASETS[5:]]
    tables=[]
    for group in groups:
        tables += ['| 方法 | '+' | '.join(group)+' |','|'+'---|'*(len(group)+1)]
        for method in METHODS:
            row=[LABELS[method]]
            for name in group:
                r=summary[name].get(method)
                row.append(f"{100*r['mean']:.2f} ± {100*r['std']:.2f}" if r else '待完成')
            tables.append('| '+' | '.join(row)+' |')
        tables.append('')
    (OUT/'main_results.md').write_text('\n'.join(tables),encoding='utf-8')
    print(json.dumps({'complete':not missing,'missing_cells':len(missing),'completed_cells':sum(map(len,summary.values()))}),flush=True)
    if strict and missing:
        raise AssertionError('Incomplete final matrix')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--datasets',nargs='+',default=DATASETS)
    ap.add_argument('--methods',nargs='+',default=METHODS)
    ap.add_argument('--aggregate',action='store_true')
    ap.add_argument('--partial',action='store_true')
    args=ap.parse_args()
    manifest=protocol()
    if args.aggregate:
        aggregate(manifest,not args.partial)
    else:
        for name in args.datasets:
            inputs=Inputs(name)
            for method in args.methods:
                run_method(inputs,method,manifest)
            del inputs
            if DEVICE.type=='cuda':
                torch.cuda.empty_cache()
        aggregate(manifest,False)
