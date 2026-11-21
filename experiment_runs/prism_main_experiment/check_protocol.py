"""Small substantive checks before committing expensive full-size runs."""
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
import suite

def main():
    suite.seed_all(6)
    rng=np.random.default_rng(6)
    x=rng.normal(size=(70,12)).astype(np.float32)
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    with tempfile.TemporaryDirectory() as folder, patch.object(suite,'CACHE',Path(folder)):
        b=suite.FastBuilder(knn_k=5,soft_membership=True)
        hes,ids=b._knn_hyperedges(x)
        dist,expected=NearestNeighbors(n_neighbors=6,metric='cosine').fit(x).kneighbors()
        assert np.array_equal(ids,expected)
        assert np.allclose(np.array([[w for _,w in row] for row in hes]),np.maximum(.001,1-dist),atol=2e-6)
    print('PASS: blockwise CUDA cosine search matches sklearn on untied inputs',flush=True)
    data=object.__new__(suite.Inputs)
    data.name='synthetic'
    data.x=torch.from_numpy(x)
    data.y=torch.arange(70)%2
    data.adj=sp.csr_matrix((np.ones(140,dtype=np.float32),(np.repeat(np.arange(70),2),np.column_stack([(np.arange(70)-1)%70,(np.arange(70)+1)%70]).reshape(-1))),shape=(70,70))
    coo=data.adj.tocoo()
    data.edges=torch.tensor(np.vstack([coo.row,coo.col]),dtype=torch.long,device=suite.DEVICE)
    data.x_gpu,data.y_gpu=data.x.to(suite.DEVICE),data.y.to(suite.DEVICE)
    data.contexts={};data.norm=None;data.meta={'metric':'auroc'}
    masks={key:np.zeros((1,70),dtype=bool) for key in ('train_mask','val_mask','test_mask')}
    masks['train_mask'][0,:30]=True;masks['val_mask'][0,30:50]=True;masks['test_mask'][0,50:]=True
    data.raw=masks
    for method in suite.METHODS:
        spec=suite.candidates('actor',method)[1]
        if 'config' in spec:
            spec['config']['loss']['distill_warmup_epochs']=1
        with tempfile.TemporaryDirectory() as folder,patch.object(suite,'CACHE',Path(folder)):
            r=suite.fit(data,method,spec,0,12,final=True,epochs=3)
            assert r['test_evaluations']==1 and 0<=r['best_test_score']<=1
            assert all('test' not in k for row in r['history'] for k in row)
            assert r['best_stopping_score']==max(row['stopping_score'] for row in r['history'])
        data.contexts.clear()
        print('PASS: '+method+' forward/backward, AUROC, stopping and test boundaries',flush=True)
    # Permuting held-out test labels cannot affect fitting or validation selection.
    spec=suite.candidates('actor','gcn')[0]
    r1=suite.fit(data,'gcn',spec,0,44,final=False,epochs=4)
    data.y_gpu=data.y_gpu.clone();data.y_gpu[50:]=1-data.y_gpu[50:]
    r2=suite.fit(data,'gcn',spec,0,44,final=False,epochs=4)
    assert r1['best_val_score']==r2['best_val_score'] and r1['best_epoch']==r2['best_epoch']
    assert r1['test_evaluations']==r2['test_evaluations']==0
    print('PASS: test-label permutation leaves fitting/validation unaffected',flush=True)

if __name__=='__main__':
    main()
