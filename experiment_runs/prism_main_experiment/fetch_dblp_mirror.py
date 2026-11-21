"""Recover the public MAGNN DBLP data from a research-code mirror.

Only numeric arrays are loaded. The original MAGNN split is reconstructed
using the exact seed and two train_test_split calls in its public notebook.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.request
import zipfile
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RAW=ROOT/'data/raw/full_suite'
folder=RAW/'dblp'
folder.mkdir(parents=True,exist_ok=True)
base='https://raw.githubusercontent.com/liuhao33/MSGNN/main/data/DBLP_processed/'
names=['adjM.npz','features_0.npz','features_1.npz','features_2.npy','labels.npy','node_types.npy']
def fetch(name):
    path=folder/name
    if not path.exists():
        with urllib.request.urlopen(base+name,timeout=30) as response,path.with_suffix(path.suffix+'.part').open('wb') as output:
            while b:=response.read(32*1024):output.write(b)
        path.with_suffix(path.suffix+'.part').replace(path)
    return name,hashlib.sha256(path.read_bytes()).hexdigest()

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=6) as pool:
        hashes=dict(pool.map(fetch,names))
    types=np.load(folder/'node_types.npy',allow_pickle=False)
    labels=np.load(folder/'labels.npy',allow_pickle=False)
    assert len(types)==26128 and np.bincount(types).tolist()==[4057,14328,7723,20]
    assert sp.load_npz(folder/'features_0.npz').shape==(4057,334)
    paper_shape=sp.load_npz(folder/'features_1.npz').shape
    assert paper_shape in [(14328,4231),(14328,4233)]
    assert np.load(folder/'features_2.npy',allow_pickle=False).shape==(7723,50)
    assert sp.load_npz(folder/'adjM.npz').shape==(26128,26128)
    assert labels.shape==(4057,) and len(np.unique(labels))==4
    # Verbatim split parameters from the original MAGNN preprocessing notebook.
    seed=1566911444
    train,val=train_test_split(np.arange(len(labels)),test_size=400,random_state=seed)
    train,test=train_test_split(train,test_size=3257,random_state=seed)
    np.savez(folder/'train_val_test_idx.npz',train_idx=np.sort(train),val_idx=np.sort(val),test_idx=np.sort(test))
    assert [len(train),len(val),len(test)]==[400,400,3257]
    with zipfile.ZipFile(RAW/'dblp.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in names+['train_val_test_idx.npz']:
            archive.write(folder/name,name)
    provenance={'mirror':base,'raw_sha256':hashes,'verified_shapes':True,
                'paper_feature_dimensions':paper_shape[1],
                'version_note':'This mirror has 4233 paper vocabulary dimensions, versus 4231 in the PyG documentation; author features remain 334 dimensions and node counts match MAGNN.',
                'split_source':'https://github.com/cynricfu/MAGNN/blob/master/preprocess_DBLP.ipynb',
                'split_reconstruction_seed':seed,'split_counts':[400,400,3257]}
    (RAW/'dblp_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance),flush=True)
