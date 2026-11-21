"""Read-only independent audit of the final experiment matrix."""
from pathlib import Path
import hashlib
import json
from statistics import mean,stdev
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while b:=stream.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def main():
    p=read(OUT/'protocol.json')
    errors,missing=[],[]
    total,tuning=0,0
    expected_source_records={}
    composed=(HERE/'comparison/summary.json').exists()
    summary=read((HERE/'comparison' if composed else OUT)/'summary.json')
    if composed:
        report_protocol=read(HERE/'comparison/protocol.json')
        unsigned={k:v for k,v in report_protocol.items() if k!='signature'}
        assert hashlib.sha256(json.dumps(unsigned,sort_keys=True).encode()).hexdigest()==report_protocol['signature']
        assert summary['protocol_signature']==report_protocol['signature']
        assert report_protocol['parent_source_signature']==p['source_signature']
    for filename,h in p['source_hashes'].items():
        assert digest(ROOT/filename)==h,filename
    sage_protocol=read(HERE/'results_sparse_sage_linear/protocol.json') if composed else None
    hcha_path=HERE/'results_sparse_hcha/protocol.json'
    hcha_protocol=read(hcha_path) if hcha_path.exists() else None
    gcnii_path=HERE/'results_sparse_gcnii/protocol.json'
    gcnii_protocol=read(gcnii_path) if gcnii_path.exists() else None
    if sage_protocol:
        assert sage_protocol['parent_source_signature']==p['source_signature']
        for filename,h in sage_protocol['source_hashes'].items():assert digest(ROOT/filename)==h
    if hcha_protocol:
        assert hcha_protocol['parent_source_signature']==p['source_signature']
        for filename,h in hcha_protocol['source_hashes'].items():assert digest(ROOT/filename)==h
    if gcnii_protocol:
        assert gcnii_protocol['parent_source_signature']==p['source_signature']
        for filename,h in gcnii_protocol['source_hashes'].items():assert digest(ROOT/filename)==h
    data_manifest=read(OUT/'data_manifest.json')
    expected_nodes={'actor':7600,'amazon_ratings':24492,'coauthor_cs':18333,'coauthor_physics':34493,
                    'dblp':26128,'ogbn_arxiv':169343,'wikics':11701,'questions':48921,'tolokers':11758}
    for name in p['datasets']:
        if not (HERE/'data'/f'{name}.npz').exists():
            missing.append('data/'+name)
            continue
        meta=read(HERE/'data'/f'{name}.json')
        assert meta==data_manifest[name]
        assert digest(HERE/'data'/f'{name}.npz')==meta['sha256']
        raw=np.load(HERE/'data'/f'{name}.npz')
        assert len(raw['y'])==expected_nodes[name]
        assert raw['x'].shape[1]==meta['features']
        assert raw['train_mask'].shape==raw['val_mask'].shape==raw['test_mask'].shape
        assert not np.any(raw['train_mask']&raw['val_mask'])
        assert not np.any(raw['train_mask']&raw['test_mask'])
        assert not np.any(raw['val_mask']&raw['test_mask'])
        assert not np.any(raw['y'][np.any(raw['train_mask']|raw['val_mask']|raw['test_mask'],axis=0)]<0)
        expected_metric='auroc' if name in ('questions','tolokers') else 'accuracy'
        fingerprints={}
        for method in p['methods']:
            folder=(HERE/'results_sparse_sage_linear' if composed and method=='sage' else OUT)/name/method
            expected_signature=sage_protocol['source_signature'] if composed and method=='sage' else p['source_signature']
            if composed and name=='ogbn_arxiv' and method=='hcha':
                folder=HERE/'results_sparse_hcha'/name/method
                expected_signature=hcha_protocol['source_signature'] if hcha_protocol else None
            if composed and name=='ogbn_arxiv' and method=='gcnii':
                folder=HERE/'results_sparse_gcnii'/name/method
                expected_signature=gcnii_protocol['source_signature'] if gcnii_protocol else None
            selected=folder/'selected.json'
            expected=20 if name=='wikics' else 10
            paths=list((folder/'final').glob('*.json'))
            if len(paths)!=expected or not selected.exists():
                missing.append(f'{name}/{method}: {len(paths)}/{expected}')
                continue
            if method not in summary['summary'].get(name,{}):
                missing.append(f'{name}/{method}: final records completed after the report snapshot; regenerate report')
                continue
            s=read(selected)
            assert s['input_sha256']==meta['sha256']
            assert s['selection']=='validation-only'
            assert s['source_signature']==expected_signature
            candidates={};histories={}
            tuning_paths=list((folder/'tuning').glob('*.json'))
            assert len(tuning_paths)==6
            for path in tuning_paths:
                r=read(path);tuning+=1
                assert r['source_signature']==expected_signature
                assert r['test_evaluations']==0 and 'best_test_score' not in r
                assert r['metric']==expected_metric
                assert r['best_stopping_score']==max(row['stopping_score'] for row in r['history'])
                assert all('test' not in k for row in r['history'] for k in row)
                histories[(r['spec']['id'],r['init'])]=r
                if r['init']==41:candidates[r['spec']['id']]=r
            assert len(candidates)==4
            first_two=sorted(candidates,key=lambda c:(-candidates[c]['best_val_score'],c))[:2]
            assert sorted(first_two)==sorted(x['spec']['id'] for x in s['confirmation'])
            for entry in s['confirmation']:
                cid=entry['spec']['id']
                v=mean([histories[(cid,41)]['best_val_score'],histories[(cid,42)]['best_val_score']])
                assert v==entry['val_mean']
            assert s['selected']==min(s['confirmation'],key=lambda r:(-r['val_mean'],r['spec']['id']))['spec']
            scores=[];keys=[]
            for path in paths:
                r=read(path);total+=1
                expected_source_records[str(path.relative_to(HERE))]=digest(path)
                assert r['source_signature']==expected_signature
                assert r['spec']==s['selected'] and r['metric']==expected_metric
                assert r['test_evaluations']==1 and 0<=r['best_test_score']<=1
                assert r['best_stopping_score']==max(row['stopping_score'] for row in r['history'])
                assert all('test' not in k for row in r['history'] for k in row)
                assert len(r['history'])<=p['epochs']
                assert r['best_epoch']==next(row['epoch'] for row in r['history'] if row['stopping_score']==r['best_stopping_score'])
                key=(r['split'],r['init']);keys.append(key)
                expected_split=0 if raw['train_mask'].shape[0]==1 else r['init']-201
                assert r['split']==expected_split
                assert 201<=r['init']<201+expected
                if key in fingerprints:assert fingerprints[key]==r['fingerprint']
                else:fingerprints[key]=r['fingerprint']
                scores.append(r['best_test_score'])
            assert len(set(keys))==expected
            reported=summary['summary'][name][method]
            assert abs(reported['mean']-mean(scores))<1e-14
            assert abs(reported['std']-stdev(scores))<1e-14
        raw.close()
    result={'audit_passed':not missing,'final_fits_audited':total,'tuning_fits_audited':tuning,
            'expected_final_fits':1200,'expected_tuning_fits':648,'missing':missing,
            'source_signature':p['source_signature']}
    if composed:
        for filename,h in summary['source_records'].items():assert digest(HERE/filename)==h
        if not missing:
            assert summary['source_records']==expected_source_records
            assert summary['complete'] and summary['final_fits']==1200
            assert total==1200 and tuning==648
        result['protocol_signature']=read(HERE/'comparison/protocol.json')['signature']
    ((HERE/'comparison' if composed else OUT)/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if missing and '--partial' not in sys.argv:sys.exit(1)

if __name__=='__main__':main()
