"""Compose the twelve-method matrix, including the frozen sparse SAGE engine."""
from pathlib import Path
from statistics import mean,stdev
import hashlib
import json
import sys
HERE=Path(__file__).resolve().parent
OUT=HERE/'comparison'
LABELS={'mlp':'MLP','gcn':'GCN','gat':'GAT','sage':'GraphSAGE','appnp':'APPNP','gprgnn':'GPR-GNN',
        'gcnii':'GCNII','hgnn':'HGNN','hcha':'HCHA','legacy_ce':'原始框架（仅CE）','legacy_full':'原始完整框架','prism':'PRISM'}
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def folder(name,method):
    engine='results'
    if method=='sage':engine='results_sparse_sage_linear'
    if name=='ogbn_arxiv' and method in ('hcha','gcnii'):engine='results_sparse_'+method
    return HERE/engine/name/method
def main():
    p=read(HERE/'results/protocol.json');s=read(HERE/'results_sparse_sage_linear/protocol.json')
    assert s['parent_source_signature']==p['source_signature']
    hp=HERE/'results_sparse_hcha/protocol.json'
    h=read(hp) if hp.exists() else None
    if h:assert h['parent_source_signature']==p['source_signature']
    gp=HERE/'results_sparse_gcnii/protocol.json'
    g=read(gp) if gp.exists() else None
    if g:assert g['parent_source_signature']==p['source_signature']
    report_protocol={'parent_source_signature':p['source_signature'],'sparse_sage_signature':s['source_signature'],
                     'datasets':p['datasets'],'methods':p['methods'],'parent_protocol':'../results/protocol.json',
                     'sage_protocol':'../results_sparse_sage_linear/protocol.json','expected_final_fits':1200,
                     'policy':'GraphSAGE uses the separately frozen equivalent sparse mean engine; Arxiv HCHA and GCNII use their separately frozen equivalent sparse engines; remaining cells use the unchanged parent engine'}
    report_protocol.update(sparse_hcha_signature=h['source_signature'] if h else None,
                          hcha_policy='Arxiv HCHA uses separately frozen equivalent sparse incidence propagation; other datasets use native PyG HCHA')
    report_protocol.update(sparse_gcnii_signature=g['source_signature'] if g else None,
                          gcnii_policy='Arxiv GCNII uses separately frozen equivalent sparse adjacency propagation; other datasets use native PyG GCNII')
    report_protocol['signature']=hashlib.sha256(json.dumps(report_protocol,sort_keys=True).encode()).hexdigest()
    save(report_protocol,OUT/'protocol.json')
    missing=[];summary={};count=0;source_records={}
    for name in p['datasets']:
        summary[name]={}
        expected=20 if name=='wikics' else 10
        for method in p['methods']:
            f=folder(name,method);records=[read(path) for path in (f/'final').glob('*.json')]
            if len(records)!=expected:
                missing.append(f'{name}/{method}: {len(records)}/{expected}');continue
            values=[r['best_test_score'] for r in records]
            summary[name][method]={'mean':mean(values),'std':stdev(values),'n':len(records),'metric':records[0]['metric']}
            count+=len(records)
            for path in (f/'final').glob('*.json'):
                source_records[str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save({'complete':not missing,'missing':missing,'summary':summary,'final_fits':count,
          'protocol_signature':report_protocol['signature'],'source_records':source_records},OUT/'summary.json')
    names={'actor':'Actor','amazon_ratings':'Amazon-Ratings','coauthor_cs':'Coauthor-CS','coauthor_physics':'Coauthor-Physics',
           'dblp':'DBLP*','ogbn_arxiv':'OGBN-Arxiv','wikics':'WikiCS','questions':'Questions (AUROC)','tolokers':'Tolokers (AUROC)'}
    tables=[]
    for group in [p['datasets'][:5],p['datasets'][5:]]:
        tables+=['| 方法 | '+' | '.join(names[n] for n in group)+' |','|'+'---|'*(len(group)+1)]
        for method in p['methods']:
            row=[LABELS[method]]
            for name in group:
                r=summary[name].get(method)
                cell=f"{100*r['mean']:.2f} ± {100*r['std']:.2f}" if r else '待完成'
                if r and len(summary[name])==len(p['methods']) and r['mean']==max(v['mean'] for v in summary[name].values()):cell='**'+cell+'**'
                row.append(cell)
            tables.append('| '+' | '.join(row)+' |')
        tables.append('')
    tables+=['单位：%；均值 ± 样本标准差。WikiCS 20次，其余10次；多划分数据集的标准差同时含划分与初始化变化。',
             'Questions/Tolokers 为 AUROC，其余为准确率。每个方法四组候选，仅通过验证集选参；最多200 epochs、patience 40。',
             '* DBLP 保留所有异构节点，将各类型原始特征分块拼接并加入类型指示，只评价作者；采用公开镜像及重建的原始MAGNN划分，不能混入其他DBLP协议的文献数值。']
    (OUT/'main_results.md').write_text('\n'.join(tables),encoding='utf-8')
    print(json.dumps({'complete':not missing,'complete_cells':sum(map(len,summary.values())),'final_fits':count,'missing_cells':len(missing)}))
    if missing and '--partial' not in sys.argv:sys.exit(1)
if __name__=='__main__':main()
