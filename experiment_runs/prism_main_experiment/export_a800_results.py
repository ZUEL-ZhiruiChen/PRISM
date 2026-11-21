"""Export verified, validation-selected results; optionally follow a detached run.

This exporter never trains a model or ranks configurations by test score.
Unfinished datasets keep a clearly marked result from the completed campaign.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import io
import json
import math
from pathlib import Path
from statistics import mean,stdev
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'server_runs'
PREVIOUS=ROOT/'a800_adaptive_20261112'
CURRENT=ROOT/'a800_refine_20261113'
DATASETS=['actor','amazon_ratings','coauthor_cs','coauthor_physics','dblp',
          'ogbn_arxiv','wikics','questions','tolokers']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def atomic(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.export.tmp')
    temp.write_text(text,encoding='utf-8')
    temp.replace(path)


def alive(folder):
    try:
        pid=int((folder/'launcher.pid').read_text())
        proc=Path('/proc')/str(pid)
        return (proc/'stat').read_text().split(')',1)[1].split()[0]!='Z' and \
               b'a800_refine.py' in (proc/'cmdline').read_bytes()
    except (OSError,ValueError,IndexError):
        return False


def export():
    progress=read(CURRENT/'progress.json') if (CURRENT/'progress.json').exists() else {}
    rows=[]
    for name in DATASETS:
        folder=CURRENT if (CURRENT/name/'final_summary.json').exists() else PREVIOUS
        path=folder/name/'final_summary.json'
        if not path.exists():
            continue
        record=read(path)
        summary=record['summary']
        values=[run['best_test_score'] for run in record['runs']]
        assert len(values)==summary['final_run_count'] and len(values)>=10
        assert all(math.isfinite(value) and 0<=value<=1 for value in values)
        assert abs(mean(values)-summary['best_test_score_mean'])<1e-10
        assert abs(stdev(values)-summary['best_test_score_std'])<1e-10
        notes=summary['paper_comparability']
        if name.startswith('coauthor_'):
            notes+='; local 20 train nodes per class, 1000 validation and 1000 test; paper split not fully specified'
        rows.append({'dataset':name,'result_stage':'refine' if folder==CURRENT else 'previous_adaptive',
                     'validation_mean':summary['best_val_score'],
                     'test_mean':mean(values),'test_std':stdev(values),
                     'single_run_test_max':max(values),'run_count':len(values),
                     'paper_reference':summary['target_from_paper'],
                     'nominal_gap_pp':100*(mean(values)-summary['target_from_paper']),
                     'paper_comparability':notes,'config_key':summary['candidate'],
                     'config':summary['config'],'source':str(path),'runs':record['runs']})
    result={'generated_at':datetime.now(timezone.utc).isoformat(),
            'campaign_status':progress.get('status','not_started'),
            'refine_completed_datasets':sum(row['result_stage']=='refine' for row in rows),
            'selection':'configuration and checkpoint selected by validation only; five-run validation audit for refine',
            'maximum_note':'Single-run maxima describe repeats of a fixed selected config and do not determine the formal mean or a SOTA claim.',
            'rows':rows}
    atomic(CURRENT/'verified_results.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    fields=['dataset','result_stage','validation_mean','test_mean','test_std','single_run_test_max',
            'run_count','paper_reference','nominal_gap_pp','paper_comparability','config_key','source']
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic(CURRENT/'verified_results.csv','\ufeff'+stream.getvalue())
    lines=['# PRISM 已完成实验结果',
           '',f"更新时间：{result['generated_at']}；扩大搜索状态：{result['campaign_status']}。",
           f"本阶段正式评估已完成 {result['refine_completed_datasets']} / 9 个数据集。",
           '', '配置和检查点仅依据验证集选择。扩大搜索结束后进行五次验证复核，再冻结配置进行正式测试。',
           '本阶段未完成的数据集保留上一阶段结果，并在表中注明。数值单位为 %；标准差为样本标准差。',
           '', '| 数据集 | 结果阶段 | 正式测试均值 ± 标准差 | 单次最高（描述值） | 次数 | 论文参考 |',
           '|---|---|---:|---:|---:|---:|']
    for row in rows:
        stage='本阶段' if row['result_stage']=='refine' else '上一阶段'
        lines.append(f"| {row['dataset']} | {stage} | {100*row['test_mean']:.2f} ± {100*row['test_std']:.2f} | {100*row['single_run_test_max']:.2f} | {row['run_count']} | {100*row['paper_reference']:.2f} |")
    lines+=['', '单次最高值来自固定配置的重复运行，不替代均值，也不参与配置筛选。',
            '论文参考来自用户提供的 mvhgcl_20261112161753.pdf 表 1；WikiCS 的最高参考方法为 GAT，其余为 MVHGCL。',
            'DBLP 特征构造与稿件不同；OGBN-Arxiv 本地为 40 类 Accuracy，稿件表注写 ROC-AUC；两者不作直接排名。',
            'Coauthor 本地每类 20 个训练节点、1,000 个验证节点和 1,000 个测试节点，稿件划分细节不足。其他数据集仍需核对划分才能声明 SOTA。',
            '', '完整配置、每次运行分数及服务器原始文件路径见同目录 verified_results.json。']
    atomic(CURRENT/'verified_results.md','\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        result=export()
        if not args.watch or not alive(CURRENT):
            print(json.dumps({key:value for key,value in result.items() if key!='rows'}),flush=True)
            return
        time.sleep(30)


if __name__=='__main__':
    main()
