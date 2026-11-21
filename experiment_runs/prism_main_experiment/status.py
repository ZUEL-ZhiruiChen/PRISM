from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
total=0;tuning_total=0
for name in p['datasets']:
    n=0;cells=0
    for method in p['methods']:
        engine=HERE/'results_sparse_sage_linear' if method=='sage' and (HERE/'results_sparse_sage_linear/protocol.json').exists() else OUT
        if name=='ogbn_arxiv' and method in ('hcha','gcnii'):engine=HERE/('results_sparse_'+method)
        files=list((engine/name/method/'final').glob('*.json'))
        tuning_total+=len(list((engine/name/method/'tuning').glob('*.json')))
        n+=len(files)
        cells+=len(files)==(20 if name=='wikics' else 10)
    total+=n
    print(f'{name}: {n}/{240 if name=="wikics" else 120} final fits, {cells}/12 complete methods')
print(f'TOTAL {total}/1200 final; {tuning_total}/648 tuning')
for log in [HERE/name for name in ['worker_a_rest.log','worker_serial_questions.log','worker_serial_arxiv.log','worker_serial_gcnii.log'] if (HERE/name).exists()]:
    lines=log.read_text(encoding='utf-8',errors='replace').splitlines()
    if any('Traceback (most recent call last)' in s for s in lines):
        print(log.name,'ERROR:',lines[-1])
    else:
        for line in reversed(lines):
            try:r=json.loads(line)
            except Exception:continue
            if 'method' in r:print(log.name,r.get('dataset'),r['method'],r.get('stage','final'),r.get('split',''),r.get('init',''))
            else:print(log.name,r.get('stage','done'))
            break
for name in ['worker_serial_questions.err.log','worker_serial_arxiv.err.log','worker_serial_gcnii.err.log','prebuild_arxiv_full.err.log','prebuild_arxiv_prism.err.log','prebuild_questions.err.log','serial_remaining.err.log']:
    path=HERE/name
    if path.exists():
        lines=path.read_text(encoding='utf-8',errors='replace').splitlines()
        if any('Traceback (most recent call last)' in line for line in lines):
            print(name,'ERROR:',lines[-1])
if (HERE/'serial_state.json').exists():
    state=json.loads((HERE/'serial_state.json').read_text(encoding='utf-8'))
    print('serial queue:',state['stage'],state.get('job',state.get('pid','')))
