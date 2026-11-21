"""Read-only status for the single-A800 campaigns and validation-selected results."""
from pathlib import Path
import json
import subprocess

ROOT=Path('/data1/Chenzr/PRISM/server_runs')

def load(folder):
    path=folder/'progress.json'
    return json.loads(path.read_text()) if path.exists() else {}

def alive(folder):
    try:
        pid=int((folder/'launcher.pid').read_text())
        proc=Path('/proc')/str(pid)
        state=(proc/'stat').read_text().split(')',1)[1].split()[0]
        return state!='Z' and b'a800' in (proc/'cmdline').read_bytes()
    except (OSError,ValueError,IndexError):
        return False

def main():
    first=ROOT/'a800_tuning_20261112'
    adaptive=ROOT/'a800_adaptive_20261112'
    refine=ROOT/'a800_refine_20261113'
    a,b,c=load(first),load(adaptive),load(refine)
    use_adaptive=bool(b) and b.get('status')!='waiting'
    folder=adaptive if use_adaptive else first
    selected=dict(b if use_adaptive else a)
    selected['phase']='adaptive' if use_adaptive else 'first_pass'
    if use_adaptive:
        selected['first_pass_results']=a.get('selected',[])
    if c and c.get('status')!='waiting':
        selected=dict(c)
        selected['phase']='refine'
        selected['previous_results']=b.get('selected',[]) or a.get('selected',[])
        selected['previous_campaign']='a800_adaptive_20261112' if b.get('selected') else 'a800_tuning_20261112'
        folder=refine
    # Single-run maxima describe repetitions of an already frozen config.
    # They never decide which configuration is shown as the formal result.
    for rows,source in ((selected.get('selected',[]),folder),
                        (selected.get('previous_results',[]),adaptive)):
        for row in rows:
            final_path=source/row['dataset']/'final_summary.json'
            if final_path.exists():
                final=json.loads(final_path.read_text())
                scores=[r['best_test_score'] for r in final.get('runs',[])]
                if scores:
                    row['single_run_test_max']=max(scores)
    gpu=subprocess.run(['nvidia-smi',
                       '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                       '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    print(json.dumps({'progress':selected,'gpu':gpu.stdout.strip(),
                      'launcher_alive':alive(folder),
                      'exit_code':(folder/'exit_code').read_text().strip()
                                  if (folder/'exit_code').exists() else None,
                      'adaptive_queued':b.get('status')=='waiting' and alive(adaptive),
                      'adaptive_status':b.get('status'),
                      'adaptive_rule':b.get('convergence_rule'),
                      'refine_queued':c.get('status')=='waiting' and alive(refine),
                      'refine_status':c.get('status'),
                      'campaign_directory':str(folder),
                      'first_pass_alive':alive(first),'adaptive_alive':alive(adaptive),
                      'refine_alive':alive(refine)}))

if __name__=='__main__':
    main()
