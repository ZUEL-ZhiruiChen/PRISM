"""Run the remaining frozen jobs sequentially, preserving completed records."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def stamp():return datetime.now(timezone.utc).isoformat()
def state(payload):
    (HERE/'serial_state.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);args=ap.parse_args()
    jobs=[
        ('questions',[str(HERE/'suite.py'),'--datasets','questions','--methods',
                      'gprgnn','gcnii','hgnn','hcha','legacy_ce','legacy_full','prism']),
        ('arxiv',[str(HERE/'suite.py'),'--datasets','ogbn_arxiv','--methods',
                  'legacy_ce','legacy_full','prism','gat']),
        ('gcnii',[str(HERE/'sparse_gcnii.py')]),
    ]
    if args.wait_pid:
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x100000,False,args.wait_pid)
        state({'stage':'waiting_for_tolokers','pid':args.wait_pid,'at':stamp(),'jobs':jobs})
        if handle:
            while True:
                result=kernel.WaitForSingleObject(handle,1000)
                if result==0:break
                if result!=258:raise OSError('WaitForSingleObject failed')
            kernel.CloseHandle(handle)
    for name,cmd in jobs:
        state({'stage':'running','job':name,'at':stamp(),'command':[sys.executable,'-u',*cmd]})
        print(json.dumps({'stage':'start','job':name,'at':stamp()}),flush=True)
        with (HERE/f'worker_serial_{name}.log').open('a',encoding='utf-8') as out, \
             (HERE/f'worker_serial_{name}.err.log').open('a',encoding='utf-8') as err:
            result=subprocess.run([sys.executable,'-u',*cmd],cwd=ROOT,stdout=out,stderr=err)
        if result.returncode:
            state({'stage':'failed','job':name,'code':result.returncode,'at':stamp()})
            raise SystemExit(result.returncode)
        print(json.dumps({'stage':'finished','job':name,'at':stamp()}),flush=True)
    state({'stage':'complete','at':stamp()})
