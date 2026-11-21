"""Persistent single-A800 population search with validation plateau stopping.

Waits for the existing campaign's GPU lock. Each dataset is searched in turn,
eight new candidates per round. The top two are confirmed on three fixed
validation runs in total. Three rounds without a 0.05 percentage point gain
end that dataset's search; held-out tests are evaluated only after all search
decisions have been frozen. No claim of a global optimum is made.
"""
from __future__ import annotations
from collections import OrderedDict
from datetime import datetime, timezone
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
for option in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[option] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIRST = ROOT/'server_runs/a800_tuning_20261112'
OUT = ROOT/'server_runs/a800_adaptive_20261112'
sys.path.insert(0,str(HERE))
import adaptive_space as space

DATASETS = ['actor','amazon_ratings','coauthor_cs','coauthor_physics','dblp',
            'ogbn_arxiv','wikics','questions','tolokers']
EPOCHS, PATIENCE, BATCH_SIZE, PLATEAU_ROUNDS, MIN_DELTA = 400,60,8,3,.0005
suite = None

def now():
    return datetime.now(timezone.utc).isoformat()

def write(value,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def log(event,**fields):
    print(json.dumps({'event':event,'timestamp':now(),**fields},ensure_ascii=False),flush=True)

def progress(state,**fields):
    state.update(fields,updated_at=now())
    write(state,OUT/'progress.json')

def signature():
    names=['run_node.py','models/model.py','models/losses.py','models/hypergraph.py',
           'utils/data_node.py']
    paths=[ROOT/name for name in names]+[Path(__file__),HERE/'adaptive_space.py',
                                        HERE/'a800_suite.py',HERE/'prepare_data.py',
                                        HERE/'check_adaptive.py']
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes

class Fitter:
    def __init__(self,inputs,state,source_signature):
        self.inputs,self.state,self.source_signature=inputs,state,source_signature
        self.contexts=OrderedDict()

    def context(self,cfg):
        m=cfg['model']
        keys=('hop_k','structural_cap','structural_candidate_cap','knn_k','n_clusters',
              'densify_k','densify_min_similarity','hyperedge_keep_ratio')
        settings={key:m[key] for key in keys}
        settings.update(prototype_soft_topk=m.get('prototype_soft_topk',3),
                        hyperedge_score=m.get('hyperedge_score','coherence_coverage'),
                        prototype_method=m.get('prototype_method','kmeans'),
                        preserve_node_coverage=True)
        key=json.dumps(settings,sort_keys=True)
        if key in self.contexts:
            self.contexts.move_to_end(key)
            return self.contexts[key]
        progress(self.state,stage='adaptive_build',current_epoch=0,
                 current_loss=None,current_stopping_score=None,
                 current_best_stopping_score=None)
        builder=suite.FastBuilder(**settings,random_state=0,soft_membership=True,
                    structure_filter=True,
                    max_elastic_pairs=cfg['loss'].get('max_elastic_pairs',12000),
                    elastic_margin=cfg['loss'].get('elastic_margin',.1))
        views=builder.build(self.inputs.x.numpy(),self.inputs.adj,
                            enabled_views=['structural','feature_knn','prototype'],
                            include_elastic=False,include_knn_pairs=False)
        ops,names,_=suite.run_node.build_view_ops(views,m)
        context={'ops':[{k:v.to(suite.DEVICE) for k,v in op.items()} for op in ops],
                 'names':names,'filter_stats':views.filter_stats}
        self.contexts[key]=context
        while len(self.contexts)>2:
            self.contexts.popitem(last=False)
        return context

    def fit(self,cfg,split,init,stage,final=False,persist=True):
        cfg=space.normalize(cfg)
        key=space.fingerprint(cfg)
        path=OUT/self.inputs.name/'fits'/key/f'split_{split}_init_{init}.json'
        if persist and path.exists():
            result=read(path)
            assert result['source_signature']==self.source_signature
            assert result['input_sha256']==self.inputs.meta['sha256']
            assert result['config']==cfg
            assert result['test_evaluations']==int(final)
            return result
        torch,F=suite.torch,suite.F
        suite.seed_all(init)
        started=time.perf_counter()
        progress(self.state,dataset=self.inputs.name,stage=stage,candidate=key,
                 split=split,init=init,current_epoch=0,current_loss=None,
                 current_stopping_score=None,current_best_stopping_score=None)
        context=self.context(cfg)
        m=cfg['model']
        names=context['names']
        priors=[m.get('view_gate_prior',{}).get(name,1.) for name in names]
        model=suite.PRISM(self.inputs.x.shape[1],m['hidden_dim'],m['proj_dim'],
                         int(self.inputs.y.max())+1,num_views=len(names),
                         K=m['K'],dropout=m['dropout'],dual_branch=True,
                         private_gate_strength=m.get('private_gate_strength',.5),
                         gate_temperature=m['gate_temperature'],
                         gate_residual=m['gate_residual'],gate_prior=priors,
                         gate_mode=m.get('gate_mode','soft'),refine=m.get('refine',False),
                         fusion_mode='embedding').to(suite.DEVICE)
        optim=cfg['optim']
        optimizer=torch.optim.AdamW(model.parameters(),lr=optim['lr'],
                                     weight_decay=optim['weight_decay'])
        x,y=self.inputs.x_gpu,self.inputs.y_gpu
        train=self.inputs.mask('train_mask',split)
        stopping=self.inputs.mask('stopping_mask' if 'stopping_mask' in self.inputs.raw
                                  else 'val_mask',split)
        validation=self.inputs.mask('val_mask',split)
        metric=self.inputs.meta['metric']
        flags=suite.run_node.parse_variant('prism+soft_hg')
        pairs=torch.empty((2,0),dtype=torch.long,device=suite.DEVICE)
        bounds=x.new_empty(0)
        best,best_epoch,bad,best_state=-1.,0,0,None
        history=[]
        try:
            for epoch in range(1,EPOCHS+1):
                model.train()
                out=model(x,context['ops'])
                lc=cfg['loss']
                ce=F.cross_entropy(out['logits'][train],y[train],
                                   label_smoothing=lc.get('label_smoothing',0.))
                view_ce=suite.run_node.view_ce_loss(out,y,train,
                              label_smoothing=lc.get('label_smoothing',0.)) if lc['lambda_view_ce']>0 else x.new_tensor(0.)
                align,var,_=suite.run_node.compute_aux_losses(cfg,flags,out,pairs,bounds,bounds,
                                                               y,train,epoch,'prism+soft_hg')
                decor=suite.shared_private_decorrelation_loss(out['shared_list'],out['private_list'])
                loss=ce+lc['lambda_view_ce']*view_ce+lc['lambda_align']*align
                loss=loss+lc['lambda_var']*var+lc['lambda_decorrelation']*decor
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite training gradients')
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    score=suite.run_node.masked_score(model(x,context['ops'])['logits'],
                                                      y,stopping,metric)
                row={'epoch':epoch,'loss':float(loss.detach()),'stopping_score':score,
                     'ce':float(ce.detach()),'view_ce':float(view_ce.detach()),
                     'align':float(align.detach()),'variance':float(var.detach()),
                     'decorrelation':float(decor.detach())}
                history.append(row)
                if score>best:
                    best,best_epoch,bad=score,epoch,0
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:
                    bad+=1
                progress(self.state,stage=stage,current_epoch=epoch,current_loss=row['loss'],
                         current_stopping_score=score,current_best_stopping_score=best,
                         current_best_epoch=best_epoch,patience_used=bad)
                log('epoch',dataset=self.inputs.name,candidate=key,stage=stage,
                    split=split,init=init,**row)
                if bad>=PATIENCE:
                    break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                logits=model(x,context['ops'])['logits']
                val=suite.run_node.masked_score(logits,y,validation,metric)
                result={'best_val_score':val,'best_stopping_score':best,'best_epoch':best_epoch,
                        'history':history,'epochs_run':len(history),'test_evaluations':0,
                        'metric':metric,'parameters':sum(p.numel() for p in model.parameters()),
                        'config':cfg,'input_sha256':self.inputs.meta['sha256'],
                        'source_signature':self.source_signature,'split':split,'init':init,
                        'execution_device':str(suite.DEVICE),'filter_stats':context['filter_stats'],
                        'seconds':time.perf_counter()-started,'finished_at':now()}
                if final:
                    result['best_test_score']=suite.run_node.masked_score(logits,y,
                                                       self.inputs.mask('test_mask',split),metric)
                    result['test_evaluations']=1
                    if persist:
                        checkpoint=path.with_suffix('.pt')
                        checkpoint.parent.mkdir(parents=True,exist_ok=True)
                        temp=checkpoint.with_suffix('.pt.tmp')
                        torch.save({'model_state':best_state,'config':cfg,
                                    'source_signature':self.source_signature,
                                    'input_sha256':self.inputs.meta['sha256'],
                                    'best_epoch':best_epoch,'split':split,'init':init},temp)
                        temp.replace(checkpoint)
                        result['checkpoint']=str(checkpoint)
            if persist:
                write(result,path)
            return result
        finally:
            del model,optimizer,best_state
            gc.collect()

def initial_parents(name):
    # Only validation trial files are read. Final/test result directories are excluded.
    candidates=[]
    for path in sorted((FIRST/name/'tuning').glob('*.json')):
        record=read(path)
        assert record['test_evaluations']==0 and 'best_test_score' not in record
        candidates.append((record['best_val_score'],record['spec']['config']))
    if not candidates:
        candidates=[(0.,suite.candidates(name,'prism')[0]['config'])]
    parents=[space.normalize(cfg) for _,cfg in sorted(candidates,key=lambda row:-row[0])]
    targeted=copy.deepcopy(parents[0])
    targeted['model'].update(hidden_dim=256,proj_dim=64,K=1,dropout=.5)
    targeted['loss'].update(lambda_align=.01,lambda_var=.005,
                            lambda_decorrelation=.005,lambda_view_ce=.2)
    targeted['optim'].update(lr=.001,weight_decay=.002)
    if name in ('coauthor_cs','coauthor_physics','ogbn_arxiv','dblp'):
        targeted['model'].update(K=3,view_gate_prior=copy.deepcopy(space.PRIORS[1]))
    elif name in ('actor','questions'):
        targeted['model']['view_gate_prior']=copy.deepcopy(space.PRIORS[2])
    else:
        targeted['model'].update(space.GRAPH_PROFILES[2])
    parents.append(space.normalize(targeted))
    for profile in (space.GRAPH_PROFILES[2],space.GRAPH_PROFILES[3]):
        cfg=copy.deepcopy(targeted)
        cfg['model'].update(profile)
        parents.append(space.normalize(cfg))
    return parents

def summary(state,stats):
    state['search_summary']={name:{'rounds':s.get('rounds',0),
                           'stale_rounds':s.get('stale_rounds',0),
                           'converged':s.get('converged',False),
                           'blocked':s.get('blocked',False),
                           'best_validation_mean':s.get('best',{}).get('validation_mean'),
                           'best_candidate':s.get('best',{}).get('key')}
                           for name,s in stats.items()}
    state['search_datasets_converged']=sum(s.get('converged',False) for s in stats.values())

def save_search(state,stats,source_signature):
    summary(state,stats)
    write({'source_signature':source_signature,'datasets':stats},OUT/'search_state.json')
    progress(state)

def search_round(name,stats,state,source_signature):
    from prepare_data import prepare
    ds=stats[name]
    progress(state,stage='adaptive_prepare',dataset=name,search_round=ds['rounds']+1)
    prepare(name)
    inputs=suite.Inputs(name)
    fitter=Fitter(inputs,state,source_signature)
    parents=[r['config'] for r in sorted(ds['confirmed'],key=lambda r:-r['validation_mean'])[:3]]
    if not parents:
        parents=initial_parents(name)
    proposals=space.candidates(name,ds['rounds'],parents,ds['seen'],BATCH_SIZE)
    screened=[]
    try:
        for index,cfg in enumerate(proposals,1):
            progress(state,candidate_index=index,candidate_total=len(proposals))
            key=space.fingerprint(cfg)
            try:
                result=fitter.fit(cfg,0,41,'adaptive_screen')
                assert result['test_evaluations']==0
            except Exception as error:
                log('candidate_failed',dataset=name,candidate=key,error=repr(error))
                write({'dataset':name,'candidate':key,'error':repr(error),'config':cfg},
                      OUT/name/'failed'/f'{key}.json')
                fitter.contexts.clear()
                gc.collect()
                suite.torch.cuda.empty_cache()
                continue
            screened.append({'key':key,'config':cfg,'best_val_score':result['best_val_score']})
            state.setdefault('trials',[]).append({'dataset':name,'candidate':key,
                           'best_val_score':result['best_val_score'],
                           'best_epoch':result['best_epoch'],'seconds':result['seconds']})
            progress(state)
        if not screened:
            raise RuntimeError('No candidate completed in this search round')
        confirmations=[]
        for row in sorted(screened,key=lambda r:(-r['best_val_score'],r['key']))[:2]:
            values=[row['best_val_score']]
            for number in (1,2):
                split=min(number,inputs.num_splits-1)
                result=fitter.fit(row['config'],split,41+number,'adaptive_confirm')
                assert result['test_evaluations']==0
                values.append(result['best_val_score'])
            confirmations.append({**row,'validation_scores':values,
                                  'validation_mean':suite.mean(values)})
        ds['confirmed'].extend(confirmations)
        winner=min(ds['confirmed'],key=lambda r:(-r['validation_mean'],r['key']))
        old_best=ds.get('best',{}).get('validation_mean',-1.)
        ds['stale_rounds']=space.update_plateau(old_best,winner['validation_mean'],
                                               ds['stale_rounds'],MIN_DELTA)
        ds['best']=copy.deepcopy(winner)
        ds['seen'].extend(space.fingerprint(cfg) for cfg in proposals)
        ds['rounds']+=1
        ds['converged']=ds['rounds']>=4 and ds['stale_rounds']>=PLATEAU_ROUNDS
        ds['failures']=0
        write(ds['best'],OUT/name/'best_validation_config.json')
        log('round_complete',dataset=name,round=ds['rounds'],
            best_validation_mean=winner['validation_mean'],
            gain=winner['validation_mean']-old_best,stale_rounds=ds['stale_rounds'],
            converged=ds['converged'],candidate=winner['key'])
        save_search(state,stats,source_signature)
    finally:
        fitter.contexts.clear()
        del fitter,inputs
        gc.collect()
        suite.torch.cuda.empty_cache()

def evaluate_frozen(name,ds,state,source_signature):
    from a800_tuning import PAPER_TARGETS,COMPARABILITY
    inputs=suite.Inputs(name)
    fitter=Fitter(inputs,state,source_signature)
    cfg=ds['best']['config']
    records=[]
    numbers=range(inputs.num_splits) if inputs.num_splits>1 else range(10)
    for number in numbers:
        split=number if inputs.num_splits>1 else 0
        result=fitter.fit(cfg,split,201+number,'adaptive_final',final=True)
        records.append({'split':split,'init':201+number,'best_test_score':result['best_test_score'],
                        'best_val_score':result['best_val_score'],'best_epoch':result['best_epoch']})
        progress(state,current_final_runs=len(records),
                 current_final_test_mean=suite.mean(r['best_test_score'] for r in records))
    values=[r['best_test_score'] for r in records]
    row={'dataset':name,'candidate':ds['best']['key'],
         'best_val_score':ds['best']['validation_mean'],
         'best_test_score_mean':suite.mean(values),'best_test_score_std':suite.stdev(values),
         'final_run_count':len(records),'target_from_paper':PAPER_TARGETS[name],
         'paper_comparability':COMPARABILITY.get(name,'nominal_metric_match; confirm split alignment'),
         'rounds':ds['rounds'],'config':cfg,'search_converged':True}
    write({'summary':row,'runs':records},OUT/name/'final_summary.json')
    state['selected'].append(row)
    progress(state)
    log('final_dataset_complete',**{k:v for k,v in row.items() if k!='config'})
    fitter.contexts.clear()
    del fitter,inputs
    gc.collect()
    suite.torch.cuda.empty_cache()

def main():
    global suite
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'launcher.pid').write_text(str(os.getpid()))
    state={'status':'waiting','phase':'adaptive','stage':'waiting_for_first_pass',
           'started_at':now(),'datasets':DATASETS,'selected':[],'trials':[],
           'epochs_per_trial':EPOCHS,'device':'cuda:0',
           'convergence_rule':{'rounds_without_gain':PLATEAU_ROUNDS,
                               'minimum_gain':MIN_DELTA,'batch_size':BATCH_SIZE},
           'selection_protocol':'validation only; top 2 in each batch confirmed on 3 fixed validation runs; all search configs frozen before final tests'}
    progress(state)
    lock=(FIRST/'campaign.lock').open('a')
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(10)
            progress(state)
    progress(state,status='running',stage='loading')
    import a800_suite as backend
    suite=backend
    assert suite.DEVICE.type=='cuda' and suite.torch.cuda.device_count()==1
    state['gpu']=suite.torch.cuda.get_device_name(0)
    assert 'A800' in state['gpu']
    source_signature,hashes=signature()
    protocol={'source_signature':source_signature,'source_hashes':hashes,
              'epochs':EPOCHS,'patience':PATIENCE,'convergence_rule':state['convergence_rule'],
              'device':'cuda:0','training_class':'models.model.PRISM',
              'search':state['selection_protocol'],
              'graph_profiles':space.GRAPH_PROFILES,'choices':space.CHOICES}
    write(protocol,OUT/'protocol.json')
    from check_adaptive import gpu_checks
    gpu_checks(suite,sys.modules[__name__])
    if (OUT/'search_state.json').exists():
        saved=read(OUT/'search_state.json')
        assert saved['source_signature']==source_signature, 'Code signature changed; use a new campaign directory'
        stats=saved['datasets']
    else:
        stats={name:{'rounds':0,'stale_rounds':0,'seen':[],
                     'confirmed':[],'converged':False,'failures':0} for name in DATASETS}
    summary(state,stats)
    progress(state)
    while any(not ds['converged'] and not ds.get('blocked') for ds in stats.values()):
        for name,ds in stats.items():
            if ds['converged'] or ds.get('blocked'):
                continue
            try:
                search_round(name,stats,state,source_signature)
            except Exception as error:
                traceback.print_exc()
                ds['failures']+=1
                ds['last_error']=repr(error)
                ds['blocked']=ds['failures']>=3
                state['errors']=[{'dataset':n,'error':s['last_error']}
                                 for n,s in stats.items() if s.get('last_error')]
                save_search(state,stats,source_signature)
                gc.collect()
                suite.torch.cuda.empty_cache()
    # Freeze every selected config before opening any held-out test scores.
    write({name:ds.get('best') for name,ds in stats.items()},OUT/'frozen_configs.json')
    for name,ds in stats.items():
        if ds.get('blocked'):
            continue
        try:
            evaluate_frozen(name,ds,state,source_signature)
        except Exception as error:
            traceback.print_exc()
            ds['final_error']=repr(error)
    failures={name:ds.get('last_error') or ds.get('final_error') for name,ds in stats.items()
              if ds.get('blocked') or ds.get('final_error')}
    progress(state,status='completed_with_errors' if failures else 'passed',
             stage='complete',completed=True,finished_at=now(),final_errors=failures)
    write(state,OUT/'completion.json')
    log('adaptive_complete',datasets=len(state['selected']),errors=failures)
    (OUT/'exit_code').write_text('1' if failures else '0')
    if failures:
        raise SystemExit(1)

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if not isinstance(error,SystemExit) or error.code:
            if (OUT/'progress.json').exists():
                failed=read(OUT/'progress.json')
                progress(failed,status='failed',stage='error',error=repr(error),completed=False)
            (OUT/'exit_code').write_text('1')
        raise
