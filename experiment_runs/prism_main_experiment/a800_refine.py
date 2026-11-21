"""Second PRISM search campaign; previous runs remain immutable.

Uses validation-only population search, then a five-run validation audit of
the best candidates and the previous incumbent. Configs are frozen before
final tests. A maximum test value is descriptive and never selects a model.
"""
from __future__ import annotations
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']='0'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREVIOUS=ROOT/'server_runs/a800_adaptive_20261112'
OUT=ROOT/'server_runs/a800_refine_20261113'
sys.path.insert(0,str(HERE))
import a800_adaptive as engine
import refine_space as space

def class_weights(torch,labels,mask,classes,power):
    """Weights use training-label counts only; held-out labels have no effect."""
    if power==0:
        return None
    counts=torch.bincount(labels[mask],minlength=classes).float().clamp_min(1)
    weights=counts.pow(-power)
    return weights/weights.mean()

def initialize_hops(torch,model,mode):
    if mode=='uniform':
        return
    with torch.no_grad():
        for encoder in model.views:
            k=encoder.K
            weights=torch.ones(k+1,device=encoder.theta_logits.device)
            if mode=='self':
                weights.fill_(.1/k)
                weights[0]=.9
            elif mode=='local':
                weights.fill_(.1/max(1,k-1))
                weights[0]=.15
                weights[1]=.75
            elif mode=='diffuse':
                weights.fill_(.95/k)
                weights[0]=.05
            else:
                raise ValueError(mode)
            weights/=weights.sum()
            encoder.theta_logits.copy_(weights.log())

class RefineFitter(engine.Fitter):
    def fit(self,cfg,split,init,stage,final=False,persist=True):
        cfg=space.normalize(cfg)
        suite=engine.suite
        torch,F=suite.torch,suite.F
        key=space.fingerprint(cfg)
        path=engine.OUT/self.inputs.name/'fits'/key/f'split_{split}_init_{init}.json'
        if persist and path.exists():
            saved=engine.read(path)
            assert saved['source_signature']==self.source_signature
            assert saved['input_sha256']==self.inputs.meta['sha256']
            assert saved['config']==cfg and saved['test_evaluations']==int(final)
            return saved
        suite.seed_all(init)
        started=time.perf_counter()
        epochs,patience=cfg['train']['epochs'],cfg['train']['patience']
        engine.progress(self.state,dataset=self.inputs.name,stage=stage,candidate=key,
                        split=split,init=init,current_epoch=0,epochs_per_trial=epochs,
                        current_loss=None,current_stopping_score=None,
                        current_best_stopping_score=None)
        context=self.context(cfg)
        m,lc,optim=cfg['model'],cfg['loss'],cfg['optim']
        names=context['names']
        model=suite.PRISM(self.inputs.x.shape[1],m['hidden_dim'],m['proj_dim'],
                    int(self.inputs.y.max())+1,num_views=len(names),
                    K=m['K'],dropout=m['dropout'],dual_branch=True,
                    private_gate_strength=m.get('private_gate_strength',.5),
                    gate_temperature=m['gate_temperature'],gate_residual=m['gate_residual'],
                    gate_prior=[m['view_gate_prior'].get(name,1.) for name in names],
                    gate_mode=m['gate_mode'],refine=m['refine'],fusion_mode='embedding').to(suite.DEVICE)
        initialize_hops(torch,model,m['hop_initialization'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=optim['lr'],weight_decay=optim['weight_decay'])
        if optim['schedule']=='cosine':
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,
                                                                eta_min=optim['lr']*.02)
        elif optim['schedule']=='plateau':
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',
                         factor=.5,patience=max(10,patience//4),min_lr=optim['lr']*.01)
        else:
            scheduler=None
        x,y=self.inputs.x_gpu,self.inputs.y_gpu
        train=self.inputs.mask('train_mask',split)
        stopping=self.inputs.mask('stopping_mask' if 'stopping_mask' in self.inputs.raw else 'val_mask',split)
        validation=self.inputs.mask('val_mask',split)
        metric=self.inputs.meta['metric']
        weights=class_weights(torch,y,train,int(self.inputs.y.max())+1,lc['class_weight_power'])
        flags=suite.run_node.parse_variant('prism+soft_hg')
        pairs=torch.empty((2,0),dtype=torch.long,device=suite.DEVICE)
        bounds=x.new_empty(0)
        best,best_epoch,bad,best_state=-1.,0,0,None
        history=[]
        try:
            for epoch in range(1,epochs+1):
                model.train()
                out=model(x,context['ops'])
                ce=F.cross_entropy(out['logits'][train],y[train],weight=weights,
                                   label_smoothing=lc['label_smoothing'])
                view_ce=x.new_tensor(0.)
                if lc['lambda_view_ce']>0:
                    view_ce=sum(F.cross_entropy(logits[train],y[train],weight=weights,
                                    label_smoothing=lc['label_smoothing'])
                                 for logits in out['view_logits_list'])/len(out['view_logits_list'])
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
                    raise FloatingPointError('Nonfinite gradients')
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    score=suite.run_node.masked_score(model(x,context['ops'])['logits'],y,stopping,metric)
                row={'epoch':epoch,'loss':float(loss.detach()),'stopping_score':score,
                     'lr':optimizer.param_groups[0]['lr'],'ce':float(ce.detach()),
                     'view_ce':float(view_ce.detach()),'align':float(align.detach()),
                     'variance':float(var.detach()),'decorrelation':float(decor.detach())}
                history.append(row)
                if scheduler is not None:
                    scheduler.step(score) if optim['schedule']=='plateau' else scheduler.step()
                if score>best:
                    best,best_epoch,bad=score,epoch,0
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:
                    bad+=1
                engine.progress(self.state,stage=stage,current_epoch=epoch,current_loss=row['loss'],
                                current_stopping_score=score,current_best_stopping_score=best,
                                current_best_epoch=best_epoch,patience_used=bad,current_lr=row['lr'])
                engine.log('epoch',dataset=self.inputs.name,candidate=key,stage=stage,
                           split=split,init=init,**row)
                if bad>=patience:
                    break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                logits=model(x,context['ops'])['logits']
                val=suite.run_node.masked_score(logits,y,validation,metric)
                result={'best_val_score':val,'best_stopping_score':best,'best_epoch':best_epoch,
                        'history':history,'epochs_run':len(history),'test_evaluations':0,
                        'metric':metric,'config':cfg,'input_sha256':self.inputs.meta['sha256'],
                        'source_signature':self.source_signature,'split':split,'init':init,
                        'execution_device':str(suite.DEVICE),'filter_stats':context['filter_stats'],
                        'parameters':sum(p.numel() for p in model.parameters()),
                        'seconds':time.perf_counter()-started,'finished_at':engine.now(),
                        'epochs_limit':epochs,'class_weights':weights.cpu().tolist() if weights is not None else None}
                if final:
                    result['best_test_score']=suite.run_node.masked_score(logits,y,self.inputs.mask('test_mask',split),metric)
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
                engine.write(result,path)
            return result
        finally:
            del model,optimizer,best_state
            gc.collect()

def initial_parents(name):
    # Search history contains only validation scores, not final test records.
    previous=engine.read(PREVIOUS/'search_state.json')['datasets'][name]
    best=sorted(previous['confirmed'],key=lambda r:(-r['validation_mean'],r['key']))
    parents=[]
    for row in best[:3]:
        cfg=copy.deepcopy(row['config'])
        cfg['train'].update(epochs=400,patience=60)
        parents.append(space.normalize(cfg))
    incumbent=copy.deepcopy(parents[0])
    long=copy.deepcopy(incumbent)
    long['train'].update(epochs=1200,patience=150)
    parents.insert(1,long)
    for schedule in ('cosine','plateau'):
        cfg=copy.deepcopy(long)
        cfg['optim']['schedule']=schedule
        parents.append(space.normalize(cfg))
    deeper=copy.deepcopy(long)
    deeper['model']['refine']=True
    deeper['optim']['schedule']='plateau'
    parents.append(space.normalize(deeper))
    cfg=copy.deepcopy(long)
    if name in ('questions','tolokers'):
        cfg['loss']['class_weight_power']=.5
        cfg['optim']['schedule']='plateau'
    elif name in ('actor','amazon_ratings'):
        cfg['model'].update(hop_initialization='self',prototype_soft_topk=1)
        cfg['loss'].update(lambda_align=.001,lambda_var=.001,lambda_decorrelation=.001)
    else:
        cfg['model'].update(hop_initialization='diffuse',K=6)
        cfg['optim']['schedule']='plateau'
    parents.append(space.normalize(cfg))
    return parents

def source_signature():
    paths=[ROOT/name for name in ('run_node.py','models/model.py','models/losses.py',
                                 'models/hypergraph.py','utils/data_node.py')]
    paths += [HERE/name for name in ('a800_refine.py','refine_space.py','check_refine.py',
                'a800_adaptive.py','adaptive_space.py','a800_suite.py','a800_tuning.py','prepare_data.py')]
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes

def audit_validation(name,ds,state,signature):
    suite=engine.suite
    inputs=suite.Inputs(name)
    fitter=RefineFitter(inputs,state,signature)
    configs=[row['config'] for row in sorted(ds['confirmed'],key=lambda r:-r['validation_mean'])[:3]]
    configs.append(initial_parents(name)[0])
    records=[]
    unique={space.fingerprint(cfg):cfg for cfg in configs}
    for key,cfg in unique.items():
        scores=[]
        for index in range(5):
            result=fitter.fit(cfg,min(index,inputs.num_splits-1),41+index,'validation_audit')
            assert result['test_evaluations']==0
            scores.append(result['best_val_score'])
        records.append({'key':key,'config':cfg,'validation_scores':scores,
                        'validation_mean':suite.mean(scores)})
    winner=min(records,key=lambda r:(-r['validation_mean'],r['key']))
    ds['best']=copy.deepcopy(winner)
    ds['validation_audit']=records
    engine.write({'selected':winner,'candidates':records},OUT/name/'validation_audit.json')
    fitter.contexts.clear()
    del fitter,inputs
    gc.collect()
    suite.torch.cuda.empty_cache()

def main():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'launcher.pid').write_text(str(os.getpid()))
    state={'status':'waiting','phase':'refine','stage':'waiting_for_first_pass','started_at':engine.now(),
           'datasets':engine.DATASETS,'selected':[],'trials':[],'epochs_per_trial':1200,
           'device':'cuda:0','convergence_rule':{'rounds_without_gain':5,'minimum_gain':.0002,'batch_size':12},
           'selection_protocol':'validation-only search; five-run validation audit including previous incumbent; configs frozen before testing; test maxima descriptive only'}
    engine.OUT=OUT
    engine.space=space
    engine.Fitter=RefineFitter
    engine.initial_parents=initial_parents
    engine.BATCH_SIZE,engine.PLATEAU_ROUNDS,engine.MIN_DELTA=12,5,.0002
    engine.progress(state)
    lock=(engine.FIRST/'campaign.lock').open('a')
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(10)
    import a800_suite as suite
    engine.suite=suite
    assert suite.DEVICE.type=='cuda' and suite.torch.cuda.device_count()==1
    assert 'A800' in suite.torch.cuda.get_device_name(0)
    engine.progress(state,status='running',stage='preflight',gpu=suite.torch.cuda.get_device_name(0))
    signature,hashes=source_signature()
    from check_refine import gpu_checks
    gpu_checks(sys.modules[__name__],suite)
    engine.write({'source_signature':signature,'source_hashes':hashes,
                  'search_space':space.CHOICES,'graph_profiles':space.GRAPH_PROFILES,
                  'selection':state['selection_protocol'],'convergence':state['convergence_rule'],
                  'training_class':'models.model.PRISM','device':'cuda:0'},OUT/'protocol.json')
    if (OUT/'search_state.json').exists():
        saved=engine.read(OUT/'search_state.json')
        assert saved['source_signature']==signature
        stats=saved['datasets']
    else:
        stats={name:{'rounds':0,'stale_rounds':0,'seen':[],'confirmed':[],
                     'converged':False,'failures':0} for name in engine.DATASETS}
    while any(not ds['converged'] and not ds.get('blocked') for ds in stats.values()):
        for name,ds in stats.items():
            if ds['converged'] or ds.get('blocked'):
                continue
            try:
                engine.search_round(name,stats,state,signature)
            except Exception as error:
                traceback.print_exc()
                ds['failures']+=1
                ds['last_error']=repr(error)
                ds['blocked']=ds['failures']>=3
                state['errors']=[{'dataset':n,'error':s['last_error']} for n,s in stats.items() if s.get('last_error')]
                engine.save_search(state,stats,signature)
                gc.collect()
                suite.torch.cuda.empty_cache()
    # The audit uses only validation records and completes for all datasets before testing.
    for name,ds in stats.items():
        if ds.get('blocked'):
            continue
        try:
            audit_validation(name,ds,state,signature)
        except Exception as error:
            traceback.print_exc()
            ds['blocked']=True
            ds['last_error']=repr(error)
        engine.save_search(state,stats,signature)
    engine.write({name:ds.get('best') for name,ds in stats.items()},OUT/'frozen_configs.json')
    for name,ds in stats.items():
        if ds.get('blocked'):
            continue
        try:
            engine.evaluate_frozen(name,ds,state,signature)
            final=engine.read(OUT/name/'final_summary.json')
            scores=[r['best_test_score'] for r in final['runs']]
            final['summary']['single_run_test_max']=max(scores)
            final['summary']['single_run_max_note']='Descriptive maximum of the fixed validation-selected config; not the method mean or evidence of SOTA'
            engine.write(final,OUT/name/'final_summary.json')
            state['selected'][-1].update(final['summary'])
            engine.progress(state)
        except Exception as error:
            traceback.print_exc()
            ds['final_error']=repr(error)
    failures={name:ds.get('last_error') or ds.get('final_error') for name,ds in stats.items()
              if ds.get('blocked') or ds.get('final_error')}
    engine.progress(state,status='completed_with_errors' if failures else 'passed',
                    stage='complete',completed=True,finished_at=engine.now(),final_errors=failures)
    engine.write(state,OUT/'completion.json')
    (OUT/'exit_code').write_text('1' if failures else '0')
    engine.log('refine_complete',datasets=len(state['selected']),errors=failures)
    if failures:
        raise SystemExit(1)

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if not isinstance(error,SystemExit) or error.code:
            if (OUT/'progress.json').exists():
                state=engine.read(OUT/'progress.json')
                engine.progress(state,status='failed',stage='error',error=repr(error))
            (OUT/'exit_code').write_text('1')
        raise
