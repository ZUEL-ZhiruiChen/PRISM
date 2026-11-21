"""Check new training paths and validation-only selection on a small GPU graph."""
import copy
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import refine_space as space
from check_adaptive import fixture


class SearchChecks(unittest.TestCase):
    def test_resume_reproducible_without_duplicates(self):
        parent=space.normalize(fixture())
        original=copy.deepcopy(parent)
        first=space.candidates('actor',0,[parent],[],12)
        self.assertEqual(first,space.candidates('actor',0,[parent],[],12))
        seen={space.fingerprint(cfg) for cfg in first}
        second=space.candidates('actor',1,[parent],seen,12)
        self.assertEqual(len(seen),12)
        self.assertFalse(seen & {space.fingerprint(cfg) for cfg in second})
        self.assertEqual(parent,original)

    def test_expanded_search_can_reach_new_options(self):
        rng=random.Random(14)
        configs=[space.mutate(fixture(),rng,True) for _ in range(200)]
        self.assertEqual({cfg['optim']['schedule'] for cfg in configs},{'none','cosine','plateau'})
        self.assertIn(1200,{cfg['train']['epochs'] for cfg in configs})
        self.assertIn(8,{cfg['model']['K'] for cfg in configs})
        self.assertEqual({cfg['model']['refine'] for cfg in configs},{True,False})
        for cfg in configs:
            self.assertEqual(cfg['train']['selection'],'val')
            self.assertTrue(cfg['model']['dual_branch'] and cfg['model']['structure_filter'])
            self.assertTrue(cfg['loss']['distill_enabled'])
            self.assertEqual(len(cfg['model']['enabled_views']),3)
            if cfg['model']['gate_mode']=='reliability':
                self.assertGreaterEqual(cfg['loss']['lambda_view_ce'],.05)

    def test_smaller_improvement_threshold(self):
        self.assertEqual(space.update_plateau(.7,.7001,2,.0002),3)
        self.assertEqual(space.update_plateau(.7,.7003,4,.0002),0)


def gpu_checks(refine,suite):
    assert suite.torch.cuda.device_count()==1 and suite.DEVICE.type=='cuda'
    torch=suite.torch
    rng=suite.np.random.default_rng(8)
    x=rng.normal(size=(70,12)).astype(suite.np.float32)
    x/=suite.np.linalg.norm(x,axis=1,keepdims=True)
    inputs=object.__new__(suite.Inputs)
    inputs.name='preflight'
    inputs.x=torch.from_numpy(x)
    inputs.y=torch.arange(70)%2
    inputs.y[:25]=0
    inputs.y[25:30]=1
    inputs.x_gpu,inputs.y_gpu=inputs.x.to(suite.DEVICE),inputs.y.to(suite.DEVICE)
    inputs.adj=suite.sp.csr_matrix((suite.np.ones(140,dtype=suite.np.float32),
                (suite.np.repeat(suite.np.arange(70),2),
                 suite.np.column_stack([(suite.np.arange(70)-1)%70,
                                       (suite.np.arange(70)+1)%70]).reshape(-1))),shape=(70,70))
    inputs.raw={key:suite.np.zeros((1,70),dtype=bool)
                for key in ('train_mask','val_mask','test_mask')}
    inputs.raw['train_mask'][0,:30]=True
    inputs.raw['val_mask'][0,30:50]=True
    inputs.raw['test_mask'][0,50:]=True
    inputs.meta={'metric':'auroc','sha256':'preflight-refine'}
    inputs.num_splits=1
    train=inputs.mask('train_mask',0)
    before=refine.class_weights(torch,inputs.y_gpu,train,2,.5)
    altered=inputs.y_gpu.clone()
    altered[~train]=1-altered[~train]
    assert torch.equal(before,refine.class_weights(torch,altered,train,2,.5))
    assert before[1]>before[0]
    cfg=space.normalize(fixture())
    cfg['model'].update(hidden_dim=16,proj_dim=8,n_clusters=5,knn_k=4,
                        structural_cap=6,structural_candidate_cap=6,densify_k=2)
    cfg['loss'].update(distill_warmup_epochs=1,lambda_view_ce=.2,class_weight_power=.5)
    cfg['train'].update(epochs=4,patience=20)
    parent=refine.OUT/'preflight'
    parent.mkdir(parents=True,exist_ok=True)
    engine=refine.engine
    with tempfile.TemporaryDirectory(dir=parent) as temp:
        with patch.object(engine,'OUT',Path(temp)),patch.object(suite,'CACHE',Path(temp)/'cache'):
            fitter=refine.RefineFitter(inputs,{},'preflight-refine')
            first=fitter.fit(cfg,0,7,'preflight',persist=False)
            inputs.y_gpu[50:]=1-inputs.y_gpu[50:]
            second=fitter.fit(cfg,0,7,'preflight',persist=False)
            assert first['best_val_score']==second['best_val_score']
            assert first['best_epoch']==second['best_epoch']
            assert suite.np.allclose([r['loss'] for r in first['history']],
                                     [r['loss'] for r in second['history']],atol=2e-5)
            assert first['class_weights']==second['class_weights']
            assert first['test_evaluations']==second['test_evaluations']==0
            assert all('test' not in key for row in first['history'] for key in row)
            assert all(row['lr']==cfg['optim']['lr'] for row in first['history'])
            cosine=copy.deepcopy(cfg)
            cosine['optim']['schedule']='cosine'
            cosine['model'].update(refine=True,K=6,hop_initialization='diffuse',
                                    hyperedge_score='coverage',prototype_soft_topk=1,
                                    gate_mode='reliability')
            c=fitter.fit(cosine,0,7,'preflight',persist=False)
            assert len(c['history'])==4 and c['history'][-1]['lr']<c['history'][0]['lr']
            assert all(math.isfinite(r['loss']) for r in c['history'])
            plateau=copy.deepcopy(cfg)
            plateau['optim']['schedule']='plateau'
            plateau['model']['hop_initialization']='self'
            plateau['train']['epochs']=14
            with patch.object(suite.run_node,'masked_score',return_value=.5):
                p=fitter.fit(plateau,0,7,'preflight',persist=False)
            assert p['history'][-1]['lr']<p['history'][0]['lr']
            assert p['test_evaluations']==0
            final=fitter.fit(cosine,0,8,'preflight',final=True,persist=True)
            assert final['test_evaluations']==1 and Path(final['checkpoint']).exists()
            checkpoint=torch.load(final['checkpoint'],map_location='cpu',weights_only=True)
            assert checkpoint['config']==space.normalize(cosine)
            assert checkpoint['best_epoch']==final['best_epoch']
            assert fitter.fit(cosine,0,8,'preflight',final=True,persist=True)==final
            # The validation audit must include the old incumbent and never test it.
            audit_cfg=space.normalize(cfg)
            stats={'rounds':1,'confirmed':[{'key':space.fingerprint(audit_cfg),
                    'config':audit_cfg,'validation_mean':.5}]}
            incumbent=copy.deepcopy(audit_cfg)
            incumbent['model']['hop_initialization']='local'
            with patch.object(refine,'OUT',Path(temp)),patch.object(suite,'Inputs',return_value=inputs),\
                 patch.object(refine,'initial_parents',return_value=[incumbent]):
                refine.audit_validation('preflight',stats,{},'preflight-refine')
            audit=engine.read(Path(temp)/'preflight/validation_audit.json')
            assert len(audit['candidates'])==2
            assert all(len(row['validation_scores'])==5 for row in audit['candidates'])
            assert audit['selected']['validation_mean']==max(r['validation_mean'] for r in audit['candidates'])
            for path in (Path(temp)/'preflight/fits').glob('*/*.json'):
                record=engine.read(path)
                assert record['test_evaluations']==int(record['init']==8)
    print('REFINE GPU PREFLIGHT PASSED: class weights isolate training labels; '
          'none/cosine/plateau schedules; refinement, hop initialization and reliability; '
          'held-out test-label isolation; checkpoint caching; five-run validation audit',flush=True)


if __name__=='__main__':
    unittest.main()
