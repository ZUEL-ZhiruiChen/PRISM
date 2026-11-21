"""Meaningful protocol and search checks; GPU checks run only under the GPU lock."""
import copy
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import adaptive_space as space

def fixture():
    return {'model':{'hidden_dim':128,'proj_dim':64,'K':2,'dropout':.4,
                    'private_gate_strength':.5,'gate_temperature':1.5,'gate_residual':.1,
                    **space.GRAPH_PROFILES[0]},
            'loss':{'lambda_align':.03,'lambda_var':.005,'lambda_decorrelation':.005,
                    'distill_warmup_epochs':5,'distill_temperature':1.,
                    'variance_gamma':1.},
            'optim':{'lr':.003,'weight_decay':.0005}}

class SearchChecks(unittest.TestCase):
    def test_reproducible_resume_and_no_repeats(self):
        parent=space.normalize(fixture())
        first=space.candidates('actor',0,[parent],[],8)
        self.assertEqual(first,space.candidates('actor',0,[parent],[],8))
        keys={space.fingerprint(cfg) for cfg in first}
        self.assertEqual(len(keys),8)
        second=space.candidates('actor',1,[parent],keys,8)
        self.assertFalse(keys & {space.fingerprint(cfg) for cfg in second})
        self.assertEqual(parent,space.normalize(fixture()))

    def test_plateau_reset_has_meaningful_threshold(self):
        stale=0
        for old,new in [(.7,.7001),(.7001,.7002),(.7002,.7003)]:
            stale=space.update_plateau(old,new,stale)
        self.assertEqual(stale,3)
        self.assertEqual(space.update_plateau(.7,.701,3),0)
        self.assertEqual(space.update_plateau(.7,.69,0),1)

    def test_configs_keep_prism_and_valid_construction(self):
        rng=random.Random(14)
        for _ in range(100):
            cfg=space.mutate(fixture(),rng,global_search=True)
            self.assertTrue(cfg['model']['dual_branch'])
            self.assertTrue(cfg['model']['structure_filter'])
            self.assertTrue(cfg['loss']['distill_enabled'])
            self.assertEqual(len(cfg['model']['enabled_views']),3)
            if cfg['model']['gate_mode']=='reliability':
                self.assertGreaterEqual(cfg['loss']['lambda_view_ce'],.05)

def gpu_checks(suite,adaptive):
    assert suite.torch.cuda.device_count()==1 and suite.DEVICE.type=='cuda'
    rng=suite.np.random.default_rng(8)
    x=rng.normal(size=(70,12)).astype(suite.np.float32)
    x/=suite.np.linalg.norm(x,axis=1,keepdims=True)
    inputs=object.__new__(suite.Inputs)
    inputs.name='preflight'
    inputs.x=suite.torch.from_numpy(x)
    inputs.y=suite.torch.arange(70)%2
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
    inputs.meta={'metric':'auroc','sha256':'preflight'}
    inputs.num_splits=1
    cfg=fixture()
    cfg['model'].update(hidden_dim=16,proj_dim=8,n_clusters=5,knn_k=4,
                        structural_cap=6,structural_candidate_cap=6,densify_k=2)
    cfg['loss'].update(distill_warmup_epochs=1,lambda_view_ce=.2)
    parent=adaptive.OUT/'preflight'
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temp:
        with patch.object(adaptive,'OUT',Path(temp)),patch.object(suite,'CACHE',Path(temp)/'cache'),\
             patch.object(adaptive,'EPOCHS',3),patch.object(adaptive,'PATIENCE',10):
            fitter=adaptive.Fitter(inputs,{},'preflight')
            a=fitter.fit(cfg,0,7,'preflight',persist=False)
            assert next(iter(fitter.contexts.values()))['filter_stats']['densify_k']==2
            inputs.y_gpu=inputs.y_gpu.clone()
            inputs.y_gpu[50:]=1-inputs.y_gpu[50:]
            b=fitter.fit(cfg,0,7,'preflight',persist=False)
            assert a['best_val_score']==b['best_val_score'] and a['best_epoch']==b['best_epoch']
            assert a['test_evaluations']==b['test_evaluations']==0
            assert all('test' not in key for row in a['history'] for key in row)
            assert suite.np.allclose([r['loss'] for r in a['history']],
                                     [r['loss'] for r in b['history']],atol=2e-5)
            reliability=copy.deepcopy(cfg)
            reliability['model']['gate_mode']='reliability'
            c=fitter.fit(reliability,0,7,'preflight',persist=False)
            assert len(c['history'])==3 and all(math.isfinite(r['loss']) for r in c['history'])
            final=fitter.fit(cfg,0,8,'preflight',final=True,persist=True)
            assert final['test_evaluations']==1 and Path(final['checkpoint']).exists()
            cached=fitter.fit(cfg,0,8,'preflight',final=True,persist=True)
            assert cached==final
            import prepare_data
            controller_root=Path(temp)/'controller'
            controller_cfg=space.normalize(cfg)
            stats={'preflight':{'rounds':0,'stale_rounds':0,'seen':[],
                               'confirmed':[],'converged':False,'failures':0}}
            state={}
            with patch.object(adaptive,'OUT',controller_root),\
                 patch.object(suite,'Inputs',return_value=inputs),\
                 patch.object(prepare_data,'prepare',return_value=None),\
                 patch.object(adaptive,'initial_parents',return_value=[controller_cfg]),\
                 patch.object(space,'candidates',return_value=[controller_cfg]):
                for _ in range(4):
                    adaptive.search_round('preflight',stats,state,'preflight-controller')
                assert stats['preflight']['rounds']==4
                assert stats['preflight']['stale_rounds']==3
                assert stats['preflight']['converged']
                saved=adaptive.read(controller_root/'search_state.json')
                assert saved['datasets']['preflight']['converged']
                for record in controller_root.glob('preflight/fits/*/*.json'):
                    row=adaptive.read(record)
                    assert row['test_evaluations']==0 and 'best_test_score' not in row
    print('ADAPTIVE GPU PREFLIGHT PASSED: one A800, warmup/distillation, reliability gate, '
          'held-out label isolation, checkpoints, resumable results and multi-round plateau stopping',flush=True)

if __name__=='__main__':
    unittest.main()
