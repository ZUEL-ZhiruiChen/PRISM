"""Verify single GPU execution, trial construction, and held-out test isolation."""
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
import a800_tuning as tuning

s = tuning.suite
assert s.torch.cuda.device_count() == 1 and s.DEVICE.type == 'cuda'
assert 'A800' in s.torch.cuda.get_device_name(0)
rng = s.np.random.default_rng(6)
x = rng.normal(size=(70, 12)).astype(s.np.float32)
x /= s.np.linalg.norm(x, axis=1, keepdims=True)
data = object.__new__(s.Inputs)
data.name = 'preflight'
data.x = s.torch.from_numpy(x)
data.y = s.torch.arange(70) % 2
data.adj = s.sp.csr_matrix((s.np.ones(140, dtype=s.np.float32),
            (s.np.repeat(s.np.arange(70), 2),
             s.np.column_stack([(s.np.arange(70)-1)%70,
                               (s.np.arange(70)+1)%70]).reshape(-1))), shape=(70,70))
data.x_gpu, data.y_gpu = data.x.to(s.DEVICE), data.y.to(s.DEVICE)
data.contexts = {}
data.meta = {'metric':'auroc'}
data.raw = {key:s.np.zeros((1,70),dtype=bool)
            for key in ('train_mask','val_mask','test_mask')}
data.raw['train_mask'][0,:30] = True
data.raw['val_mask'][0,30:50] = True
data.raw['test_mask'][0,50:] = True
spec = copy.deepcopy(tuning.candidate_grid('actor')[0])
spec['config']['model'].update(hidden_dim=16,proj_dim=8,n_clusters=5,knn_k=4,
                              structural_cap=6,structural_candidate_cap=6,densify_k=2)
spec['config']['loss']['distill_warmup_epochs'] = 1
events = []
with tempfile.TemporaryDirectory() as folder, patch.object(s,'CACHE',Path(folder)):
    s.EPOCH_CALLBACK = events.append
    a = s.fit(data,'prism',spec,split=0,init=44,final=False,epochs=3)
    context = data.contexts['prism']
    assert context['filter_stats']['densify_k'] == 2
    assert len(context['ops']) == 3
    data.y_gpu = data.y_gpu.clone()
    data.y_gpu[50:] = 1-data.y_gpu[50:]
    b = s.fit(data,'prism',spec,split=0,init=44,final=False,epochs=3)
    s.EPOCH_CALLBACK = None
    assert a['best_val_score'] == b['best_val_score']
    assert a['best_epoch'] == b['best_epoch']
    assert a['test_evaluations'] == b['test_evaluations'] == 0
    assert len(events) == 6
    assert all('test' not in key for row in a['history'] for key in row)
    assert all(s.math.isfinite(row['loss']) for row in a['history'])
print('PREFLIGHT PASSED: one A800, changed construction applied, epoch callback, '
      'warmup and distillation gradients, test-label permutation isolation',flush=True)
