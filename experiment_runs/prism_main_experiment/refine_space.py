"""Extended PRISM search: train longer, schedules, class weights and view geometry."""
import copy
import hashlib
import random
import adaptive_space as previous

GRAPH_PROFILES = previous.GRAPH_PROFILES + [
    dict(hop_k=1,structural_cap=8,structural_candidate_cap=16,knn_k=4,
         n_clusters=128,densify_k=1,densify_min_similarity=.4,hyperedge_keep_ratio=.9),
    dict(hop_k=2,structural_cap=64,structural_candidate_cap=64,knn_k=24,
         n_clusters=256,densify_k=2,densify_min_similarity=.4,hyperedge_keep_ratio=.9),
    dict(hop_k=3,structural_cap=32,structural_candidate_cap=48,knn_k=8,
         n_clusters=128,densify_k=2,densify_min_similarity=.1,hyperedge_keep_ratio=.6),
]
CHOICES = {**previous.CHOICES,
    'model.K':[1,2,3,4,6,8], 'model.refine':[False,True],
    'model.hop_initialization':['uniform','self','local','diffuse'],
    'model.hyperedge_score':['coherence_coverage','coherence','coverage'],
    'model.prototype_soft_topk':[1,2,3,5],
    'loss.class_weight_power':[0.,.5,1.],
    'loss.lambda_align':[.0003,.001,.003,.01,.03,.1,.2],
    'loss.lambda_var':[.0001,.001,.005,.02,.05],
    'loss.lambda_decorrelation':[.0001,.001,.005,.02,.05],
    'optim.schedule':['none','cosine','plateau'],
    'train.epochs':[400,800,1200], 'train.patience':[60,100,150]}
PRIORS = previous.PRIORS
fingerprint=previous.fingerprint
update_plateau=previous.update_plateau

def normalize(config):
    cfg=copy.deepcopy(config)
    m,loss=cfg['model'],cfg['loss']
    m.update(model_type='prism',dual_branch=True,structure_filter=True,
             enabled_views=['structural','feature_knn','prototype'],fusion_mode='embedding')
    m.setdefault('gate_mode','soft')
    m.setdefault('view_gate_prior',copy.deepcopy(PRIORS[0]))
    m.setdefault('refine',False)
    m.setdefault('hop_initialization','uniform')
    m.setdefault('hyperedge_score','coherence_coverage')
    m.setdefault('prototype_soft_topk',3)
    m['preserve_node_coverage']=True
    m['proj_dim']=min(m['hidden_dim'],m['proj_dim'])
    loss.update(lambda_elastic=0.,distill_enabled=True)
    loss.setdefault('lambda_view_ce',0.)
    loss.setdefault('label_smoothing',0.)
    loss.setdefault('class_weight_power',0.)
    if m['gate_mode']=='reliability':
        loss['lambda_view_ce']=max(.05,loss['lambda_view_ce'])
    cfg['optim'].setdefault('schedule','none')
    train=cfg.setdefault('train',{})
    train.setdefault('epochs',400)
    train.setdefault('patience',60)
    train['selection']='val'
    assert 1<=m['K']<=8 and 1<=m['knn_k']<=32
    assert 0<m['hyperedge_keep_ratio']<=1 and m['gate_temperature']>0
    assert m['hyperedge_score'] in ('coherence_coverage','coherence','coverage')
    assert cfg['optim']['schedule'] in ('none','cosine','plateau')
    assert train['epochs']>=1 and train['patience']>=1
    assert 0<=loss['class_weight_power']<=1
    return cfg

def mutate(config,rng,global_search=False):
    cfg=normalize(config)
    keys=list(CHOICES)
    for key in keys if global_search else rng.sample(keys,rng.randint(1,5)):
        section,name=key.split('.')
        cfg[section][name]=rng.choice(CHOICES[key])
    if global_search or rng.random()<.25:
        cfg['model'].update(rng.choice(GRAPH_PROFILES))
    if global_search or rng.random()<.2:
        cfg['model']['view_gate_prior']=copy.deepcopy(rng.choice(PRIORS))
    return normalize(cfg)

def candidates(dataset,round_index,parents,seen,batch_size=12):
    seed=int(hashlib.sha256(f'{dataset}:{round_index}:prism-refine-20261113'.encode()).hexdigest()[:12],16)
    rng=random.Random(seed)
    result,used=[],set(seen)
    for attempt in range(10000):
        if round_index==0 and attempt<len(parents):
            cfg=normalize(parents[attempt])
        else:
            cfg=mutate(rng.choice(parents[:3]),rng,len(result)%4==3 or attempt%17==16)
        key=fingerprint(cfg)
        if key in used:
            continue
        used.add(key)
        result.append(cfg)
        if len(result)==batch_size:
            return result
    raise RuntimeError('Search space exhausted')
