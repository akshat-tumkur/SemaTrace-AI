"""Full-encoder FP16 adaptation; fixed data/objective, measured resource gate."""
from contextlib import nullcontext
from pathlib import Path
import hashlib,json,math,platform,random,time,shutil
import numpy as np
import torch,yaml
from training.data import CODE,digest,save,load_rows
from training.model import ReuseRoberta,weighted_loss
from training.engine import seed,resource,save_best,load_best
from training.metrics import select
from .data import frozen,verify_selection,unopened,check_ancestor
from .cache import Encoded,batches

def configuration(root,path=None):
    config=yaml.safe_load(Path(path or CODE/'configs/roberta_practical.yaml').read_text())
    if (root/'subset/selection.json').exists():
        config['positive_weight']=verify_selection(root,config)['positive_weight']
    return config

def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)

def environment():
    import importlib.metadata as md
    names=('torch','transformers','tokenizers','numpy','scikit-learn','scipy','PyYAML','onnx','onnxruntime','safetensors','huggingface-hub')
    return {'python':platform.python_version(),'packages':{name:md.version(name) for name in names}}

def identity(root,config):
    files=sorted(Path(__file__).parent.glob('*.py'))+[CODE/'backend/training'/n for n in ('model.py','metrics.py','data.py','engine.py','evaluation.py')]
    return {'configuration':config,'prepared_split_hashes':{s:r['sha256'] for s,r in frozen(root)['splits'].items()},
        'selection_sha256':digest(root/'subset/selection.json'),'tokens_sha256':digest(root/'tokens/manifest.json'),
        'code_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest(),'environment':environment()}

def initialize(config):
    from transformers import RobertaModel
    from .cache import tokenizer
    tok=tokenizer(config)
    encoder=RobertaModel.from_pretrained(config['model'],revision=config['revision'],add_pooling_layer=False,trust_remote_code=False,use_safetensors=True)
    encoder.resize_token_embeddings(len(tok),mean_resizing=False)
    if config['gradient_checkpointing']:encoder.gradient_checkpointing_enable()
    else:encoder.gradient_checkpointing_disable()
    model=ReuseRoberta(encoder)
    assert all(p.requires_grad for p in model.roberta.parameters())
    return model,tok

def autocast(device):
    return torch.amp.autocast('cuda',dtype=torch.float16) if device=='cuda' else nullcontext()

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

def save_checkpoint(root,state,name='last'):
    """Two slots: the old pointer stays usable until the new hash is committed."""
    folder=root/'checkpoints';folder.mkdir(exist_ok=True)
    pointer=root/(name+'.json');old=json.loads(pointer.read_text()) if pointer.exists() else None
    slot=1-old['slot'] if old else 0
    path=folder/(name+'-'+str(slot)+'.pt');temporary=path.with_suffix('.tmp')
    torch.save(state,temporary);checksum=digest(temporary);temporary.replace(path)
    atomic(pointer,{'slot':slot,'file':path.name,'sha256':checksum})
    return pointer

def load_checkpoint(root,name='last'):
    value=json.loads((root/(name+'.json')).read_text())
    expected=name+'-'+str(value['slot'])+'.pt'
    if value['file']!=expected:raise ValueError('Invalid checkpoint filename')
    path=root/'checkpoints'/expected
    if digest(path)!=value['sha256']:raise ValueError('Checkpoint integrity failure; original slots retained')
    return torch.load(path,map_location='cpu',weights_only=False) # Only own trusted optimizer/RNG checkpoints.

def update(model,optimizer,scaler,dataset,ids,config,device='cuda',probe_gradient=False):
    labels=dataset.labels[np.asarray(ids,dtype=int)]
    denominator=float(np.where(labels==1,config['positive_weight'],1.).sum())
    before_rng=rng_state();retries=0
    while True:
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for start in range(0,len(ids),config['micro_batch_size']):
            inputs,truth=dataset.collate(ids[start:start+config['micro_batch_size']],config['pad_to_multiple_of'])
            inputs={k:v.to(device) for k,v in inputs.items()};truth=truth.to(device)
            with autocast(device):
                logits=model(**inputs)
                loss=weighted_loss(logits.float(),truth,config['positive_weight'],denominator)
            scaler.scale(loss).backward();losses.append(loss.detach())
        loss=float(torch.stack(losses).sum())
        if not math.isfinite(loss):raise ValueError('Nonfinite loss; no successful training result')
        scale=scaler.get_scale();scaler.unscale_(optimizer)
        gradient=float(model.roberta.encoder.layer[0].attention.self.query.weight.grad.abs().sum()) if probe_gradient else None
        scaler.step(optimizer);scaler.update()
        if scaler.get_scale()>=scale:return {'loss':loss,'gradient_l1':gradient,'amp_overflow_retries':retries}
        retries+=1
        if retries>3:raise RuntimeError('Repeated AMP overflow; preserve pilot/checkpoint for inspection')
        restore_rng(before_rng) # Same batch/dropout, smaller scale; never silently skip examples.

@torch.no_grad()
def predict(model,dataset,config,device='cuda',indices=None):
    model.eval();indices=np.arange(len(dataset)) if indices is None else np.asarray(indices)
    scores=[];started=time.perf_counter()
    for start in range(0,len(indices),config['evaluation_batch_size']):
        batch,_=dataset.collate(indices[start:start+config['evaluation_batch_size']],config['pad_to_multiple_of'])
        with autocast(device):logits=model(**{k:v.to(device) for k,v in batch.items()})
        scores.extend(logits.float().softmax(-1)[:,1].cpu().tolist())
    if device=='cuda':torch.cuda.synchronize()
    return np.asarray(scores),time.perf_counter()-started

def pilot_indices(selection,seed,count):
    """Largest-remainder stratum allocation, then equal length-quartile coverage."""
    groups={}
    for name in {e['stratum'] for e in selection['entries']}:
        groups[name]=[i for i,e in enumerate(selection['entries']) if e['stratum']==name]
    quotas={s:len(v)*count//selection['examples'] for s,v in groups.items()}
    remaining=count-sum(quotas.values())
    for s in sorted(groups,key=lambda s:(-(len(groups[s])*count%selection['examples']),s))[:remaining]:quotas[s]+=1
    rng=np.random.default_rng(seed);chosen=[]
    for name in sorted(groups):
        for q in range(4):
            pool=[i for i in groups[name] if selection['entries'][i]['length_quartile']==q]
            n=quotas[name]//4+int(q<quotas[name]%4)
            chosen.extend(rng.permutation(pool)[:n].tolist())
    return np.asarray(chosen,dtype=int)

def pilot(root,config):
    unopened(root)
    if not torch.cuda.is_available():raise RuntimeError('Manual free-GPU runtime required; no local CPU full pilot')
    current=identity(root,config)
    if (root/'pilot.json').exists():
        if json.loads((root/'pilot.json').read_text())['identity']!=current:raise ValueError('Pilot identity changed')
        print((root/'pilot.json').read_text());return
    seed(config['seed']);dataset=Encoded(root/'tokens/train.npz');selection=verify_selection(root,config)
    model,tok=initialize(config);model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats()
    warm=config['pilot_warmup_steps']*config['effective_batch_size'];measured=config['pilot_measured_steps']*config['effective_batch_size']
    ids=pilot_indices(selection,config['seed']+17,warm+measured)
    # One representative draw, length grouping identical to the real epoch.
    order=batches(dataset.lengths[ids],config['seed']+18,config['effective_batch_size'],config['length_group_size'])
    steps=[];gradients=[];losses=[];overflows=0;measured_ids=[]
    for at,local in enumerate(order):
        chosen=ids[local];torch.cuda.synchronize();started=time.perf_counter()
        row=update(model,optimizer,scaler,dataset,chosen,config,probe_gradient=True);torch.cuda.synchronize()
        if not row['gradient_l1']>0:raise ValueError('Zero transformer gradient')
        if at>=config['pilot_warmup_steps']:
            steps.append(time.perf_counter()-started);measured_ids.extend(chosen.tolist())
        gradients.append(row['gradient_l1']);losses.append(row['loss']);overflows+=row['amp_overflow_retries']
    probe=pilot_indices(selection,config['seed']+99,256)
    before,inference_seconds=predict(model,dataset,config,indices=probe)
    state={'identity':current,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'cursor':measured}
    torch.cuda.synchronize();started=time.perf_counter();save_checkpoint(root,state,'pilot-recovery')
    checkpoint_write_seconds=time.perf_counter()-started;restored=load_checkpoint(root,'pilot-recovery')
    assert restored['identity']==current
    model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer']);scaler.load_state_dict(restored['scaler']);restore_rng(restored['rng'])
    after,_=predict(model,dataset,config,indices=probe)
    np.testing.assert_allclose(after,before,atol=1e-5,rtol=1e-4)
    updates=len(batches(dataset.lengths,config['seed'],config['effective_batch_size'],config['length_group_size']))*config['epochs']
    train_seconds=float(np.mean(steps))*updates
    conservative_train_seconds=float(np.quantile(steps,.9))*updates
    validation_seconds=inference_seconds/len(probe)*len(Encoded(root/'tokens/val.npz'))*config['epochs']
    checkpoint_seconds=math.ceil(conservative_train_seconds/config['checkpoint_seconds'])*checkpoint_write_seconds+config['epochs']*checkpoint_write_seconds
    projected=(conservative_train_seconds+validation_seconds+checkpoint_seconds)/3600
    value={'identity':current,'scope':'Training-only representative throughput; no validation/final predictions or accuracy claim.',
        'gpu':torch.cuda.get_device_name(),'warmup_examples':warm,'measured_examples':measured,'measured_optimizer_steps':len(steps),
        'measured_positive_examples':int(dataset.labels[measured_ids].sum()),
        'measured_strata':dict(__import__('collections').Counter(selection['entries'][i]['stratum'] for i in measured_ids)),
        'measured_source_families':len({selection['entries'][i]['source_family'] for i in measured_ids}),
        'measured_token_length_quantiles':np.quantile(dataset.lengths[measured_ids],[0,.25,.5,.75,.9,1]).tolist(),
        'strata_and_length_quartiles_represented':True,'initial_loss':losses[0],'final_loss':losses[-1],
        'nonzero_transformer_gradients':all(g>0 for g in gradients),'checkpoint_reload_verified':True,'scaler_optimizer_rng_reloaded':True,
        'seconds_per_update':steps,'examples_per_second':measured/sum(steps),'estimated_training_hours':train_seconds/3600,
        'p90_step_projected_training_hours':conservative_train_seconds/3600,'validation_hours_training_probe_estimate':validation_seconds/3600,
        'checkpoint_write_seconds':checkpoint_write_seconds,'checkpoint_hours_estimate':checkpoint_seconds/3600,
        'estimated_training_plus_validation_checkpoints_hours':projected,'within_resource_budget':projected<=config['maximum_estimated_training_hours'],
        'amp_overflow_retries':overflows,'resources':resource(),'final_and_baseline_and_export_time_included':False}
    save(root/'pilot.json',value);print(json.dumps(value,indent=2),flush=True)
    # Discard only this disposable pilot's own weights. Full training resets pretrained initialization.
    for path in [root/'pilot-recovery.json',root/'checkpoints/pilot-recovery-0.pt',root/'checkpoints/pilot-recovery-1.pt']:
        if path.exists():path.unlink()

def train(root,config):
    unopened(root);current=identity(root,config)
    p=json.loads((root/'pilot.json').read_text())
    if p['identity']!=current or not p['within_resource_budget']:raise RuntimeError('A matching representative GPU pilot within the4-hour budget is required; do not silently shrink data/epochs')
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    if (root/'training-complete.json').exists():print('All configured epochs already complete');return
    seed(config['seed']);model,tok=initialize(config);model.cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate']);scaler=torch.amp.GradScaler('cuda')
    dataset=Encoded(root/'tokens/train.npz');validation=Encoded(root/'tokens/val.npz')
    progress={'epoch':0,'cursor':0,'step':0,'best':-1.,'best_epoch':None,'history':[],'training_seconds':0.,'overflow_retries':0}
    if (root/'last.json').exists():
        state=load_checkpoint(root)
        if state['identity']!=current:raise ValueError('Resume data/config/code/environment mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        progress=state['progress'];restore_rng(state['rng'])
    started=time.perf_counter();last_save=started
    def persist():
        save_checkpoint(root,{'identity':current,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scaler':scaler.state_dict(),'rng':rng_state(),'progress':progress})
        atomic(root/'progress.json',{**progress,'resources':resource()})
    while progress['epoch']<config['epochs']:
        order=batches(dataset.lengths,config['seed']+progress['epoch'],config['effective_batch_size'],config['length_group_size'])
        while progress['cursor']<len(order):
            if progress['training_seconds']>=config['maximum_estimated_training_hours']*3600:
                persist();raise RuntimeError('Actual training budget exhausted; partial checkpoints preserved. No automatic new campaign.')
            ids=order[progress['cursor']];t=time.perf_counter()
            row=update(model,optimizer,scaler,dataset,ids,config)
            progress['training_seconds']+=time.perf_counter()-t;progress['cursor']+=1;progress['step']+=1;progress['overflow_retries']+=row['amp_overflow_retries']
            if progress['step']%50==0:print(json.dumps({'epoch':progress['epoch']+1,'step':progress['step'],'loss':row['loss'],'training_hours':progress['training_seconds']/3600}),flush=True)
            now=time.perf_counter()
            if now-last_save>=config['checkpoint_seconds'] or now-started>=config['session_seconds']:
                persist();last_save=time.perf_counter()
            if now-started>=config['session_seconds']:print('Private checkpoint saved; rerun train to resume.');return
        scores,seconds=predict(model,validation,config);choice=select(load_rows(root,'val'),scores,config)
        progress['history'].append({'epoch':progress['epoch']+1,'validation':choice,'inference_seconds':seconds})
        if choice['research']['sentence']['f1']>progress['best']:
            progress['best']=choice['research']['sentence']['f1'];progress['best_epoch']=progress['epoch']+1
            epoch_root=root/'selected-epochs'/str(progress['best_epoch']);epoch_root.mkdir(parents=True,exist_ok=True)
            save_best(epoch_root,model,tok,config)
            save(root/'best-selection.json',{'epoch':progress['epoch']+1,'identity':current,'validation':choice})
        progress['epoch']+=1;progress['cursor']=0;persist()
        atomic(root/'validation-history.json',progress['history'])
    # Immutable epoch artifacts avoid coupling a new best file to an older recovery pointer.
    selected=root/'selected-epochs'/str(progress['best_epoch'])/'best';destination=root/'best';destination.mkdir(exist_ok=True)
    for path in selected.iterdir():
        if path.is_file():shutil.copyfile(path,destination/path.name)
    if digest(destination/'model.pt')!=json.loads((destination/'identity.json').read_text())['weights_sha256']:raise ValueError('Selected checkpoint copy failed')
    save(root/'training-complete.json',{'identity':current,'progress':progress,'resources':resource()})
