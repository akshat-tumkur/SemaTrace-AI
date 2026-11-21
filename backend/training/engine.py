"""GPU campaign with deterministic batch-boundary recovery and sealed testing."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from pathlib import Path
import json,time,random,sys,platform
import numpy as np
import torch
import yaml
from .data import CODE,digest,save,verify,load_rows
from .model import initialize,PairDataset,collate,weighted_loss,ReuseRoberta
from .metrics import metrics,select

def seed(value):
    random.seed(value);np.random.seed(value%(2**32));torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False

def configuration(path=None):return yaml.safe_load(Path(path or CODE/'configs/roberta_training.yaml').read_text(encoding='utf-8'))
def identity(root,config):
    import hashlib
    files=sorted(Path(__file__).parent.glob('*.py'))
    return {'data_sha256':digest(root/'prepared/manifest.json'),'configuration':config,
        'code_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()}
def unopened(root):
    if (root/'test-opened.json').exists():raise RuntimeError('Final evaluation exposed; further training/selection is prohibited')
def resource():
    if sys.platform!='win32':
        import resource as rs
    return {'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
        'peak_rss_mib':rs.getrusage(rs.RUSAGE_SELF).ru_maxrss/1024 if sys.platform!='win32' else None,
        'python':platform.python_version(),'torch':torch.__version__}

@torch.no_grad()
def predict(model,tokenizer,records,config,device):
    model.eval();dataset=PairDataset(records);values=[];began=time.perf_counter()
    for at in range(0,len(dataset),config['evaluation_batch_size']):
        batch,_=collate([dataset[i] for i in range(at,min(len(dataset),at+config['evaluation_batch_size']))],tokenizer,config['max_length'])
        logits=model(**{k:v.to(device) for k,v in batch.items()})
        values.extend(logits.softmax(-1)[:,1].cpu().tolist())
    return np.asarray(values),time.perf_counter()-began

def save_best(root,model,tokenizer,config):
    dest=root/'best';dest.mkdir(exist_ok=True)
    path=dest/'model.pt';temporary=dest/'model.tmp'
    torch.save(model.state_dict(),temporary);temporary.replace(path)
    model.roberta.config.save_pretrained(dest);tokenizer.save_pretrained(dest)
    save(dest/'identity.json',{'weights_sha256':digest(path),'configuration':config})

def load_best(root,device):
    from transformers import RobertaConfig,RobertaModel,AutoTokenizer
    dest=root/'best';meta=json.loads((dest/'identity.json').read_text())
    if digest(dest/'model.pt')!=meta['weights_sha256']:raise ValueError('Selected weights changed')
    model=ReuseRoberta(RobertaModel(RobertaConfig.from_pretrained(dest),add_pooling_layer=False))
    model.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
    return model.to(device),AutoTokenizer.from_pretrained(dest),meta

def train(root,config):
    verify(root);unopened(root)
    if not torch.cuda.is_available():raise RuntimeError('Full training requires the manually selected free GPU runtime; no CPU full training')
    if not (root/'pilot.json').exists():raise RuntimeError('Complete the training-only feasibility pilot first')
    if (root/'training-complete.json').exists():print('Training complete; use calibrate.');return
    seed(config['seed']);device='cuda';model,tokenizer=initialize(config);model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    records=load_rows(root,'train');validation=load_rows(root,'val');dataset=PairDataset(records)
    current=identity(root,config);checkpoint=root/'last.pt';epoch=0;cursor=0;best=-1.;history=[];step=0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False) # Only locally produced trusted checkpoints.
        if state['identity']!=current:raise ValueError('Resume configuration/data/code mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        epoch,cursor,best,history,step=(state[k] for k in ('epoch','cursor','best','history','step'))
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    began=time.perf_counter();last_save=began;session_examples=0
    def persist():
        temporary=root/'last.tmp'
        state={'identity':current,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'cursor':cursor,'best':best,'history':history,'step':step,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        torch.save(state,temporary);temporary.replace(checkpoint)
        save(root/'progress.json',{'epoch':epoch,'next_sentence':cursor,'optimizer_steps':step,'history':history,'resources':resource()})
    while epoch<config['epochs']:
        order=np.random.default_rng(config['seed']+epoch).permutation(len(dataset))
        while cursor<len(order):
            ids=order[cursor:cursor+config['effective_batch_size']];items=[dataset[int(i)] for i in ids]
            batch,labels=collate(items,tokenizer,config['max_length']);denominator=float(torch.where(labels==1,config['positive_weight'],1.).sum())
            model.train();optimizer.zero_grad(set_to_none=True);loss_total=0.
            for at in range(0,len(ids),config['micro_batch_size']):
                stop=at+config['micro_batch_size'];inputs={k:v[at:stop].to(device) for k,v in batch.items()}
                logits=model(**inputs);loss=weighted_loss(logits,labels[at:stop].to(device),config['positive_weight'],denominator)
                loss.backward();loss_total+=float(loss.detach())
            optimizer.step();cursor+=len(ids);step+=1;session_examples+=len(ids)
            if step%50==0:print(json.dumps({'epoch':epoch+1,'step':step,'loss':loss_total,'examples_per_second':session_examples/max(time.perf_counter()-began,1)}),flush=True)
            now=time.perf_counter()
            if now-last_save>=config['checkpoint_seconds'] or now-began>=config['session_seconds']:
                persist();last_save=now
            if now-began>=config['session_seconds']:print('Session boundary checkpoint saved. Rerun train to resume; final evaluation remains blocked.');return
        scores,seconds=predict(model,tokenizer,validation,config,device);choice=select(validation,scores,config)
        history.append({'epoch':epoch+1,'validation':choice,'inference_seconds':seconds})
        f1=choice['research']['sentence']['f1']
        if f1>best:best=f1;save_best(root,model,tokenizer,config);save(root/'best-selection.json',{'epoch':epoch+1,'validation':choice,'identity':current})
        epoch+=1;cursor=0;persist();save(root/'validation-history.json',history)
    save(root/'training-complete.json',{'identity':current,'epochs':epoch,'history':history,'resources':resource()})

def pilot(root,config):
    verify(root);unopened(root)
    if not torch.cuda.is_available():raise RuntimeError('Select a free GPU runtime before the pilot')
    if (root/'pilot.json').exists():print('Pilot already recorded; no test data used.');return
    seed(config['seed']);model,tokenizer=initialize(config);model.to('cuda');model.train()
    dataset=PairDataset(load_rows(root,'train'));labels=[dataset[i][2] for i in range(len(dataset))]
    # Training labels only; deterministic positive/negative coverage for correctness.
    positive=[i for i,y in enumerate(labels) if y][:128];negative=[i for i,y in enumerate(labels) if not y][:128]
    ids=np.random.default_rng(config['seed']).permutation(positive+negative).tolist()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate']);torch.cuda.reset_peak_memory_stats()
    began=time.perf_counter();losses=[];gradients=[];processed=0
    for at in range(0,len(ids),config['effective_batch_size']):
        chosen=ids[at:at+config['effective_batch_size']];inputs,truth=collate([dataset[i] for i in chosen],tokenizer,config['max_length'])
        denominator=float(torch.where(truth==1,config['positive_weight'],1.).sum());optimizer.zero_grad(set_to_none=True);total=0.
        for lo in range(0,len(chosen),config['micro_batch_size']):
            hi=lo+config['micro_batch_size'];logits=model(**{k:v[lo:hi].cuda() for k,v in inputs.items()});loss=weighted_loss(logits,truth[lo:hi].cuda(),config['positive_weight'],denominator);loss.backward();total+=float(loss.detach())
        grad=float(model.roberta.embeddings.word_embeddings.weight.grad.abs().sum());gradients.append(grad)
        if not grad>0:raise ValueError('Transformer gradients are zero')
        optimizer.step();losses.append(total);processed+=len(chosen)
    torch.cuda.synchronize();elapsed=time.perf_counter()-began
    path=root/'pilot-checkpoint.pt';torch.save(model.state_dict(),path)
    model.eval();sample,_=collate([dataset[ids[0]]],tokenizer,config['max_length'])
    with torch.no_grad():before=model(**{k:v.cuda() for k,v in sample.items()}).cpu()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    with torch.no_grad():after=model(**{k:v.cuda() for k,v in sample.items()}).cpu()
    if not torch.allclose(before,after,atol=1e-6,rtol=1e-5):raise ValueError('Pilot checkpoint reload mismatch')
    # Validation inference correctness uses additional TRAINING examples only.
    chosen_ids=set(ids);probe=[]
    for i in range(len(dataset)):
        if i not in chosen_ids:probe.append(i)
        if len(probe)==8:break
    with torch.no_grad():assert torch.isfinite(model(**{k:v.cuda() for k,v in collate([dataset[i] for i in probe],tokenizer,config['max_length'])[0].items()})).all()
    save(root/'pilot.json',{'training_examples':processed,'steps':len(losses),'losses':losses,'transformer_gradient_l1':gradients,
        'seconds':elapsed,'examples_per_second':processed/elapsed,'estimated_full_training_hours':len(dataset)*config['epochs']/(processed/elapsed)/3600,
        'checkpoint_reload_verified':True,'resources':resource(),'accuracy_claim':False,'scope':'TRAIN only; short balanced pilot is optimistic; validation/checkpoint I/O overhead and full length distribution can increase runtime.'})
    path.unlink() # Own disposable pilot weights; never selected or transferred into full training.

def export(root,config):
    if not (root/'final-results.json').exists():raise RuntimeError('Finish sealed final evaluation before export')
    model,tokenizer,meta=load_best(root,'cpu');model.eval();model.roberta.gradient_checkpointing_disable()
    sample=tokenizer('Source [SSS] second source sentence.','A target sentence.',return_tensors='pt')
    path=root/'best/model.onnx'
    torch.onnx.export(model,(sample['input_ids'],sample['attention_mask']),path,input_names=['input_ids','attention_mask'],output_names=['logits'],
        dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'logits':{0:'batch'}},opset_version=17)
    import onnxruntime as ort
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    max_error=0.
    for a,b in [('Source.','Target.'),('Long source text. '*40,'A substantially different target.'),('Unicode αβγ source.','Another sentence.')]:
        batch=tokenizer([a,a],[b,b],padding=True,truncation=True,max_length=512,return_tensors='pt')
        with torch.no_grad():expected=model(batch['input_ids'],batch['attention_mask']).numpy()
        actual=session.run(None,{k:batch[k].numpy() for k in ('input_ids','attention_mask')})[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4);max_error=max(max_error,float(np.max(np.abs(actual-expected))))
    files={p.name:digest(p) for p in (root/'best').iterdir() if p.is_file()}
    save(root/'export.json',{'files':files,'onnx_max_logit_error':max_error,'model_revision':config['revision'],'license':'Research only; ARC-Sim CC BY-NC-SA. No commercial clearance.','production_enabled':False})
    print(json.dumps({'exported':str(path),'max_logit_error':max_error}),flush=True)
