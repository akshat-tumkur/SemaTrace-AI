"""Tiny structural fixture only: exact optimizer/RNG resume, weighting and ONNX."""
from pathlib import Path
import json,copy
import numpy as np
import torch
from transformers import RobertaConfig,RobertaModel
from training.model import ReuseRoberta,weighted_loss
from training.engine import seed
from training.data import save
from .cache import tokenizer,Encoded,batches
from .engine import update,rng_state,restore_rng,save_checkpoint,load_checkpoint,environment

def smoke(root,config):
    seed(config['seed']);torch.set_num_threads(2)
    tok=tokenizer(config,root);dest=root/'smoke-practical';dest.mkdir(exist_ok=True)
    texts=[('First source. [SSS] A second source sentence.','Target sentence '+str(i)+'.' if i%2 else 'A longer target sentence about research '+str(i)+'.') for i in range(32)]
    encoded=tok([a for a,b in texts],[b for a,b in texts],padding=False,truncation='longest_first',max_length=512)
    flat=[];offsets=[0]
    for ids in encoded['input_ids']:flat.extend(ids);offsets.append(len(flat))
    np.savez(dest/'fixture.npz',input_ids=np.asarray(flat,dtype=np.int32),offsets=np.asarray(offsets,dtype=np.int64),labels=np.asarray([i%2 for i in range(32)],dtype=np.int64))
    dataset=Encoded(dest/'fixture.npz');cfg=RobertaConfig(vocab_size=len(tok),hidden_size=24,num_hidden_layers=2,num_attention_heads=4,intermediate_size=48,max_position_embeddings=514)
    make=lambda:ReuseRoberta(RobertaModel(cfg,add_pooling_layer=False))
    model=make();optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate']);scaler=torch.amp.GradScaler('cuda',enabled=False)
    ids=np.arange(32);first=update(model,optimizer,scaler,dataset,ids,config,device='cpu',probe_gradient=True)
    assert first['gradient_l1']>0
    state={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'scaler':scaler.state_dict(),'rng':rng_state(),'cursor':1}
    save_checkpoint(dest,state)
    expected=update(model,optimizer,scaler,dataset,ids[::-1].copy(),config,device='cpu')
    expected_weights=copy.deepcopy(model.state_dict())
    reloaded=make();opt=torch.optim.Adam(reloaded.parameters(),lr=config['learning_rate']);sc=torch.amp.GradScaler('cuda',enabled=False)
    restored=load_checkpoint(dest);reloaded.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer']);sc.load_state_dict(restored['scaler']);restore_rng(restored['rng'])
    actual=update(reloaded,opt,sc,dataset,ids[::-1].copy(),config,device='cpu')
    for name,value in reloaded.state_dict().items():torch.testing.assert_close(value,expected_weights[name],rtol=0,atol=0)
    assert expected['loss']==actual['loss']
    # Class weighting must be normalized once per effective batch, not per microbatch.
    logits=torch.randn(11,2,requires_grad=True);truth=torch.tensor([0,1,0,0,0,1,0,0,0,0,1])
    full=weighted_loss(logits,truth,config['positive_weight']);expected_grad=torch.autograd.grad(full,logits,retain_graph=True)[0]
    denom=float(torch.where(truth==1,config['positive_weight'],1.).sum())
    accumulated=sum(weighted_loss(logits[i:i+3],truth[i:i+3],config['positive_weight'],denom) for i in range(0,11,3))
    torch.testing.assert_close(torch.autograd.grad(accumulated,logits)[0],expected_grad,atol=1e-7,rtol=1e-6)
    order=batches(dataset.lengths,config['seed'],batch_size=7,group_size=14)
    assert sorted(int(i) for b in order for i in b)==list(range(32))
    assert [b.tolist() for b in order]==[b.tolist() for b in batches(dataset.lengths,config['seed'],7,14)]
    # Commit second slot; corruption of the inactive old slot must not invalidate the active pointer.
    save_checkpoint(dest,restored)
    (dest/'checkpoints/last-0.pt').write_bytes(b'own inactive test slot')
    assert load_checkpoint(dest)['cursor']==1
    reloaded.eval();inputs,_=dataset.collate(ids[:2],8)
    assert inputs['input_ids'].shape[1]%8==0
    assert tok.convert_tokens_to_ids('[SSS]') in inputs['input_ids'][0]
    path=dest/'fixture.onnx'
    torch.onnx.export(reloaded,(inputs['input_ids'],inputs['attention_mask']),path,input_names=['input_ids','attention_mask'],output_names=['logits'],
        dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'logits':{0:'batch'}},opset_version=17,dynamo=False)
    import onnxruntime as ort
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);errors=[]
    for text_pairs in [texts[:2],texts[:1],[('Source text. '*70,'A longer target example.')]]:
        batch=tok([a for a,b in text_pairs],[b for a,b in text_pairs],padding=True,truncation='longest_first',max_length=512,return_tensors='pt')
        values={k:v for k,v in batch.items() if k in ('input_ids','attention_mask')}
        with torch.no_grad():reference=reloaded(**values).numpy()
        result=session.run(None,{k:v.numpy() for k,v in values.items()})[0]
        np.testing.assert_allclose(result,reference,atol=1e-4,rtol=1e-4);errors.append(float(np.max(np.abs(result-reference))))
    report={'fixture_only':True,'task_model_trained':False,'accuracy_claim':False,'weighted_accumulation_matches_effective_batch':True,
        'transformer_gradient_l1':first['gradient_l1'],'optimizer_scaler_rng_resume_exact_cpu':True,'two_slot_checkpoint_integrity':True,
        'deterministic_complete_epoch_order':True,'dynamic_microbatch_padding':True,'onnx_dynamic_parity_max_logit_error':max(errors),
        'environment':environment(),'gpu_amp_pilot_executed':False}
    save(root/'local-practical-smoke.json',report);print(json.dumps(report,indent=2))
