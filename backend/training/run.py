"""Portable CLI. Full training is GPU-only; smoke trains a tiny test fixture."""
from pathlib import Path
import sys,argparse,json
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.data import prepare,save,CODE,verify

def smoke(root,config):
    import torch,numpy as np
    from transformers import AutoTokenizer,RobertaModel,RobertaConfig
    from training.model import ReuseRoberta,collate,weighted_loss
    from training.engine import seed
    seed(config['seed']);torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(config['model'],revision=config['revision'],cache_dir=root/'hf-cache',trust_remote_code=False)
    tokenizer.add_special_tokens({'additional_special_tokens':['[SSS]']})
    assert tokenizer.convert_tokens_to_ids('[SSS]')!=tokenizer.unk_token_id
    cfg=RobertaConfig(vocab_size=len(tokenizer),hidden_size=24,num_hidden_layers=2,num_attention_heads=4,intermediate_size=48,max_position_embeddings=514)
    model=ReuseRoberta(RobertaModel(cfg,add_pooling_layer=False));optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    examples=[('First source. [SSS] Second source.','Corresponding target sentence.',1,0,0),('Another source.','A different target sentence.',0,0,1)]
    batch,labels=collate(examples,tokenizer,512)
    assert labels.tolist()==[1,0] and batch['input_ids'].shape[0]==2
    assert tokenizer.convert_tokens_to_ids('[SSS]') in batch['input_ids'][0]
    loss=weighted_loss(model(**batch),labels,5.);loss.backward()
    gradient=float(model.roberta.encoder.layer[0].attention.self.query.weight.grad.abs().sum())
    assert gradient>0 and torch.isfinite(loss);optimizer.step();model.eval()
    dest=root/'smoke';dest.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),dest/'fixture.pt')
    with torch.no_grad():expected=model(**batch)
    model.load_state_dict(torch.load(dest/'fixture.pt',weights_only=True))
    with torch.no_grad():torch.testing.assert_close(model(**batch),expected)
    path=dest/'fixture.onnx'
    torch.onnx.export(model,(batch['input_ids'],batch['attention_mask']),path,input_names=['input_ids','attention_mask'],output_names=['logits'],dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'logits':{0:'batch'}},opset_version=17)
    import onnxruntime as ort
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    errors=[]
    for items in [examples,examples[:1],[('Source words. '*60,'Target words.',0,0,0)]]:
        inputs,_=collate(items,tokenizer,512)
        with torch.no_grad():truth=model(**inputs).numpy()
        values=session.run(None,{k:v.numpy() for k,v in inputs.items()})[0]
        np.testing.assert_allclose(values,truth,atol=1e-4,rtol=1e-4);errors.append(float(np.max(np.abs(values-truth))))
    result={'fixture_only':True,'trained_task_model':False,'actual_pinned_tokenizer':True,'input_label_alignment':True,'transformer_gradient_l1':gradient,'loss':float(loss.detach()),'checkpoint_reload':True,'onnx_dynamic_shapes':True,'onnx_max_logit_error':max(errors),'device':'CPU','accuracy_claim':False}
    save(root/'local-smoke.json',result);print(json.dumps(result,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','smoke','pilot','train','calibrate','final','export','status'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--archives',type=Path);parser.add_argument('--config',type=Path)
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':prepare(root,args.archives);return
    if args.command=='status':print(json.dumps(verify(root),indent=2));return
    from training.engine import configuration,pilot,train,export
    config=configuration(args.config)
    if args.command=='smoke':smoke(root,config)
    elif args.command=='pilot':pilot(root,config)
    elif args.command=='train':train(root,config)
    elif args.command=='export':export(root,config)
    else:
        from training.evaluation import calibrate,final
        (calibrate if args.command=='calibrate' else final)(root,config)
if __name__=='__main__':main()
