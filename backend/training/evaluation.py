"""Matched fresh-partition baselines and final exposure seals, without refitting heads."""
import json,time,hashlib
import numpy as np
from .data import CODE,digest,save,verify,load_rows
from .engine import predict,load_best,identity,unopened
from .metrics import metrics,select

def embedding_scores(root,records):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    lock=json.loads((CODE/'backend/corpus/minilm-onnx.lock.json').read_text(encoding='utf-8'))
    paths={}
    for name in ('onnx/model.onnx','tokenizer.json'):
        local=CODE/'backend/data/models/all-MiniLM-L6-v2'/lock['revision']/name
        path=str(local) if local.is_file() else hf_hub_download(lock['model'],name,revision=lock['revision'])
        entry=next(r for r in lock['files'] if r['rfilename']==name)
        if 'lfs' in entry:
            if digest(path)!=entry['lfs']['sha256']:raise ValueError('MiniLM content hash mismatch')
        else:
            data=open(path,'rb').read()
            if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=entry['blobId']:raise ValueError('MiniLM tokenizer hash mismatch')
        paths[name]=path
    tokenizer=Tokenizer.from_file(paths['tokenizer.json']);tokenizer.enable_truncation(max_length=256);tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(paths['onnx/model.onnx'],sess_options=options,providers=['CPUExecutionProvider'])
    texts=sorted({' '.join(r['cited_paper_abstract']) for r in records}|{s for r in records for s in r['citing_paper_section_text']},key=lambda s:(len(s),s))
    index={t:i for i,t in enumerate(texts)};vectors=np.empty((len(texts),384),dtype=np.float32);names={i.name for i in session.get_inputs()};truncated=0
    for at in range(0,len(texts),32):
        encoded=tokenizer.encode_batch(texts[at:at+32]);truncated+=sum(bool(e.overflowing) for e in encoded)
        mask=np.asarray([e.attention_mask for e in encoded],dtype=np.int64)
        inputs={'input_ids':np.asarray([e.ids for e in encoded],dtype=np.int64),'attention_mask':mask,'token_type_ids':np.asarray([e.type_ids for e in encoded],dtype=np.int64)}
        output=session.run(None,{k:v for k,v in inputs.items() if k in names})[0]
        pooled=(output*mask[...,None]).sum(1)/mask.sum(1)[:,None];pooled/=np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-9)
        vectors[at:at+len(encoded)]=pooled
    old=json.loads((CODE/'research/results/arc-cpu-results.json').read_text(encoding='utf-8'))['head_selection']
    artifact=root/'inputs/legacy-head.npz'
    if digest(artifact)!=old['artifact_sha256']:raise ValueError('Upload the exact private frozen legacy head artifact')
    with np.load(artifact,allow_pickle=False) as head:
        cosine=[];learned=[]
        for r in records:
            u=vectors[index[' '.join(r['cited_paper_abstract'])]];v=vectors[[index[s] for s in r['citing_paper_section_text']]]
            cosine.extend((u*v).sum(1).tolist())
            x=np.concatenate((np.broadcast_to(u,v.shape),v,np.abs(u-v)),axis=1)
            logits=(x@head['coef'].T).reshape(-1)+float(head['intercept'][0]);learned.extend((1/(1+np.exp(-np.clip(logits,-50,50)))).tolist())
    return {'minilm':np.asarray(cosine),'previous-supervised-minilm':np.asarray(learned)},truncated

def tfidf_scores(vectorizer,records):
    scores=[]
    for r in records:
        u=vectorizer.transform([' '.join(r['cited_paper_abstract'])]);v=vectorizer.transform(r['citing_paper_section_text']);scores.extend((v@u.T).toarray().reshape(-1).tolist())
    return np.asarray(scores)

def calibrate(root,config):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    unopened(root);verify(root)
    if not (root/'training-complete.json').exists():raise RuntimeError('Resume training until all configured epochs complete')
    if (root/'selection.json').exists():raise FileExistsError('Validation selection already frozen')
    import torch
    device='cuda' if torch.cuda.is_available() else 'cpu';model,tokenizer,meta=load_best(root,device)
    validation=load_rows(root,'val');scores,seconds=predict(model,tokenizer,validation,config,device)
    choices={'roberta':select(validation,scores,config)};del model
    if device=='cuda':torch.cuda.empty_cache()
    training=load_rows(root,'train');collection={' '.join(r['cited_paper_abstract']) for r in training}|{s for r in training for s in r['citing_paper_section_text']}
    vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=200000,min_df=2,dtype=np.float32).fit(sorted(collection))
    del training,collection
    joblib.dump(vectorizer,root/'tfidf.joblib')
    baseline,truncated=embedding_scores(root,validation);baseline['tfidf']=tfidf_scores(vectorizer,validation)
    for name,values in baseline.items():choices[name]=select(validation,values,config)
    selection={'identity':identity(root,config),'model_sha256':meta['weights_sha256'],'tfidf_sha256':digest(root/'tfidf.joblib'),
        'methods':choices,'validation_roberta_inference_seconds':seconds,'validation_minilm_truncated_texts':truncated,
        'baseline_notes':'TF-IDF SP fits unique training sentences/abstracts, vocabulary capped200k/min_df2 for memory. MiniLM is the original pinned encoder. The previous frozen reuse head is not retrained; its threshold is recalibrated on the same validation partition for a matched comparison. No inference of character/source-span accuracy.'}
    save(root/'selection.json',selection)

def final(root,config):
    import joblib,torch
    verify(root);selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    if selection['identity']!=identity(root,config):raise ValueError('Campaign changed after validation')
    seal={'selection_sha256':digest(root/'selection.json'),'data_sha256':digest(root/'prepared/manifest.json')}
    marker=root/'test-opened.json'
    if marker.exists():
        if json.loads(marker.read_text())!=seal:raise ValueError('Final selection seal mismatch')
        if (root/'final-results.json').exists():print('Completed final result preserved.');return
    else:save(marker,seal)
    records=load_rows(root,'final');device='cuda' if torch.cuda.is_available() else 'cpu'
    model,tokenizer,meta=load_best(root,device)
    if meta['weights_sha256']!=selection['model_sha256'] or digest(root/'tfidf.joblib')!=selection['tfidf_sha256']:raise ValueError('Selected model/baseline changed')
    began=time.perf_counter();scores,seconds=predict(model,tokenizer,records,config,device);del model
    baseline,truncated=embedding_scores(root,records);baseline['tfidf']=tfidf_scores(joblib.load(root/'tfidf.joblib'),records);baseline['roberta']=scores
    methods={name:metrics(records,values,selection['methods'][name]['research']['threshold']) for name,values in baseline.items()}
    auto=selection['methods']['roberta']['automatic'];automatic=metrics(records,scores,auto['threshold']) if auto else None
    a=config['acceptance'];gain=methods['roberta']['sentence']['f1']-methods['tfidf']['sentence']['f1']
    accepted=bool(automatic and gain>=a['sentence_f1_gain_over_tfidf'] and automatic['sentence']['precision']>=a['automatic_precision'] and automatic['sentence']['recall']>=a['automatic_recall'] and automatic['document']['false_positive_rate']<=a['negative_document_fpr'] and automatic['sentence']['tp']+automatic['sentence']['fp']>=a['minimum_predicted_positive_sentences'])
    np.savez(root/'final-scores.npz',**baseline)
    with (root/'final-predictions.jsonl').open('w',encoding='utf-8') as f:
        at=0
        for r in records:
            for j,label in enumerate(r['sentence_labels']):
                f.write(json.dumps({'pair_id':r['pair_id'],'source_family':r['cited_paper_id'],'target_paper':r['citing_paper_id'],'sentence_index':j,'label':label,'score':float(scores[at]),'predicted':bool(scores[at]>=selection['methods']['roberta']['research']['threshold'])})+'\n');at+=1
    result={'seal':seal,'methods':methods,'automatic_operating_point':automatic,'research_quality_gate_passed':accepted,'f1_gain_over_tfidf':gain,
        'roberta_inference_seconds':seconds,'all_evaluation_seconds':time.perf_counter()-began,'minilm_truncated_texts':truncated,
        'false_positive_count':methods['roberta']['sentence']['fp'],'false_negative_count':methods['roberta']['sentence']['fn'],
        'production_enabled':False,'remaining_gates':'Source-side localization, application transfer/domain-shift evaluation, CPU model latency/memory and usage rights. ARC labels alone do not validate a plagiarism engine.',
        'published_reference_f1':.416,'published_score_comparable':False,'reason':'Conservative target/source purges and prior-exposure exclusions change the official test composition.'}
    save(root/'final-results.json',result)
    print(json.dumps(result,indent=2))
