"""Unchanged validation/final partitions and baseline definitions; one final seal."""
import json,time
import numpy as np
from training.data import CODE,load_rows,digest,save
from training.engine import load_best,export as original_export
from training.evaluation import embedding_scores,tfidf_scores
from training.metrics import metrics,select
from .data import unopened,check_ancestor,frozen
from .engine import identity,predict,atomic
from .cache import Encoded

def calibrate(root,config):
    import joblib,torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    unopened(root);current=identity(root,config)
    if not (root/'training-complete.json').exists():raise RuntimeError('All three epochs must complete before final validation selection')
    if json.loads((root/'training-complete.json').read_text())['identity']!=current:raise ValueError('Training campaign changed')
    if (root/'selection.json').exists():raise FileExistsError('Frozen validation selection already exists')
    device='cuda' if torch.cuda.is_available() else 'cpu';model,tok,meta=load_best(root,device)
    records=load_rows(root,'val');values,seconds=predict(model,Encoded(root/'tokens/val.npz'),config,device)
    methods={'roberta':select(records,values,config)};del model
    if device=='cuda':torch.cuda.empty_cache()
    # Preserve the original full-training TF-IDF baseline protocol, not a new tuned classifier.
    training=load_rows(root,'train')
    texts={' '.join(r['cited_paper_abstract']) for r in training}|{s for r in training for s in r['citing_paper_section_text']}
    vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=200000,min_df=2,dtype=np.float32).fit(sorted(texts))
    del training,texts
    joblib.dump(vectorizer,root/'tfidf.joblib')
    baseline,truncated=embedding_scores(root,records);baseline['tfidf']=tfidf_scores(vectorizer,records)
    for name,scores in baseline.items():methods[name]=select(records,scores,config)
    atomic(root/'selection.json',{'identity':current,'methods':methods,'model_sha256':meta['weights_sha256'],
        'tfidf_sha256':digest(root/'tfidf.joblib'),'validation_inference_seconds':seconds,'minilm_truncated_texts':truncated,
        'baseline_protocol':'Unchanged original definitions. TF-IDF fits unique full official training texts, (1,2)-grams/200k/min_df2; MiniLM and prior head weights stay frozen; validation thresholds only. No MiniLM head retraining.'})

def final(root,config):
    import joblib,torch
    frozen(root);check_ancestor(root)
    selection=json.loads((root/'selection.json').read_text());current=identity(root,config)
    if selection['identity']!=current:raise ValueError('Campaign changed after validation selection')
    seal={'campaign':config['campaign'],'selection_sha256':digest(root/'selection.json'),
          'frozen_final_sha256':frozen(root)['splits']['final']['sha256']}
    for path in (root.parent/'sematrace-roberta-final-seal.json',root/'test-opened.json'):
        if path.exists() and json.loads(path.read_text())!=seal:raise ValueError('Existing final seal conflicts; never overwrite it')
    if (root/'final-results.json').exists():print('Completed final result preserved.');return
    for path in (root.parent/'sematrace-roberta-final-seal.json',root/'test-opened.json'):atomic(path,seal)
    records=load_rows(root,'final');device='cuda' if torch.cuda.is_available() else 'cpu'
    model,tok,meta=load_best(root,device)
    if meta['weights_sha256']!=selection['model_sha256'] or digest(root/'tfidf.joblib')!=selection['tfidf_sha256']:raise ValueError('Selected weights/baseline changed')
    began=time.perf_counter();values,seconds=predict(model,Encoded(root/'tokens/final.npz'),config,device);del model
    if device=='cuda':torch.cuda.empty_cache()
    baseline,truncated=embedding_scores(root,records);baseline['tfidf']=tfidf_scores(joblib.load(root/'tfidf.joblib'),records);baseline['roberta']=values
    methods={name:metrics(records,scores,selection['methods'][name]['research']['threshold']) for name,scores in baseline.items()}
    point=selection['methods']['roberta']['automatic'];automatic=metrics(records,values,point['threshold']) if point else None
    gain=methods['roberta']['sentence']['f1']-methods['tfidf']['sentence']['f1'];a=config['acceptance']
    accepted=bool(automatic and gain>=a['sentence_f1_gain_over_tfidf'] and automatic['sentence']['precision']>=a['automatic_precision']
        and automatic['sentence']['recall']>=a['automatic_recall'] and automatic['document']['false_positive_rate']<=a['negative_document_fpr']
        and automatic['sentence']['tp']+automatic['sentence']['fp']>=a['minimum_predicted_positive_sentences'])
    np.savez(root/'final-scores.npz',**baseline)
    with (root/'final-predictions.jsonl').open('w',encoding='utf-8') as f:
        at=0
        for row in records:
            for j,label in enumerate(row['sentence_labels']):
                f.write(json.dumps({'pair_id':row['pair_id'],'source_family':row['cited_paper_id'],'target_paper':row['citing_paper_id'],
                    'sentence_index':j,'label':label,'score':float(values[at]),'predicted':bool(values[at]>=selection['methods']['roberta']['research']['threshold'])})+'\n');at+=1
    result={'campaign':config['campaign'],'adaptation':'Computationally constrained 30000-example full-encoder fine-tuning, not exact reproduction',
        'seal':seal,'methods':methods,'automatic_operating_point':automatic,'f1_gain_over_tfidf':gain,
        'research_quality_gate_passed':accepted,'production_enabled':False,'roberta_inference_seconds':seconds,
        'all_evaluation_seconds':time.perf_counter()-began,'minilm_truncated_texts':truncated,
        'published_reference_f1':.416,'published_score_comparable':False,
        'remaining_gates':'Local CPU latency/memory, application transfer, source-side passage localization and source-attribution evaluation before integration. ARC sentence labels do not authorize semantic coverage.'}
    atomic(root/'final-results.json',result);print(json.dumps(result,indent=2))

def export(root,config):
    result=json.loads((root/'final-results.json').read_text())
    if not result['research_quality_gate_passed']:raise RuntimeError('Final acceptance failed; preserve results. No automatic export/new model campaign.')
    original_export(root,config)
