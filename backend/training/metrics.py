"""Sentence and document metrics without giving negative pairs ranking credit."""
import numpy as np,re
from sklearn.metrics import average_precision_score

def classification(y,p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=bool)
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return {'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'false_positive_rate':fp/max(fp+tn,1)}

def metrics(records,scores,threshold):
    labels=np.asarray([y for r in records for y in r['sentence_labels']],dtype=np.int8);scores=np.asarray(scores);pred=scores>=threshold
    if len(labels)!=len(scores):raise ValueError('Sentence/score identity mismatch')
    dp=[];dl=[];maps=[];acc={1:[],3:[],5:[]};low=[];negative=[];at=0
    for r in records:
        n=len(r['sentence_labels']);ys=labels[at:at+n];ss=scores[at:at+n];dp.append(bool(pred[at:at+n].any()));dl.append(r['doc_label'])
        source=set(re.findall(r'\w+',' '.join(r['cited_paper_abstract']).casefold()))
        for text in r['citing_paper_section_text']:
            target=set(re.findall(r'\w+',text.casefold()));low.append(len(source&target)/max(len(source|target),1)<=.1);negative.append(r['doc_label']==0)
        if ys.any():
            maps.append(float(average_precision_score(ys,ss)));rank=np.argsort(-ss,kind='stable')
            for k in acc:acc[k].append(bool(ys[rank[:k]].any()))
        at+=n
    low=np.asarray(low);negative=np.asarray(negative)
    return {'sentence':classification(labels,pred),'document':classification(dl,dp),'threshold':float(threshold),
        'low_lexical_overlap':dict(classification(labels[low],pred[low]),sentences=int(low.sum()),positive_sentences=int(labels[low].sum())),
        'related_negative_sentences':classification(labels[negative],pred[negative]),
        'ranking':{'map':float(np.mean(maps)) if maps else None,**{f'acc@{k}':float(np.mean(v)) if v else None for k,v in acc.items()}}}

def select(records,scores,config):
    labels=np.asarray([y for r in records for y in r['sentence_labels']]);scores=np.asarray(scores)
    starts=np.cumsum([0]+[len(r['sentence_labels']) for r in records])[:-1]
    dl=[r['doc_label'] for r in records]
    values=[]
    for t in sorted(set(np.linspace(0,1,101).tolist()+[config['published_threshold'],1.000001])):
        pred=scores>=t
        values.append({'threshold':t,'sentence':classification(labels,pred),'document':classification(dl,np.maximum.reduceat(pred,starts))})
    best=max(values,key=lambda r:(r['sentence']['f1'],r['document']['f1'],r['threshold']))
    a=config['acceptance']
    automatic=[r for r in values if r['sentence']['precision']>=a['automatic_precision'] and r['document']['false_positive_rate']<=a['negative_document_fpr'] and r['sentence']['tp']+r['sentence']['fp']>=a['minimum_predicted_positive_sentences']]
    auto=max(automatic,key=lambda r:(r['sentence']['recall'],r['threshold'])) if automatic else None
    return {'research':metrics(records,scores,best['threshold']),'automatic':metrics(records,scores,auto['threshold']) if auto else None}
