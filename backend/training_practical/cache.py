"""Pretokenized, unpadded native RoBERTa pairs; no change to max512/truncation."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import numpy as np
from training.data import save,digest,load_rows
from training.model import PairDataset
from .data import frozen,verify_selection

def tokenizer(config,root=None):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(config['model'],revision=config['revision'],trust_remote_code=False,
        **({'cache_dir':root/'hf-cache'} if root else {}))
    tok.add_special_tokens({'additional_special_tokens':['[SSS]']})
    return tok

def training_items(root,config):
    records=load_rows(root,'train');selection=verify_selection(root,config,records)
    return [(' [SSS] '.join(records[e['row']]['cited_paper_abstract']),
             records[e['row']]['citing_paper_section_text'][e['sentence']],e['label']) for e in selection['entries']]

def encode(root,config):
    import importlib.metadata as md
    frozen(root);selection=verify_selection(root,config)
    dest=root/'tokens';dest.mkdir(exist_ok=True)
    binding={'selection_sha256':digest(root/'subset/selection.json'),'model':config['model'],'revision':config['revision'],
        'max_length':config['max_length'],'special_token':'[SSS]','truncation':'longest_first',
        'transformers':md.version('transformers'),'tokenizers':md.version('tokenizers')}
    path=dest/'manifest.json'
    if path.exists():
        value=json.loads(path.read_text())
        if value['binding']!=binding:raise ValueError('Token cache identity changed')
        for split,record in value['files'].items():
            if digest(dest/(split+'.npz'))!=record['sha256']:raise ValueError('Token cache corrupted')
        return
    tok=tokenizer(config,root);files={};audit={}
    assert tok.pad_token_id==1 and tok.bos_token_id==0 and tok.eos_token_id==2
    for split in ('train','val','final'):
        if split=='train':items=training_items(root,config)
        else:
            dataset=PairDataset(load_rows(root,split));items=[dataset[i][:3] for i in range(len(dataset))]
        flat=[];offsets=[0];labels=[];lengths=[];source_lengths={};target_lengths={};truncated=0;by_stratum=defaultdict(list)
        for at in range(0,len(items),64):
            chunk=items[at:at+64]
            encoded=tok([r[0] for r in chunk],[r[1] for r in chunk],padding=False,truncation='longest_first',max_length=config['max_length'])
            for j,(item,ids) in enumerate(zip(chunk,encoded['input_ids'])):
                flat.extend(ids);offsets.append(len(flat));labels.append(item[2]);lengths.append(len(ids))
                a,b=item[:2]
                if a not in source_lengths:source_lengths[a]=len(tok.encode(a,add_special_tokens=False))
                if b not in target_lengths:target_lengths[b]=len(tok.encode(b,add_special_tokens=False))
                truncated+=source_lengths[a]+target_lengths[b]+tok.num_special_tokens_to_add(pair=True)>config['max_length']
                if split=='train':by_stratum[selection['entries'][at+j]['stratum']].append(len(ids))
        temporary=dest/(split+'.tmp.npz')
        np.savez_compressed(temporary,input_ids=np.asarray(flat,dtype=np.int32),offsets=np.asarray(offsets,dtype=np.int64),labels=np.asarray(labels,dtype=np.int64))
        temporary.replace(dest/(split+'.npz'))
        files[split]={'examples':len(items),'sha256':digest(dest/(split+'.npz'))}
        describe=lambda x:{'minimum':int(min(x)),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'maximum':int(max(x)),
            'bins_1_128_256_384_512':np.histogram(x,bins=[0,129,257,385,513])[0].tolist()}
        audit[split]={'lengths':describe(lengths),'truncated_pairs':int(truncated),
            'by_training_stratum':{k:describe(v) for k,v in by_stratum.items()}}
        print(json.dumps({'tokenized_split':split,**files[split],'lengths':audit[split]['lengths']}),flush=True)
    tok.save_pretrained(dest/'tokenizer')
    save(path,{'binding':binding,'files':files,'audit':audit,'accuracy_claim':False})

class Encoded:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as value:
            self.ids=value['input_ids'];self.offsets=value['offsets'];self.labels=value['labels']
        self.lengths=np.diff(self.offsets)
        if len(self.labels)!=len(self.lengths) or self.offsets[0]!=0 or self.offsets[-1]!=len(self.ids):
            raise ValueError('Invalid token cache offsets')
    def __len__(self):return len(self.labels)
    def collate(self,indices,multiple=8):
        import torch
        indices=np.asarray(indices,dtype=np.int64);longest=int(max(self.lengths[indices]));longest=((longest+multiple-1)//multiple)*multiple
        values=torch.full((len(indices),longest),1,dtype=torch.long) # RoBERTa pad_token_id=1.
        mask=torch.zeros_like(values)
        for at,i in enumerate(indices):
            ids=torch.from_numpy(self.ids[self.offsets[i]:self.offsets[i+1]].astype(np.int64))
            values[at,:len(ids)]=ids;mask[at,:len(ids)]=1
        return {'input_ids':values,'attention_mask':mask},torch.from_numpy(self.labels[indices].copy())

def batches(lengths,seed,batch_size=32,group_size=1024):
    rng=np.random.default_rng(seed);permutation=rng.permutation(len(lengths));result=[]
    for start in range(0,len(permutation),group_size):
        group=permutation[start:start+group_size]
        group=group[np.argsort(np.asarray(lengths)[group],kind='stable')]
        result.extend(group[at:at+batch_size] for at in range(0,len(group),batch_size))
    return [result[int(i)] for i in rng.permutation(len(result))]
