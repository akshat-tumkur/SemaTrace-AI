"""Full encoder fine-tuning of source-abstract / target-sentence pairs."""
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer,RobertaModel,RobertaConfig

class ReuseRoberta(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.roberta=encoder;self.dropout=nn.Dropout(.1);self.classifier=nn.Linear(encoder.config.hidden_size,2)
        encoder._init_weights(self.classifier)
    def forward(self,input_ids,attention_mask):
        cls=self.roberta(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0]
        return self.classifier(self.dropout(cls))

def initialize(config):
    tokenizer=AutoTokenizer.from_pretrained(config['model'],revision=config['revision'],trust_remote_code=False)
    tokenizer.add_special_tokens({'additional_special_tokens':['[SSS]']})
    encoder=RobertaModel.from_pretrained(config['model'],revision=config['revision'],add_pooling_layer=False,trust_remote_code=False)
    encoder.resize_token_embeddings(len(tokenizer))
    if config['gradient_checkpointing']:encoder.gradient_checkpointing_enable()
    return ReuseRoberta(encoder),tokenizer

class PairDataset:
    def __init__(self,records):
        self.records=records;self.sources=[' [SSS] '.join(r['cited_paper_abstract']) for r in records]
        self.index=np.asarray([(i,j) for i,r in enumerate(records) for j in range(len(r['sentence_labels']))],dtype=np.int32)
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        i,j=map(int,self.index[index]);r=self.records[i]
        return self.sources[i],r['citing_paper_section_text'][j],int(r['sentence_labels'][j]),i,j

def collate(items,tokenizer,max_length=512):
    encoded=tokenizer([r[0] for r in items],[r[1] for r in items],padding=True,truncation='longest_first',max_length=max_length,return_tensors='pt')
    return {'input_ids':encoded['input_ids'],'attention_mask':encoded['attention_mask']},torch.tensor([r[2] for r in items],dtype=torch.long)

def weighted_loss(logits,labels,positive_weight,denominator=None):
    weights=logits.new_tensor([1.,positive_weight])
    value=nn.functional.cross_entropy(logits,labels,weight=weights,reduction='sum')
    return value/(weights[labels].sum() if denominator is None else denominator)
