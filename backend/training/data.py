"""Official ARC identities and conservative leakage purges; no label synthesis."""
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile
import hashlib,json,re,urllib.request

CODE=Path(__file__).resolve().parents[2]
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def norm(text):return ' '.join(text.casefold().split())
def text_hash(text):return hashlib.sha256(norm(text).encode()).hexdigest()
def load_archive(path):
    with ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('.jsonl') and not n.startswith('__MACOSX/')]
        if len(names)!=1:raise ValueError('Ambiguous official ARC file')
        with z.open(names[0]) as stream:
            for i,line in enumerate(stream):
                row=json.loads(line);row['original_row']=i;yield row
def load_rows(root,split):
    with (root/'prepared'/f'{split}.jsonl').open(encoding='utf-8') as f:return [json.loads(line) for line in f]
def verify(root):
    manifest=json.loads((root/'prepared/manifest.json').read_text(encoding='utf-8'))
    for split,record in manifest['splits'].items():
        if digest(root/'prepared'/f'{split}.jsonl')!=record['sha256']:raise ValueError('Prepared data changed')
    return manifest

def prepare(root,archives=None):
    dest=root/'prepared'
    if (dest/'manifest.json').exists():verify(root);return
    dest.mkdir(parents=True,exist_ok=True)
    lock=json.loads((CODE/'configs/corpora-20261120.lock.json').read_text(encoding='utf-8'))
    official={};identities={}
    for split in ('train','val','test'):
        record=next(r for r in lock['artifacts'] if r.get('split')==split)
        path=(archives or root/'archives')/(split+'.jsonl.zip');path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_suffix('.partial')
            urllib.request.urlretrieve(record['url'],partial);partial.replace(path)
        if digest(path)!=record['sha256']:raise ValueError('Official archive hash mismatch')
        official[split]=list(load_archive(path));identities[split]=record['sha256']
    families=[{r['cited_paper_id'] for r in official[s]} for s in ('train','val','test')]
    if any(families[i]&families[j] for i in range(3) for j in range(i)):raise ValueError('Official source split collision')
    source={r['cited_paper_id']:' '.join(r['cited_paper_abstract']) for records in official.values() for r in records}
    shingles={sid:set(tuple(tokens[i:i+5]) for i in range(max(0,len(tokens)-4))) for sid,text in source.items() for tokens in [re.findall(r'\w+',norm(text))]}
    inverted=defaultdict(set);past_sources=set();past_hashes=set();bad_sources=set();near=[]
    hashes={sid:text_hash(text) for sid,text in source.items()}
    for split in ('train','val','test'):
        current={r['cited_paper_id'] for r in official[split]}
        for sid in current:
            s=shingles[sid];candidates=set()
            # At Jaccard >= .8, at most 20% of either set can be absent.
            for shingle in sorted(s,key=lambda g:len(inverted.get(g,())) )[:int(.2*len(s))+1]:candidates.update(inverted.get(shingle,()))
            for other in candidates:
                t=shingles[other]
                if len(s&t)/max(len(s|t),1)>=.8:bad_sources.add(sid);near.append([other,sid])
            if hashes[sid] in past_hashes:bad_sources.add(sid)
        for sid in current:
            for g in shingles[sid]:inverted[g].add(sid)
        past_sources.update(current)
        past_hashes.update(hashes[sid] for sid in current)
    previous=json.loads((CODE/'research/results/arc-cpu-selection.json').read_text(encoding='utf-8'))
    exposed_indices=set(previous['selection']['test']['original_rows'])
    exposed=[r for r in official['test'] if r['original_row'] in exposed_indices]
    exposed_sources={r['cited_paper_id'] for r in exposed};exposed_papers={r['citing_paper_id'] for r in exposed}
    exposed_sections={text_hash(' '.join(r['citing_paper_section_text'])) for r in exposed}
    seen_papers=set();seen_sections=set();splits={};exclusions={}
    for split,output in [('train','train'),('val','val'),('test','final')]:
        kept=[];excluded=Counter()
        for row in official[split]:
            sentences=row['citing_paper_section_text'];labels=row['sentence_labels']
            if len(sentences)!=len(labels) or any(y not in (0,1) for y in labels):raise ValueError('Invalid actual ARC labels')
            section=text_hash(' '.join(sentences));why=None
            if split!='train' and row['cited_paper_id'] in bad_sources:why='source_duplicate_or_near_duplicate'
            elif split!='train' and row['citing_paper_id'] in seen_papers:why='target_paper_overlap'
            elif split!='train' and section in seen_sections:why='target_section_duplicate'
            elif split=='test' and (row['cited_paper_id'] in exposed_sources or row['citing_paper_id'] in exposed_papers or section in exposed_sections):why='previous_test_exposure'
            if why:excluded[why]+=1;continue
            row=dict(row,pair_id=f'{split}:{row["original_row"]}')
            kept.append(row)
        if not kept or not sum(sum(r['sentence_labels']) for r in kept):raise ValueError('No evaluable positives remain after leakage controls')
        path=dest/f'{output}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept),encoding='utf-8')
        splits[output]={'pairs':len(kept),'source_families':len({r['cited_paper_id'] for r in kept}),'target_papers':len({r['citing_paper_id'] for r in kept}),
            'sentences':sum(len(r['sentence_labels']) for r in kept),'positives':sum(sum(r['sentence_labels']) for r in kept),'sha256':digest(path)}
        exclusions[output]=dict(excluded)
        seen_papers.update(r['citing_paper_id'] for r in official[split]);seen_sections.update(text_hash(' '.join(r['citing_paper_section_text'])) for r in official[split])
    save(dest/'manifest.json',{'campaign':'arc-roberta-sequence-pair-v1','archives':identities,'splits':splits,'exclusions':exclusions,
        'near_duplicate_source_pairs':near,'previous_exposure_pairs':len(exposed),'previous_exposed_source_families':sorted(exposed_sources),
        'previous_selection_sha256':digest(CODE/'research/results/arc-cpu-selection.json'),
        'license':'ARC-Sim CC BY-NC-SA 3.0; noncommercial private research. No commercial service clearance.',
        'protocol':'Keep official source splits; purge cross-split citing papers/exact sections and source 5-shingle Jaccard >= .8. Final excludes all source families and citing papers from the previously exposed242 pairs. No fresh final outcomes inspected.'})
    print(json.dumps(splits,indent=2),flush=True)
