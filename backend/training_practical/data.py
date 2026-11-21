"""Fixed training-only sampling; official validation/final bytes remain frozen."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib, json, re, shutil
import numpy as np
from training.data import CODE, digest, save, prepare as prepare_original, verify, load_rows

FROZEN = {
    'train': '6f2f554837e64a5dfed4952d6c4a4c77e42743d0f157a82fc8897750f92b8ec8',
    'val': '77a1df2b17601e3c0bcc63820b97bb681769d879559d0d94029593caded213a8',
    'final': '43bcd85df3e2032e6990e01197208bb8de79417dce722fa7eb70a1d870486cb6',
}
STRATA = ('positive_low_overlap','positive_other','negative_related_high_overlap',
          'negative_positive_section_high_overlap','negative_related_other','negative_positive_section_other')

def check_ancestor(root):
    path=root/'ancestry.json'
    if path.exists():
        ancestor=Path(json.loads(path.read_text())['original_root'])
        if (ancestor/'test-opened.json').exists():
            raise RuntimeError('Original RoBERTa final was opened; preserve its seal. This adaptation cannot reuse it as fresh.')

def unopened(root):
    check_ancestor(root)
    if (root/'test-opened.json').exists() or (root.parent/'sematrace-roberta-final-seal.json').exists():
        raise RuntimeError('Final exposure seal exists; training/selection is closed.')

def frozen(root):
    manifest=verify(root)
    if {s:r['sha256'] for s,r in manifest['splits'].items()}!=FROZEN:
        raise ValueError('Official prepared partitions differ from the frozen protocol')
    return manifest

def stage_frozen(source_root, target_root):
    """Verify source bytes first; stage only the exact frozen CRLF serialization."""
    import tempfile
    source_root, target_root = Path(source_root), Path(target_root)
    manifest = verify(source_root)  # Never repair a manifest/file integrity mismatch.
    if set(manifest['splits']) != set(FROZEN):
        raise ValueError('Unexpected prepared partitions')
    records = {}
    # Validate every candidate before publishing any destination file.
    with tempfile.TemporaryDirectory(prefix='prepared-newlines-', dir=target_root) as scratch:
        scratch = Path(scratch)
        for split, expected in FROZEN.items():
            raw = (source_root/'prepared'/f'{split}.jsonl').read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != manifest['splits'][split]['sha256']:
                raise ValueError(f'{split}: source changed during verification')
            lf = raw.replace(b'\r\n', b'\n')
            canonical = lf.replace(b'\n', b'\r\n')
            if b'\r' in lf or hashlib.sha256(canonical).hexdigest() != expected:
                raise ValueError(f'{split}: content differs from frozen data beyond newline serialization')
            (scratch/f'{split}.jsonl').write_bytes(canonical)
            records[split] = {'source_sha256': actual, 'frozen_sha256': expected,
                              'newline_only_conversion': raw != canonical}
        corrected = json.loads(json.dumps(manifest))
        for split, expected in FROZEN.items():
            corrected['splits'][split]['sha256'] = expected
        corrected['serialization'] = {
            'format': 'UTF-8 JSONL with CRLF record separators',
            'source_manifest_sha256': digest(source_root/'prepared/manifest.json'),
            'verified_source_files': records,
        }
        (scratch/'manifest.json').write_bytes(json.dumps(corrected, indent=2).encode('utf-8'))
        destination = target_root/'prepared'
        destination.mkdir(exist_ok=True)
        for split in FROZEN:
            (scratch/f'{split}.jsonl').replace(destination/f'{split}.jsonl')
        (scratch/'manifest.json').replace(destination/'manifest.json')
    return frozen(target_root)

def ensure_frozen(root):
    """Only the new campaign may be canonicalized, and only before training."""
    try:
        return frozen(root)
    except ValueError:
        unopened(root)
        if any((root/name).exists() for name in
               ('pilot.json','last.json','progress.json','training-complete.json','selection.json','best','selected-epochs','checkpoints')):
            raise RuntimeError('Cannot change prepared serialization after model work started')
        return stage_frozen(root, root)

def prepare(root,config,archives=None,original_root=None,selection=None):
    root.mkdir(parents=True,exist_ok=True)
    if original_root is not None:
        original_root=original_root.resolve()
        if original_root == root.resolve():raise ValueError('Original and practical campaign folders must differ')
        record={'original_root':str(original_root)}
        if (root/'ancestry.json').exists() and json.loads((root/'ancestry.json').read_text())!=record:
            raise ValueError('Original campaign path changed; do not bypass its seals')
        if not (root/'ancestry.json').exists():save(root/'ancestry.json',record)
    dest=root/'subset/selection.json'
    if (root/'prepared/manifest.json').exists() and dest.exists():
        ensure_frozen(root);verify_selection(root,config)
        if selection is not None and digest(selection)!=digest(dest):raise ValueError('Uploaded subset differs from existing selection')
        return
    unopened(root)
    if not (root/'prepared/manifest.json').exists():
        if original_root and (original_root/'prepared/manifest.json').exists():
            stage_frozen(original_root, root)
        else: prepare_original(root,archives)
    ensure_frozen(root)
    if dest.exists():
        verify_selection(root,config);return
    if selection is not None:
        dest.parent.mkdir(exist_ok=True);shutil.copyfile(selection,dest)
        verify_selection(root,config)
    else:
        sample(root,config)

def fingerprint(source,target):
    return hashlib.sha256(source+b'\0'+' '.join(target.casefold().split()).encode()).digest()[:16]

def sample(root,config):
    """Hash-ranked round robin by source family within each stratum/length cell."""
    manifest=frozen(root);records=load_rows(root,'train')
    total=sum(len(r['sentence_labels']) for r in records)
    dtype=[('row','i4'),('sentence','i4'),('family','i4'),('label','i1'),('doc','i1'),
           ('overlap','f4'),('length','i4'),('priority','u8'),('valid','?')]
    pool=np.empty(total,dtype=dtype);families=sorted({r['cited_paper_id'] for r in records})
    family_ids={s:i for i,s in enumerate(families)};sources={};seen={};conflicts=set();count=0;duplicate=0
    for i,row in enumerate(records):
        sid=row['cited_paper_id']
        if sid not in sources:
            value=' '.join(row['cited_paper_abstract']);tokens=re.findall(r'\w+',value.casefold())
            sources[sid]=(hashlib.sha256(' '.join(value.casefold().split()).encode()).digest(),set(tokens),len(tokens))
        source_hash,source_words,source_length=sources[sid]
        for j,(target,label) in enumerate(zip(row['citing_paper_section_text'],row['sentence_labels'])):
            key=fingerprint(source_hash,target)
            if key in seen:
                duplicate+=1;previous=seen[key]
                if int(pool[previous]['label'])!=label:
                    pool[previous]['valid']=False;conflicts.add(key)
                continue
            tokens=re.findall(r'\w+',target.casefold());target_words=set(tokens)
            overlap=len(source_words&target_words)/max(len(source_words|target_words),1)
            priority=int.from_bytes(hashlib.sha256(str(config['seed']).encode()+key).digest()[:8],'big')
            pool[count]=(i,j,family_ids[sid],label,row['doc_label'],overlap,source_length+len(tokens),priority,True)
            seen[key]=count;count+=1
    pool=pool[:count];valid=pool['valid']
    quantile=config['sampling']['high_negative_overlap_quantile'];thresholds={}
    for d in (0,1):
        values=pool['overlap'][valid&(pool['label']==0)&(pool['doc']==d)]
        thresholds[d]=float(np.quantile(values,quantile))
    masks={
        'positive_low_overlap':valid&(pool['label']==1)&(pool['overlap']<=.1),
        'positive_other':valid&(pool['label']==1)&(pool['overlap']>.1),
        'negative_related_high_overlap':valid&(pool['label']==0)&(pool['doc']==0)&(pool['overlap']>=thresholds[0]),
        'negative_positive_section_high_overlap':valid&(pool['label']==0)&(pool['doc']==1)&(pool['overlap']>=thresholds[1]),
        'negative_related_other':valid&(pool['label']==0)&(pool['doc']==0)&(pool['overlap']<thresholds[0]),
        'negative_positive_section_other':valid&(pool['label']==0)&(pool['doc']==1)&(pool['overlap']<thresholds[1]),
    }
    selected=[];all_family=Counter();positive_family=Counter();summary={}
    for name in STRATA:
        indices=np.flatnonzero(masks[name]);quota=config['sampling'][name]
        cuts=np.quantile(pool['length'][indices],[.25,.5,.75]).tolist()
        bins=np.searchsorted(cuts,pool['length'][indices],side='right');counts=[]
        for length_bin in range(4):
            need=quota//4+int(length_bin<quota%4);groups=defaultdict(list)
            for k in indices[bins==length_bin]:
                groups[int(pool[k]['family'])].append(int(k))
            for group in groups.values():group.sort(key=lambda k:int(pool[k]['priority']))
            ordered=sorted(groups,key=lambda family:hashlib.sha256(f"{config['seed']}:{name}:{length_bin}:{families[family]}".encode()).digest())
            cursor=Counter();chosen=[]
            while len(chosen)<need:
                changed=False
                for family in ordered:
                    if cursor[family]>=len(groups[family]):continue
                    if all_family[family]>=config['sampling']['maximum_per_source_family']:continue
                    if name.startswith('positive') and positive_family[family]>=config['sampling']['maximum_positive_per_source_family']:continue
                    k=groups[family][cursor[family]];cursor[family]+=1;chosen.append(k);all_family[family]+=1
                    if name.startswith('positive'):positive_family[family]+=1
                    changed=True
                    if len(chosen)==need:break
                if not changed:raise RuntimeError(f'Insufficient distinct training families in {name}/length{length_bin}; no silent quota change')
            selected.extend((k,name,length_bin) for k in chosen);counts.append(len(chosen))
        summary[name]={'available_unique_examples':len(indices),'selected':quota,'input_word_length_quartiles':cuts,'selected_by_length_quartile':counts}
    entries=[]
    for k,name,length_bin in sorted(selected,key=lambda x:(int(pool[x[0]]['row']),int(pool[x[0]]['sentence']))):
        item=pool[k];row=records[int(item['row'])]
        entries.append({'row':int(item['row']),'original_row':row['original_row'],'sentence':int(item['sentence']),
            'label':int(item['label']),'source_family':row['cited_paper_id'],'stratum':name,'length_quartile':length_bin})
    positive=sum(e['label'] for e in entries);negative=len(entries)-positive
    old_p=manifest['splits']['train']['positives'];old_n=manifest['splits']['train']['sentences']-old_p
    weight=config['original_positive_weight']*(old_p/old_n)*(negative/positive)
    result={'campaign':config['campaign'],'seed':config['seed'],'official_partition':'train','frozen_split_sha256':FROZEN,
        'sampling_configuration':config['sampling'],'examples':len(entries),'positives':positive,'negatives':negative,
        'source_families':len(all_family),'positive_source_families':len(positive_family),
        'largest_family_contribution':max(all_family.values()),'unique_pool_examples':count-int((~valid).sum()),
        'duplicate_inputs_excluded':duplicate,'conflicting_input_keys_excluded':len(conflicts),
        'high_negative_lexical_overlap_threshold_by_doc_label':thresholds,'strata':summary,
        'positive_weight':weight,'weight_derivation':'5 * (58151 / (1440966 - 58151)) * (selected_negative / selected_positive); preserves aggregate weighted class odds, not unbiased within-class sampling.',
        'selection_policy':'Actual training labels only. Low overlap Jaccard<=.10 is the preexisting evaluation definition. Hard negatives use upper training-only lexical-overlap quartile separately in reuse-positive and reuse-negative sections; no model/cosine predictions.',
        'entries':entries}
    save(root/'subset/selection.json',result);verify_selection(root,config,records)
    public={k:v for k,v in result.items() if k!='entries'}
    public['private_selection_sha256']=digest(root/'subset/selection.json')
    save(CODE/'research/results/roberta-practical-selection.json',public)
    print(json.dumps(public,indent=2),flush=True)

def verify_selection(root,config,records=None):
    ensure_frozen(root);path=root/'subset/selection.json';value=json.loads(path.read_text())
    lock=CODE/'configs/roberta_practical_selection.lock.json'
    if lock.exists() and digest(path)!=json.loads(lock.read_text())['selection_sha256']:raise ValueError('The one frozen subset changed')
    if value['campaign']!=config['campaign'] or value['sampling_configuration']!=config['sampling'] or value['frozen_split_sha256']!=FROZEN:raise ValueError('Frozen selection/configuration mismatch')
    if value['seed']!=config['seed']:raise ValueError('Sampling seed changed')
    records=records or load_rows(root,'train');seen=set();keys=set();counts=Counter();families=Counter()
    for e in value['entries']:
        i,j=e['row'],e['sentence'];r=records[i];identity=(i,j)
        assert identity not in seen and r['original_row']==e['original_row'] and r['cited_paper_id']==e['source_family']
        assert r['sentence_labels'][j]==e['label']
        key=fingerprint(hashlib.sha256(' '.join(' '.join(r['cited_paper_abstract']).casefold().split()).encode()).digest(),r['citing_paper_section_text'][j])
        assert key not in keys;keys.add(key);seen.add(identity);counts[e['stratum']]+=1;families[e['source_family']]+=1
    assert len(seen)==value['examples']==30000
    assert counts=={k:config['sampling'][k] for k in STRATA}
    assert max(families.values())<=config['sampling']['maximum_per_source_family']
    positive=sum(e['label'] for e in value['entries']);negative=len(seen)-positive
    assert positive==value['positives']==3000 and negative==value['negatives']==27000
    assert abs(value['positive_weight']-config['original_positive_weight']*58151/(1440966-58151)*negative/positive)<1e-12
    return value
