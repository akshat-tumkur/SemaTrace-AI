# Standalone Colab cell: run once before rerunning prepare/cell4.
from pathlib import Path
import hashlib, json, tempfile
practical = Path(globals().get('ROOT', '/content/drive/MyDrive/SemaTrace-private/arc-roberta-practical-v1'))
original = Path(globals().get('PRIOR', '/content/drive/MyDrive/SemaTrace-private/arc-roberta-v1'))
expected = {
    'train': '6f2f554837e64a5dfed4952d6c4a4c77e42743d0f157a82fc8897750f92b8ec8',
    'val': '77a1df2b17601e3c0bcc63820b97bb681769d879559d0d94029593caded213a8',
    'final': '43bcd85df3e2032e6990e01197208bb8de79417dce722fa7eb70a1d870486cb6',
}
blocked = ['pilot.json','last.json','progress.json','training-complete.json','selection.json','test-opened.json','best','selected-epochs','checkpoints']
assert practical.resolve() != original.resolve(), 'Campaign folders must differ'
assert not any((practical/n).exists() for n in blocked), 'Model work/seal exists: stop'
assert not (original/'test-opened.json').exists(), 'Original final seal exists: stop'
assert not (practical.parent/'sematrace-roberta-final-seal.json').exists(), 'Shared final seal exists: stop'
sha = lambda b: hashlib.sha256(b).hexdigest()
def checked(folder):
    manifest_bytes = (folder/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert set(manifest['splits']) == set(expected), 'Unexpected partitions'
    for split, frozen_hash in expected.items():
        raw = (folder/(split+'.jsonl')).read_bytes()
        assert sha(raw) == manifest['splits'][split]['sha256'], split+': manifest/file mismatch'
        lf = raw.replace(b'\r\n', b'\n')
        assert b'\r' not in lf and sha(lf.replace(b'\n', b'\r\n')) == frozen_hash, split+': not a newline-only difference'
    return manifest, sha(manifest_bytes)
manifest, source_manifest_hash = checked(original/'prepared')
if (practical/'prepared/manifest.json').exists(): checked(practical/'prepared')
practical.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=practical, prefix='newline-repair-') as staging:
    staging = Path(staging)
    audit = {}
    for split, frozen_hash in expected.items():
        raw = (original/'prepared'/(split+'.jsonl')).read_bytes()
        canonical = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        assert sha(raw) == manifest['splits'][split]['sha256'] and sha(canonical) == frozen_hash
        (staging/(split+'.jsonl')).write_bytes(canonical)
        audit[split] = {'source_sha256': sha(raw), 'frozen_sha256': frozen_hash, 'newline_only_conversion': raw != canonical}
        manifest['splits'][split]['sha256'] = frozen_hash
    manifest['serialization'] = {'format':'UTF-8 JSONL with CRLF record separators','source_manifest_sha256':source_manifest_hash,'verified_source_files':audit}
    (staging/'manifest.json').write_bytes(json.dumps(manifest, indent=2).encode('utf-8'))
    destination = practical/'prepared'
    destination.mkdir(exist_ok=True)
    for name in [s+'.jsonl' for s in expected]+['manifest.json']:
        (staging/name).replace(destination/name)
checked(practical/'prepared')
print('Verified frozen CRLF files installed in the practical campaign. Original files unchanged. Rerun cell4; no reinstall/restart needed.')
