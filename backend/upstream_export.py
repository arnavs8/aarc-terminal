"""Export reproducible model comparisons without credentials or duplicate candles."""
import hashlib,json,zipfile
from pathlib import Path
from dotenv import dotenv_values


def main():
    files=list(Path('benchmarks/upstream-comparison').glob('*'))+list(Path('backend').glob('upstream_*.py'))+[Path('UPSTREAM_COMPARISON.md')]
    files+=list(Path('benchmark-runs/upstream-comparison').rglob('*.json'))
    secret_values=[v.encode() for k,v in dotenv_values('.env').items() if ('KEY' in k or 'TOKEN' in k) and v and len(v)>15]
    entries=[];output=Path('benchmarks/evidence/AARC-upstream-comparison-evidence.zip')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(set(files)):
            if not path.is_file():continue
            raw=path.read_bytes();packed=raw;note=None
            if str(path).replace('\\','/').startswith('benchmark-runs/') and path.name!='manifest.json':
                data=json.loads(raw);data.get('quant',{}).pop('chart_data',None)
                data['events']={k:v for k,v in data.get('events',{}).items() if k in ('bull','bear','cro')}
                packed=json.dumps(data).encode();note='Duplicate chart candles and repeated quant/news SSE copies omitted; scalar/nested model features and final answers preserved. Frozen candles are in AARC-crisis-evidence.zip.'
            assert not any(secret in packed for secret in secret_values),'Secret in evidence input'
            archive.writestr(path.as_posix(),packed)
            entries.append({'path':path.as_posix(),'sha256':hashlib.sha256(packed).hexdigest(),'original_worker_sha256':hashlib.sha256(raw).hexdigest(),'note':note})
        archive.writestr('UPSTREAM-EVIDENCE-MANIFEST.json',json.dumps(entries,indent=2))
    print('Exported',len(entries),'files;',output.stat().st_size,'bytes; sha256',hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__':main()
