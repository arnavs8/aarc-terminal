"""Package allowlisted public evidence, never runtime secrets or account state."""
import hashlib
import json
from pathlib import Path
import zipfile
from dotenv import dotenv_values


def main():
    files=[]
    for folder in ['benchmark-runs/challenger/prices','benchmark-runs/challenger/forecast-cases','benchmark-runs/challenger/qwen-classification','benchmark-runs/strategy','benchmark-runs/flash-crash','benchmarks/challenger','benchmarks/strategy','benchmarks/flash-crash']:
        files.extend(p for p in Path(folder).rglob('*') if p.is_file())
    for name in ['model-manifests.json','price-manifest.json','sentiment-dataset.json','sentiment-validation.csv','qwen-provider-errors.jsonl','live-qwen-news-v3.json']:
        p=Path('benchmark-runs/challenger')/name
        if p.exists():files.append(p)
    files += [p for p in Path('backend').rglob('*.py') if not any(x in p.parts for x in ('.venv','venv','__pycache__','data_cache'))]
    files += [Path('RESEARCH_RESULTS.md'),Path('CHALLENGER_MODELS.md'),Path('backend/requirements-challenger.txt')]
    secrets=[v.encode() for k,v in dotenv_values('.env').items() if ('KEY' in k or 'TOKEN' in k) and v and len(v)>15]
    manifest=[];output=Path('benchmarks/evidence/AARC-research-evidence.zip');output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in sorted(set(files)):
            content=p.read_bytes()
            if any(secret in content for secret in secrets):raise ValueError('Secret detected in allowlisted evidence: '+p.as_posix())
            archive.writestr(p.as_posix(),content)
            manifest.append({'path':p.as_posix(),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        archive.writestr('RESEARCH-EVIDENCE-MANIFEST.json',json.dumps(manifest,indent=2))
    print('Packaged',len(manifest),'files;',output.stat().st_size,'bytes; SHA256',hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__':main()
