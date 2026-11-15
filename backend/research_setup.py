"""Download reviewed sentiment models/data. Market inputs ship in evidence archive."""
import hashlib
import json
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download,snapshot_download


def main():
    root=Path('benchmark-runs/challenger');root.mkdir(parents=True,exist_ok=True)
    sources=json.loads(Path('benchmarks/challenger/model-sources.json').read_text())
    for model in sources:
        if model['id']=='neoyipeng/ModernFinBERT-base':continue
        snapshot_download(model['id'],revision=model['revision'],allow_patterns=['*.json','*.txt','*.safetensors','*.model','LICENSE','README.md'])
    (root/'model-manifests.json').write_text(json.dumps(sources,indent=2))
    spec={'id':'zeroshot/twitter-financial-news-sentiment','revision':'ccbe24de388e287beb92dd393a335c376b350ac3'}
    src=hf_hub_download(spec['id'],'sent_valid.csv',repo_type='dataset',revision=spec['revision'])
    shutil.copyfile(src,root/'sentiment-validation.csv')
    (root/'sentiment-dataset.json').write_text(json.dumps(spec,indent=2))
    print('Prepared pinned models and validation CSV:',hashlib.sha256((root/'sentiment-validation.csv').read_bytes()).hexdigest())


if __name__=='__main__':main()
