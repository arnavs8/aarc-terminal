"""Pinned cross-corpus evaluation; no fine-tuning on the evaluation labels."""
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from backend.features.financial_lexicon import LoughranMcDonaldVerifier

ROOT=Path('benchmark-runs/challenger');OUT=Path('benchmarks/challenger')


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True,parents=True)
    frame=pd.read_csv(ROOT/'sentiment-validation.csv');texts=frame.text.astype(str).tolist()
    names=['negative','neutral','positive'];labels=[{0:0,1:2,2:1}[int(x)] for x in frame.label]
    manifest=json.loads((ROOT/'model-manifests.json').read_text())
    specs=[next(r for r in manifest if r['id']=='ProsusAI/finbert'),manifest[1]]
    predictions={};results={}
    for spec in specs:
        start=time.perf_counter();model_id=spec['id'];print('Evaluating',model_id,len(texts),flush=True)
        tok=AutoTokenizer.from_pretrained(model_id,revision=spec['revision'],local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(model_id,revision=spec['revision'],local_files_only=True,attn_implementation='sdpa').eval()
        mapping={int(k):str(v).lower() for k,v in model.config.id2label.items()}
        if set(mapping.values())!=set(names):raise ValueError('Unverified label mapping')
        order=[next(k for k,v in mapping.items() if v==n) for n in names];probs=[]
        with torch.inference_mode():
            for i in range(0,len(texts),16):
                x=tok(texts[i:i+16],padding=True,truncation=True,max_length=128,return_tensors='pt')
                probs.extend(torch.softmax(model(**x).logits,dim=-1).cpu().numpy()[:,order].tolist())
                if i%256==0:print(model_id,i,'/',len(texts),flush=True)
        a=np.array(probs);pred=a.argmax(axis=1);gate=LoughranMcDonaldVerifier(confidence_threshold=.75)
        integrated=[]
        for text,row,guess in zip(texts,a,pred):
            v=gate.verify_and_calibrate(text,names[guess],float(row[guess]),float(row[2]-row[0]))
            integrated.append(names.index(v['final_label']))
        def score(p):return {'accuracy':float(accuracy_score(labels,p)),'macro_f1':float(f1_score(labels,p,average='macro')),'confusion_matrix':confusion_matrix(labels,p,labels=[0,1,2]).tolist()}
        results[model_id]={'revision':spec['revision'],'raw':score(pred),'with_existing_lexicon_gate':score(integrated),'seconds':round(time.perf_counter()-start,2)}
        predictions[model_id]=[{'predicted':names[int(p)],'integrated':names[int(g)],'probabilities':dict(zip(names,map(float,row)))} for p,g,row in zip(pred,integrated,a)]
        del model,tok
        import gc;gc.collect()
    result={'dataset':json.loads((ROOT/'sentiment-dataset.json').read_text()),'split':'sent_valid.csv; full provided validation split; no label tuning',
            'rows':len(texts),'source_sha256':hashlib.sha256((ROOT/'sentiment-validation.csv').read_bytes()).hexdigest(),
            'labels':names,'results':results,'scope':'Annotated tweet/news sentiment, not next-day stock returns or company-targeted Qwen reasoning.',
            'limitations':['Both published classifiers report PhraseBank training; unreported pretraining overlap cannot be ruled out.','Single cross-domain dataset, not comprehensive news-quality or trading-profit validation.','Financial text labels describe sentiment, not realized returns.','Latency includes model loading and is not a dedicated serving benchmark.']}
    (OUT/'sentiment-report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    records=[{'id':i,'text':t,'expected':names[labels[i]],'models':{k:v[i] for k,v in predictions.items()}} for i,t in enumerate(texts)]
    with gzip.open(OUT/'sentiment-predictions.json.gz','wt',encoding='utf8') as f:json.dump(records,f)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
