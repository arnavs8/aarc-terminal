"""Exploratory component check: rerun saved decisions with macro news unavailable.

This is explicitly post-hoc after inspecting FinBERT labels. It does not tune the model.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from backend import graph
from backend.stress_benchmark import ROOT,replay_nodes,trade_path
from backend.calibration.bayesian_calibrator import BetaBinomialConfidenceCalibrator as Cal
from backend.calibration.rolling_tracker import RollingAccuracyTracker
import pandas as pd


def main():
    tasks=json.loads((ROOT/'plan.json').read_text())['tasks'];changed=[]
    with tempfile.TemporaryDirectory() as temp:
        class Tracker(RollingAccuracyTracker):
            def __init__(self):super().__init__(str(Path(temp)/'rolling.json'))
            def log_prediction(self,**kwargs):pass
        graph._get_gemini_client=lambda:None
        graph.RollingAccuracyTracker=Tracker
        graph.BetaBinomialConfidenceCalibrator=lambda:Cal(storage_path=str(Path(temp)/'bayes.json'))
        for t in tasks:
            p=ROOT/'cases'/(t['id']+'.json');row=json.loads(p.read_text())
            news={'headlines':[],'aggregate_sentiment_score':None,'news_available':False,'news_error':'Exploratory news-removal comparison'}
            events=replay_nodes(graph,row['quant'],news);v=events['cro'];original=row['current']['events']['cro']
            future=pd.DataFrame(row['future_bars']);future.Date=pd.to_datetime(future.Date);future=future.set_index('Date')
            row['price_only']={'events':events,'trade':trade_path(v,future)}
            if not row['news']['news_available']:
                assert (v['action'],v['recommended_allocation_pct'])==(original['action'],original['recommended_allocation_pct'])
            if v['action']!=original['action']:changed.append(row['id'])
            p.write_text(json.dumps(row),encoding='utf8')
    result={'status':'Exploratory post-hoc component check, not a predeclared confirmation test.',
            'question':'How did adding the curated macro summaries change otherwise identical committee decisions?',
            'changed_actions':changed,'cases':len(tasks),'production_rules_changed':False}
    (ROOT/'news-ablation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Exploratory news-removal check:',len(changed),'changed actions of',len(tasks))

if __name__=='__main__':main()
