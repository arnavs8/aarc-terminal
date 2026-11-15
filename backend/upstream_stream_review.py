"""Render each repository's actual final SSE payload using frozen quant/news inputs."""
import argparse,asyncio,contextlib,hashlib,io,json,os,sys,tempfile
from copy import deepcopy
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',required=True);p.add_argument('--folder',required=True);args=p.parse_args()
    repo=Path(args.repository).resolve();folder=Path(args.folder).resolve();sys.path.insert(0,str(repo))
    from backend import graph
    assert Path(graph.__file__).resolve().is_relative_to(repo)
    graph._get_gemini_client=lambda:None;graph.RollingAccuracyTracker.log_prediction=lambda *a,**kw:None
    manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    runtime=[repo/'backend/graph.py',repo/'backend/committee_audit.py',*(repo/'backend/calibration').glob('*.py')]
    manifest['stream_code_sha256']={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime if p.exists()}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    paths=[p for p in sorted(folder.glob('*.json')) if p.name!='manifest.json'];assert len(paths)==356
    async def collect(row):
        graph.analyze_ticker_quant=lambda *a:deepcopy(row['quant'])
        graph.analyze_ticker_sentiment=lambda *a:deepcopy(row['news'])
        return {x['node']:x['data'] async for x in graph.stream_committee(row['ticker'],row['date'])}
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp);os.environ['AARC_STATE_DIR']=tmp
        for path in paths:
            row=json.loads(path.read_text(encoding='utf8'))
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):events=asyncio.run(collect(row))
            assert 'cro' in events,events
            row['events']=events;row['stream_payload_reviewed']=True
            path.write_text(json.dumps(row),encoding='utf8')
        os.chdir(repo)
    print('Verified final streamed payloads:',len(paths))


if __name__=='__main__':main()
