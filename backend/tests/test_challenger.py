import numpy as np
import pandas as pd
import pytest
from backend.challenger_benchmark import rebalance,simulate
from backend.models.timestral_forecast import decide
from backend.tools.qwen_news import validate_response,analyze


def test_qwen_retries_only_provider_json_failure(monkeypatch):
    import json
    import httpx
    from backend.tools import qwen_news as q
    monkeypatch.setenv('GROQ_API_KEY','test-placeholder')
    monkeypatch.setattr(q.time,'sleep',lambda _:None)
    answers=[httpx.Response(400,json={'error':{'code':'json_validate_failed'}}),httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'items':[{'id':0,'label':'positive','relevance':'company','evidence_quote':'profit rises','explanation':'Reported profit increased.'}]})}}]})]
    calls=[]
    def post(*a,**kw):calls.append(kw);return answers.pop(0)
    monkeypatch.setattr(q.httpx,'post',post)
    result=q.analyze('EXAMPLE',[{'title':'Example profit rises'}])
    assert result['available'] and result['provider_attempts']==2
    assert calls[0]['json']==calls[1]['json']


def test_qwen_quota_error_is_not_retried(monkeypatch):
    import httpx
    from backend.tools import qwen_news as q
    monkeypatch.setenv('GROQ_API_KEY','test-placeholder')
    monkeypatch.setattr(q.time,'sleep',lambda _:None)
    calls=[]
    def post(*a,**kw):calls.append(1);return httpx.Response(429,json={'error':{'code':'rate_limit_exceeded'}})
    monkeypatch.setattr(q.httpx,'post',post)
    assert not q.analyze('EXAMPLE',[{'title':'Example profit rises'}])['available']
    assert len(calls)==1


def test_rebalance_preserves_cash_and_charges_only_turnover():
    cash,shares,fee=rebalance(10000,0,100,1,.0015)
    assert cash==pytest.approx(0)
    assert shares*100+fee==pytest.approx(10000)
    c,s,f=rebalance(cash,shares,100,1,.0015)
    assert f==pytest.approx(0)
    c,s,f=rebalance(c,s,90,0,.0015)
    assert s==0 and c==pytest.approx(shares*90*(1-.0015))


def test_next_open_execution_and_dividends_go_to_prior_holders():
    dates=pd.bdate_range('2024-01-01',periods=4)
    f=pd.DataFrame({'Open':[100,100,100,100],'Close':[100,100,100,100],'Dividends':[0,10,1,0]},index=dates)
    r={'date':'2024-01-01','timestral':{'sleeve_weight':1}}
    out=simulate(f,[r],2024,'timestral',0,10000)
    assert out['ending_balance']==10100  # No dividend on entry date; held for next ex-date.
    assert out['ledger'][0]['execution_date']=='2024-01-02'
    assert out['curve'][0]['balance']==10000


def test_forecast_really_changes_the_decision_without_leverage():
    f=pd.DataFrame({'Close':np.linspace(100,200,252)},index=pd.bdate_range('2023-01-01',periods=252))
    good={'available':True,'median_return':.02};bad={'available':True,'median_return':-.02}
    assert decide(f,good)['action']=='LONG'
    assert 0<decide(f,good)['sleeve_weight']<=1
    assert decide(f,bad)['action']=='REJECT_TRADE'
    assert decide(f,bad,use_forecast=False)['action']=='LONG'


def test_qwen_rejects_invented_quotes_and_duplicate_evidence():
    h=[{'title':'Company profit fell.'}]
    item={'id':0,'label':'negative','relevance':'company','evidence_quote':'profit fell','explanation':'Profit decreased.'}
    assert validate_response({'items':[item]},h)[0]['label']=='negative'
    with pytest.raises(ValueError):validate_response({'items':[{**item,'evidence_quote':'profit rose'}]},h)
    with pytest.raises(ValueError):validate_response({'items':[item,item]},h*2)


def test_macro_wording_does_not_manufacture_a_company_direction():
    h=[{'title':'Fed raises rates.'}]
    item={'id':0,'label':'positive','relevance':'macro','evidence_quote':'raises rates','explanation':'Rates increased.'}
    assert validate_response({'items':[item]},h)[0]['label']=='uncertain'


def test_missing_qwen_key_is_explicit_and_does_not_call_network(monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    monkeypatch.setattr('backend.tools.qwen_news.httpx.post',lambda *a,**k:pytest.fail('Unexpected network call'))
    assert analyze('AAPL',[{'title':'Profit fell.'}])['available'] is False
