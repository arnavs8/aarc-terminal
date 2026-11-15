"""Build a portable, offline visual report and complete reproducibility archive."""
import base64
import gzip
import io
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from collections import Counter
import pandas as pd
from backend.stress_benchmark import ROOT, PUBLIC, sha
from backend.stress_report import compact


def main():
    report=json.loads((PUBLIC/'report.json').read_text())
    tasks=json.loads((ROOT/'plan.json').read_text())['tasks']
    rows=[json.loads((ROOT/'cases'/(t['id']+'.json')).read_text()) for t in tasks]
    assert len(rows)==report['case_count'] and report['failed_cases']==0
    counts=Counter((r['episode'],r['market']) for r in rows)
    report['coverage']=[{'episode':e,'market':m,'completed':n,'planned':n,'status':'COMPLETE'} for (e,m),n in counts.items()]
    report['complete_portfolios']={e+'-'+m:r['variants'] for e,ms in report['results'].items() for m,r in ms.items()}
    payload={'report':report,'cases':[compact(r) for r in rows]}
    encoded=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0)).decode()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.style.use('dark_background')
    colors={'current':'#22d3ee','original':'#c084fc','hold':'#fbbf24','matched_long':'#94a3b8'}
    labels={'current':'Current committee','original':'Original committee','hold':'Buy and hold','matched_long':'Always buy, same sizing'}
    charts={};panels=[]
    for e,ms in report['results'].items():
        for m,r in ms.items():
            title=report['config']['periods'][e]['title']+' / '+m.upper()
            fig,ax=plt.subplots(figsize=(10,3.7),layout='constrained')
            for k in colors:
                c=r['variants'][k]['curve']
                ax.plot(pd.to_datetime([x['date'] for x in c]),[x['balance'] for x in c],label=labels[k],color=colors[k],lw=2)
            ax.axhline(10000,color='#64748b',ls='--',lw=1)
            ax.set_title(title+' — starting with 10,000 '+('USD' if m=='us' else 'INR'))
            ax.set_ylabel('Simulated account balance');ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.grid(alpha=.15);ax.legend(fontsize=8,loc='lower right');fig.patch.set_facecolor('#101827');ax.set_facecolor('#101827')
            buf=io.StringIO();fig.savefig(buf,format='svg');plt.close(fig)
            svg=buf.getvalue();svg=svg[svg.index('<svg'):];charts[e+'-'+m]=svg
            table='<table><tr><th>Approach</th><th>Ending balance</th><th>Worst decline</th></tr>'
            for k in ['current','original','hold','matched_long','no_short','price_only','cash']:
                v=r['variants'][k]
                label=labels.get(k,{'no_short':'Committee without shorts','price_only':'Without macro news (exploratory)','cash':'Stay in cash'}.get(k,k))
                table+=f"<tr><td>{label}</td><td>{v['ending_balance']:,.2f}</td><td>{v['max_closing_drawdown_pct']:.2f}%</td></tr>"
            v=r['variants']['current']
            panels.append(f'<article class="card"><h3>{title}</h3>{svg}<p>{v["wins"]} profitable trades out of {v["trades"]}; {r["cases"]-v["trades"]} decisions stayed in cash.</p><div class="scroll">{table}</table></div><p class="muted">Removing macro news is a post-hoc diagnostic, not a predeclared test or a production change.</p></article>')
    page=HTML.replace('DATA_GZIP',encoded).replace('CHART_SVG',charts['covid-us']).replace('ALL_CHARTS',''.join(panels))
    public=Path('frontend/public');(public/'stress-offline.html').write_text(page,encoding='utf8')
    models=PUBLIC/'models';models.mkdir(exist_ok=True)
    for p in (ROOT/'models').glob('*'):shutil.copy2(p,models/p.name)
    outputs=Path('../../outputs');outputs.mkdir(exist_ok=True)
    shutil.copy2(public/'stress-offline.html',outputs/'AARC-crisis-benchmark.html')
    shutil.copy2(PUBLIC/'report.md',outputs/'AARC-crisis-report.md')
    with ZipFile(outputs/'AARC-crisis-evidence.zip','w',ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            if p.is_file() and 'snapshots' not in p.parts and p.suffix not in ('.pending','.interrupted'):
                z.write(p,p.relative_to(ROOT.parent))
        for p in PUBLIC.rglob('*'):
            if p.is_file():z.write(p,p.as_posix())
        for p in Path('backend').glob('stress_*.py'):z.write(p,p.as_posix())
    print(f'Packaged {len(rows)} cases, four charts, four trained models, input data and proof hashes.')

HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AARC crisis benchmark — completed evidence</title><style>
*{box-sizing:border-box}body{margin:0;background:#080e18;color:#e2e8f0;font:16px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px 22px}h1{font-size:clamp(28px,4vw,44px);line-height:1.15;color:white}h2{color:white;font-size:23px}.muted{color:#94a3b8}.warning{border-left:4px solid #fbbf24;padding:12px 18px;background:#1c1a14;color:#fde68a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card,details{border:1px solid #334155;background:#101827;border-radius:12px;padding:20px;margin:16px 0}select{max-width:100%;background:#1e293b;color:#e2e8f0;padding:12px;border:1px solid #475569;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #334155}.scroll{overflow:auto}a{color:#67e8f9}.number{font-size:30px;color:white;font-weight:650}svg{width:100%;height:auto}summary{cursor:pointer;color:#67e8f9}pre{white-space:pre-wrap;font:inherit;font-size:14px}.hash{overflow-wrap:anywhere;font-size:11px;color:#94a3b8}.tag{color:#22d3ee;font-size:12px;text-transform:uppercase;letter-spacing:.12em}section{margin-top:36px}</style><main>
<p class="tag">AARC · historical evidence · complete release</p><h1>How did the committee behave in two crises?</h1><p class="warning"><b>All 356 scheduled decisions completed.</b> COVID and 2022 rate hikes, US and India. These are retrospective simulations, not verified real trades or guaranteed future performance.</p>
<p>This report contains actual recorded outputs, not a hypothetical sales demo. It tests the current rules-based committee; Gemini was not active. No claim is made that the system predicted either crisis in advance.</p><div id="coverage" class="scroll card"></div>
<section><h2>If you started with $10,000</h2><p class="muted">Completed US COVID test, February–June 2020. Four equal starting portions: SPY, AAPL, JPM and XOM. The committee leaves unused money in cash.</p><div id="balances" class="grid"></div><div class="card">CHART_SVG</div><p class="muted">Buy-and-hold invests all the money. A smaller loss with little exposure is not proof of good predictions. “Always buy, same sizing” uses the committee’s entry dates, sizes and stop distances, but always bets on a rise.</p><div id="comparison" class="scroll card"></div></section>
<section><h2>All four market / crisis comparisons</h2>ALL_CHARTS</section><section><h2>Inspect any completed decision</h2><p>Choose a stock/date to see what was known then, what the committee said, and what happened next. Losses and no-trades are included.</p><select id="case" aria-label="Completed decision"></select><div id="detail"></div></section>
<section><h2>How this was tested</h2><ol><li>Freeze daily prices from Yahoo Finance and check dates, missing values and price bounds.</li><li>Train the separate price models only through December 2019 for COVID, and December 2021 for the 2022 test.</li><li>At each date, run production GARCH, market-regime, Isolation Forest and LSTM models on past prices only.</li><li>Run actual FinBERT on already-published, curated summaries of official Fed/WHO announcements. Missing news stays unavailable.</li><li>Run the original and revised committee on identical evidence, then reveal the next five sessions only to the trade evaluator.</li><li>Enter at the next open. Use the stated fixed stop, worse opening-gap fills, dividends, 0.30% round-trip costs and 3% annual short-borrow assumption.</li></ol></section>
<section><h2>What remains unproven</h2><ul><li>Historical data and current rules do not prove future performance. The instruments were selected from surviving large caps.</li><li>These are hypothetical fractional positions, not broker executions. Short availability, Indian overnight-short arrangements, whole-share/lot sizes, taxes, cash interest and funding are not established.</li><li>News coverage is narrow and curated. FinBERT’s wording score is not event truth or an economic forecast. Gemini quality is untested.</li><li>The price predictor is a separate panel: its probabilities do not drive the production CRO decision.</li><li>Closing-balance drawdown can miss worse intraday losses. Yahoo prices may have been revised and use a split-adjusted basis.</li><li>The calibration state is a frozen starting prior. Ratings are not measured probabilities of profit.</li></ul><p>Exact input files, complete events, model artifacts and hashes are supplied in the accompanying evidence archive. The fork includes the report, ledger, metadata, trained artifacts and reproduction code.</p></section><p id="status" class="muted">Loading embedded evidence…</p></main>
<script>
const enc='DATA_GZIP';const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const money=x=>'$'+Number(x).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});const action=x=>x==='LONG'?'Buy':x==='SHORT'?'Bet on a fall (short)':'Wait / keep cash';
(async()=>{const bytes=Uint8Array.from(atob(enc),c=>c.charCodeAt(0));const data=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());const r=data.report;const p=r.complete_portfolios['covid-us'];const names={current:'Current committee',original:'Original committee',hold:'Buy and hold',matched_long:'Always buy, same sizing',no_short:'Current committee, no shorts'};
document.querySelector('#coverage').innerHTML='<table><tr><th>Period</th><th>Market</th><th>Completed</th><th>Status</th></tr>'+r.coverage.map(c=>`<tr><td>${c.episode==='covid'?'COVID 2020':'Rate hikes 2022'}</td><td>${c.market.toUpperCase()}</td><td>${c.completed}/${c.planned}</td><td>${c.status}</td></tr>`).join('')+'</table>';
document.querySelector('#balances').innerHTML=['current','hold','matched_long'].map(k=>`<div class="card"><div class="tag">${names[k]}</div><div class="number">${money(p[k].ending_balance)}</div><p>${p[k].return_pct.toFixed(2)}% over the completed period</p></div>`).join('');
document.querySelector('#comparison').innerHTML='<table><tr><th>Approach</th><th>Ending balance</th><th>Worst closing decline</th></tr>'+Object.keys(names).map(k=>`<tr><td>${names[k]}</td><td>${money(p[k].ending_balance)}</td><td>${p[k].max_closing_drawdown_pct.toFixed(2)}%</td></tr>`).join('')+'<tr><td>Stay in cash</td><td>$10,000.00</td><td>0%</td></tr></table>';
const sel=document.querySelector('#case');data.cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${c.market.toUpperCase()} · ${c.date} · ${c.ticker} · ${action(c.current.action)} · ${c.current.trade.traded?(c.current.trade.net_return>0?'profit':'loss'):'no trade'}`;sel.append(o)});
function render(){const c=data.cases[Number(sel.value)||0],v=c.current,t=v.trade,q=c.quant;let html=`<div class="grid"><div class="card"><p class="tag">1 · Evidence at ${esc(c.date)}</p><h3>${esc(c.ticker)} · close ${q.current_price.toFixed(2)}</h3><p>Trend: ${esc(q.sma20_trend)}. RSI: ${q.rsi_14.toFixed(1)}. Volatility: ${q.annualized_volatility.toFixed(2)}%. Estimated severe daily loss threshold: ${q.daily_VaR_95_pct.toFixed(2)}%.</p><p class="muted">Regime: ${esc(q.hmm_regime.regime_name)}. Recent anomalies: ${q.recent_10d_anomalies}. These are estimates, not guarantees.</p></div><div class="card"><p class="tag">2 · News known then</p><p class="muted">Curated factual paraphrases of official announcements. Available from the next day; not a company-news archive.</p>${c.news.headlines.length?c.news.headlines.map(h=>`<p><a href="${esc(h.source_url)}" target="_blank" rel="noreferrer">${esc(h.title)}</a><br><small>${esc(h.published_at)} · FinBERT: ${esc(h.label)}</small></p>`).join(''):'No curated announcement in the prior 30 days; news unavailable.'}</div></div>`;
html+=`<div class="grid"><div class="card"><p class="tag">3 · Separate price model</p><p>Training ended ${esc(c.forecast.training_data_end)}. Guess: ${esc(c.forecast.prediction)}. Model decision: ${esc(c.forecast.decision)}. Actual future class: ${esc(c.classification_target)}.</p><p>${Object.entries(c.forecast.probabilities).map(([k,x])=>esc(k)+': '+(x*100).toFixed(1)+'%').join(' · ')}</p><p class="muted">This model’s probabilities do not drive the final CRO trade.</p></div><div class="card"><p class="tag">4 · Final answer</p><h3>${action(v.action)} · ${v.allocation_pct}% allocation</h3><p>${esc(v.summary)}</p><details><summary>Actual bull and bear arguments</summary><pre>${esc(v.bull)}</pre><pre>${esc(v.bear)}</pre></details></div></div>`;
html+=`<div class="card"><p class="tag">5 · Reveal the next five sessions</p><div class="scroll"><table><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th></tr>${c.future_bars.map(b=>`<tr><td>${b.Date.slice(0,10)}</td>${[b.Open,b.High,b.Low,b.Close].map(n=>'<td>'+n.toFixed(2)+'</td>').join('')}</tr>`).join('')}</table></div>`;
html+=t.traded?`<p>Entry ${t.entry_date} at ${t.entry_price.toFixed(2)}; exit ${t.exit_date} at ${t.exit_price.toFixed(2)}. Stop hit: ${t.stop_hit?'yes':'no'}.</p><p>On a standalone 10,000-unit account, ${v.allocation_pct}% allocates ${(10000*t.allocation).toFixed(2)} units. The position returned ${(t.net_return*100).toFixed(2)}% after stated costs, dividends and borrowing.</p><p class="number">Net gain/loss: ${(10000*t.weighted_return).toFixed(2)} units</p><p class="muted">The portfolio chart splits the account across four instruments. This single-case example instead applies one recommendation to the full starting account.</p>`:'<p>No position approved. Gain/loss: zero. This is not counted as a winning prediction.</p>';
html+='</div><details><summary>Original answer and proof hashes</summary><p>'+esc(c.original.summary)+'</p><p>Original single-case allocated return: '+(c.original.trade.weighted_return*100).toFixed(3)+'%.</p><p class="hash">Case SHA-256: '+c.full_evidence_sha256+'<br>Price input SHA-256: '+c.input_price_sha256+'<br>Trained model SHA-256: '+c.model_sha256+'</p></details>';document.querySelector('#detail').innerHTML=html}
sel.onchange=render;render();document.querySelector('#status').textContent=`${data.cases.length} completed cases embedded. This file works offline in a modern browser; source links require internet.`;
})().catch(e=>document.querySelector('#status').textContent='Could not load embedded evidence: '+e.message);
</script></html>'''

if __name__=='__main__':main()
