'use client';

import { useEffect, useState } from 'react';

type Verdict = { action: string; recommended_allocation_pct: number; verdict_summary: string; confidence: number };
type Result = { cases: number; trades: number; coverage: number; net_win_rate: number | null; mean_allocated_five_session_return_pct: number; checks_passed: Record<string, number> };
type Replay = { events: { cro: Verdict; bull: { bull_thesis: string }; bear: { bear_thesis: string } }; checks: Record<string, boolean>; execution: { traded: boolean; weighted_return: number; net_trade_return: number; stop_hit: boolean } };
type Case = { ticker: string; date: string; market: string; before: Replay; after: Replay; buy_and_hold_return_pct: number };
type Data = { report: { scope: string; method: string; limitations: string[]; results: Record<string, Record<string, Result>> }; cases: Case[] };
const labels: Record<string, string> = {
  allocation_within_risk_cap: 'Allocation respects the risk cap',
  summary_matches_allocation: 'Written and final allocations agree',
  zero_exposure_is_no_trade: 'Zero exposure is labelled no trade',
  no_unsupported_institutional_flow_claim: 'No invented institutional-flow claim',
  no_unfitted_probability_claim: 'No fixed probabilities labelled calibrated',
  score_disclosed_as_heuristic: 'Score disclosed as a heuristic',
};
const pct = (v: number | null) => v === null ? 'Not measured' : `${(v * 100).toFixed(1)}%`;

export default function BenchmarkPage() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState('');
  const [market, setMarket] = useState('india');
  const [caseId, setCaseId] = useState('');
  useEffect(() => { fetch('/committee-benchmark.json').then(r => { if (!r.ok) throw new Error('Benchmark report unavailable'); return r.json(); }).then(setData).catch(e => setError(String(e))); }, []);
  if (!data) return <main className="p-12 text-white">{error || 'Loading recorded benchmark…'}</main>;
  const results = data.report.results[market];
  const cases = data.cases.filter(c => c.market === market);
  const selected = cases.find(c => `${c.ticker}/${c.date}` === caseId) || cases[0];
  return <main className="min-h-screen bg-[#090d15] text-slate-200 p-6 md:p-12">
    <div className="max-w-6xl mx-auto space-y-8">
      <a href="/" className="text-cyan-300 text-sm">← Back to terminal</a>
      <header className="space-y-3">
        <p className="uppercase tracking-[.2em] text-cyan-300 text-xs">AARC · recorded evaluation</p>
        <h1 className="text-3xl md:text-4xl font-semibold text-white">Can you trust the final answer?</h1>
        <p className="max-w-3xl text-slate-400">Two different questions: does the answer match its evidence, and does its trade work? Passing an evidence check does not prove future accuracy.</p>
        <p className="text-sm border-l-2 border-amber-400 pl-4 text-amber-200">{data.report.scope}</p>
      </header>
      <div className="flex items-center gap-4">
        <label htmlFor="market">Market</label>
        <select id="market" value={market} onChange={e => { setMarket(e.target.value); setCaseId(''); }} className="bg-slate-800 rounded px-4 py-2"><option value="india">India · 2020</option><option value="us">US · 2015–2017</option></select>
        <span className="text-sm text-slate-400">{results.after.cases} paired cases · 30 bps assumed round-trip cost</span>
      </div>
      <section className="grid md:grid-cols-2 gap-4">
        {(['before','after'] as const).map(version => { const r = results[version]; return <div key={version} className="border border-slate-700 rounded-xl p-6 bg-slate-900/70">
          <h2 className="text-lg text-white mb-4">{version === 'before' ? 'Before · original committee' : 'After · audited committee'}</h2>
          <div className="grid grid-cols-3 gap-3">
            <div><p className="text-2xl font-semibold">{r.trades}/{r.cases}</p><p className="text-xs text-slate-400">Cases with exposure</p></div>
            <div><p className="text-2xl font-semibold">{pct(r.net_win_rate)}</p><p className="text-xs text-slate-400">Net win rate on trades</p></div>
            <div><p className="text-2xl font-semibold">{r.mean_allocated_five_session_return_pct.toFixed(3)}%</p><p className="text-xs text-slate-400">Mean allocated return per case</p></div>
          </div>
        </div>; })}
      </section>
      <p className="text-sm text-slate-400">A no-trade decision earns zero and stays in the per-case average. It is excluded from trade win rate. Returns reflect position sizing, five sessions, fixed stops and adverse opening gaps. These are per-decision results, not annual or compounded portfolio returns.</p>
      <section className="border border-slate-700 rounded-xl overflow-hidden">
        <h2 className="p-5 font-semibold text-white">Evidence and consistency · cases passing each check</h2>
        <table className="w-full text-sm"><thead className="bg-slate-800 text-left"><tr><th className="p-3">Check</th><th className="p-3">Before</th><th className="p-3">After</th></tr></thead>
          <tbody>{Object.entries(labels).map(([key,label]) => <tr key={key} className="border-t border-slate-800"><td className="p-3">{label}</td><td className="p-3">{results.before.checks_passed[key]}/{results.before.cases}</td><td className="p-3 text-cyan-300">{results.after.checks_passed[key]}/{results.after.cases}</td></tr>)}</tbody>
        </table>
      </section>
      <section className="space-y-4">
        <h2 className="text-xl text-white">Inspect the actual final answer</h2>
        <select aria-label="Historical case" value={`${selected.ticker}/${selected.date}`} onChange={e => setCaseId(e.target.value)} className="bg-slate-800 rounded px-4 py-2 w-full md:w-auto">{cases.map(c => <option key={`${c.ticker}/${c.date}`} value={`${c.ticker}/${c.date}`}>{c.ticker} · {c.date}</option>)}</select>
        <p className="text-sm text-slate-400">Same-window fully invested long reference, after costs: {selected.buy_and_hold_return_pct.toFixed(2)}%. Different exposure from the committee; this is context, not a risk-matched comparison.</p>
        <div className="grid md:grid-cols-2 gap-4">
          {(['before','after'] as const).map(version => { const c=selected[version]; const v=c.events.cro; return <article key={version} className="border border-slate-700 rounded-xl p-5 space-y-4">
            <p className="uppercase text-xs text-cyan-300">{version}</p>
            <h3 className="font-semibold text-white">{v.action} · {v.recommended_allocation_pct}% allocation</h3>
            <p className="text-sm leading-relaxed">{v.verdict_summary}</p>
            <div className="bg-slate-800 rounded p-3 text-sm">{c.execution.traded ? `Allocated return: ${(c.execution.weighted_return*100).toFixed(3)}%. Trade return after costs: ${(c.execution.net_trade_return*100).toFixed(2)}%. Stop hit: ${c.execution.stop_hit ? 'yes' : 'no'}.` : 'No allocated trade. Return: 0%. No win is credited.'}</div>
            <details><summary className="text-sm cursor-pointer text-slate-300">Read bull and bear arguments</summary><p className="whitespace-pre-line text-sm mt-3">{c.events.bull.bull_thesis}</p><p className="whitespace-pre-line text-sm mt-4">{c.events.bear.bear_thesis}</p></details>
          </article>; })}
        </div>
      </section>
      <section className="space-y-3 border-t border-slate-700 pt-6"><h2 className="text-lg text-white">What remains unproven</h2><ul className="list-disc pl-5 text-sm text-slate-400 space-y-2">{data.report.limitations.map(x => <li key={x}>{x}</li>)}</ul><p className="text-sm text-slate-400">Method: {data.report.method}</p></section>
    </div>
  </main>;
}
