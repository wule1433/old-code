from pathlib import Path

p=Path('eftm_strict_open/build_strict_open_dataset.py')
s=p.read_text()
start=s.index('def build_events(')
end=s.index('\ndef attach_statements', start)
new=r'''def build_events(ledger_eps, speb_hist, profile, cal):
    """Build unique events with correct source semantics.

    LEDGER earnings_date is the announcement timestamp/date.
    speb earnings_history.report_date is a snapshot date; `quarter` is the
    fiscal period. Therefore speb event dates come from earning_calendar and
    are joined to earnings_history by symbol + fiscal quarter ending.
    """
    le=ledger_eps.copy(); le.columns=[str(c).strip() for c in le.columns]
    le['symbol']=le[col(le,'ticker')].map(clean_symbol)
    led_ts=pd.to_datetime(le[col(le,'earnings_date')],errors='coerce',utc=True)
    le['event_ts_utc']=led_ts
    le['event_date']=led_ts.dt.tz_convert(None).dt.normalize()
    le['epsEst']=pd.to_numeric(le[col(le,'eps_estimate')],errors='coerce')
    le['epsAct']=pd.to_numeric(le[col(le,'reported_eps')],errors='coerce')
    le['source']='ledger_ccby4'
    le['time_hint']=le.event_ts_utc.map(ledger_time_class)
    le['calendar_fiscal_end']=pd.NaT
    le['release_time']=le['time_hint']
    le=le[['symbol','event_date','event_ts_utc','epsEst','epsAct','source','time_hint','calendar_fiscal_end','release_time']]

    prof=latest_profile(profile)
    tech=set(prof[(prof.sector.str.contains('tech',case=False,na=False)) |
                  (prof.industry.str.contains('semiconductor',case=False,na=False))].symbol)
    tech.update(SEMIS)

    # Calendar is authoritative for speb historical event dates/release timing.
    sc=cal[cal.symbol.isin(tech)].copy()
    sc=sc[sc.release_time.isin(['amc','bmo']) & sc.fiscal_end.notna()].copy()

    sh=speb_hist.copy(); sh.columns=[str(c).strip() for c in sh.columns]
    if len(sh) and len(sc):
        sh['symbol']=sh[col(sh,'symbol')].map(clean_symbol)
        sh=sh[sh.symbol.isin(tech)].copy()
        qcol=col(sh,'quarter')
        if not qcol: raise ValueError(f'speb earnings_history lacks quarter: {list(sh.columns)}')
        sh['fiscal_end']=pd.to_datetime(sh[qcol],errors='coerce').dt.normalize()
        sh['epsEst']=pd.to_numeric(sh[col(sh,'eps_estimate')],errors='coerce')
        sh['epsAct']=pd.to_numeric(sh[col(sh,'eps_actual')],errors='coerce')
        snap=col(sh,'report_date')
        if snap:
            sh['_snapshot']=pd.to_datetime(sh[snap],errors='coerce')
            sh=sh.sort_values(['symbol','fiscal_end','_snapshot']).drop_duplicates(['symbol','fiscal_end'],keep='last')
        else:
            sh=sh.drop_duplicates(['symbol','fiscal_end'],keep='last')
        z=sc.merge(sh[['symbol','fiscal_end','epsEst','epsAct']],on=['symbol','fiscal_end'],how='inner')
        z['event_ts_utc']=pd.NaT
        z['source']='speb_ccby4_technology'
        z['time_hint']=np.nan
        z['calendar_fiscal_end']=z['fiscal_end']
        sp=z[['symbol','event_date','event_ts_utc','epsEst','epsAct','source','time_hint','calendar_fiscal_end','release_time']]
    else:
        sp=pd.DataFrame(columns=le.columns)

    ev=pd.concat([le,sp],ignore_index=True).dropna(subset=['symbol','event_date','epsAct'])

    # A same-date speb calendar row may provide fiscal-period metadata for
    # LEDGER. It may only fill a missing release class; never override a valid
    # LEDGER timestamp classification.
    ledmask=ev.source.eq('ledger_ccby4')
    if ledmask.any() and len(cal):
        base=ev.loc[ledmask,['symbol','event_date','event_ts_utc','epsEst','epsAct','source','time_hint']].copy()
        upd=asof_calendar(base,cal)
        ev.loc[ledmask,'calendar_fiscal_end']=upd['calendar_fiscal_end'].to_numpy()
        current=ev.loc[ledmask,'release_time'].to_numpy(dtype=object)
        fallback=upd['calendar_release'].to_numpy(dtype=object)
        ev.loc[ledmask,'release_time']=[v if pd.notna(v) else fallback[i] for i,v in enumerate(current)]

    merged=[]
    for (sym,d),g in ev.groupby(['symbol','event_date'],sort=False):
        g=g.sort_values('source',key=lambda x:x.eq('speb_ccby4_technology'),ascending=False)
        r=g.iloc[0].copy()
        r['source']='+'.join(sorted(set(g.source)))
        if pd.isna(r.get('release_time')):
            vals=g.release_time.dropna(); r['release_time']=vals.iloc[0] if len(vals) else np.nan
        if pd.isna(r.get('calendar_fiscal_end')):
            vals=g.calendar_fiscal_end.dropna(); r['calendar_fiscal_end']=vals.iloc[0] if len(vals) else pd.NaT
        merged.append(r)
    ev=pd.DataFrame(merged)
    ev=ev[ev.release_time.isin(['amc','bmo'])].copy()
    ev['epsSurpriseAbs']=ev.epsAct-ev.epsEst
    ev['epsSurprise_pct']=np.where(ev.epsEst.notna() & ev.epsEst.ne(0),(ev.epsAct-ev.epsEst)/ev.epsEst.abs(),np.nan)
    ev=ev.sort_values(['symbol','event_date'])
    ev['epsAct_YoY']=np.nan
    for sym,g in ev.groupby('symbol'):
        for i,r in g.iterrows():
            prior=g[(g.event_date<r.event_date)&((r.event_date-g.event_date).dt.days.between(300,430))]
            if len(prior):
                pr=prior.iloc[-1]
                if pd.notna(r.epsAct) and pd.notna(pr.epsAct) and pr.epsAct!=0:
                    ev.at[i,'epsAct_YoY']=r.epsAct/pr.epsAct-1
    return ev,tech,prof
'''
s=s[:start]+new+s[end:]
s=s.replace("dataset=ds.dataset(str(d),format='parquet',partitioning='hive')","dataset=ds.dataset(str(d),format='parquet')")
s=s.replace("spp=normalize_prices(load_speb_prices(sp,tech),'speb_ccby4')","spp=normalize_prices(load_speb_prices(sp,set(ev.symbol.dropna().astype(str))),'speb_ccby4')")
p.write_text(s)
print('patched',p)
