from pathlib import Path
import nbformat as nbf

out=Path(__file__).with_name('rogii_contact_forensics_v3.ipynb')
cells=[]
def md(x): cells.append(nbf.v4.new_markdown_cell(x))
def code(x): cells.append(nbf.v4.new_code_cell(x))

md('''# ROGII contact forensics v3

## tl;dr

This is the final targeted research attempt around the known `7.043` solution. It reproduces the contact equation from the public anchor, evaluates all six geological surfaces and several offset calibrations, and compares legal prefix-only calibration against the public-overlap oracle calibration. No submission is written.''')
md('''## Contact equation

For formation `c`:

`raw_TVT = typewell_contact_TVT(c) - (Z - formation_depth(c))`

The original public anchor calibrated a global offset from the full train copy, then interpolated the resulting path by MD. Here we test whether the visible prefix can reproduce that offset and whether another formation/calibration beats `EGFDU` on exact public copies.''')
code(r'''from pathlib import Path
import json, warnings
import numpy as np,pandas as pd
warnings.filterwarnings('ignore')
def root():
    ps=[Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),Path('/kaggle/input/rogii-wellbore-geology-prediction'),Path.cwd()/'datasets',*(parent/'datasets' for parent in list(Path.cwd().parents)[:4])]
    return next((p for p in ps if (p/'train').exists() and (p/'test').exists()),None)
ROOT=root();
if ROOT is None: raise FileNotFoundError('dataset root not found')
FORM=['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA']; METHODS=['prefix_mean','prefix_median','prefix_trimmed','prefix_linear','oracle_mean']
def ids(split): return sorted(p.name.split('__')[0] for p in (ROOT/split).glob('*__horizontal_well.csv'))
def load(w,split):
    b=ROOT/split; return pd.read_csv(b/f'{w}__horizontal_well.csv'),pd.read_csv(b/f'{w}__typewell.csv')
def rmse(a,b): return float(np.sqrt(np.nanmean((np.asarray(a,float)-np.asarray(b,float))**2)))
TRAIN=ids('train'); TEST=ids('test'); print('ROOT',ROOT,'train',len(TRAIN),'test',len(TEST))
''')
md('## Candidate generator')
code(r'''def ref_tvt(tw,ref):
    if not {'Geology','TVT'}.issubset(tw.columns): return np.nan
    x=tw.loc[tw.Geology.astype(str)==ref,'TVT']
    return float(x.min()) if len(x) else np.nan
def raw_contact(hw,tw,ref):
    rt=ref_tvt(tw,ref)
    if not np.isfinite(rt) or ref not in hw: return np.full(len(hw),np.nan)
    return rt-(hw.Z.to_numpy(float)-hw[ref].to_numpy(float))
def correction(hw,raw,method):
    known=hw.TVT_input.notna().to_numpy() & np.isfinite(raw); d=hw.TVT_input.to_numpy(float)-raw; md=hw.MD.to_numpy(float)
    if method=='oracle_mean': known=hw.TVT.notna().to_numpy() & np.isfinite(raw); d=hw.TVT.to_numpy(float)-raw
    x=d[known]; z=md[known]
    if len(x)<30: return np.nan
    if method in ('prefix_mean','oracle_mean'): return float(np.nanmean(x))
    if method=='prefix_median': return float(np.nanmedian(x))
    if method=='prefix_trimmed':
        lo,hi=np.nanpercentile(x,[10,90]); return float(np.nanmean(x[(x>=lo)&(x<=hi)]))
    if method=='prefix_linear':
        tail=min(500,len(x)); zz=z[-tail:]; xx=x[-tail:];
        try: return (float(np.polyfit(zz-zz[-1],xx,1)[1]),float(np.polyfit(zz-zz[-1],xx,1)[0]))
        except Exception: return (float(np.nanmedian(xx)),0.)
def predict(hw,tw,ref,method):
    raw=raw_contact(hw,tw,ref); c=correction(hw,raw,method)
    if not isinstance(c,tuple) and not np.isfinite(c): return np.full(len(hw),np.nan),np.nan
    if isinstance(c,tuple): off,slope=c; pred=raw+off+slope*(hw.MD.to_numpy(float)-float(hw.MD.iloc[-1]))
    else: off=c; pred=raw+off
    return pred,c
''')
md('## Full train suffix OOF screen')
code(r'''rows=[]
for w in TRAIN:
    hw,tw=load(w,'train'); suffix=hw.TVT_input.isna().to_numpy()
    for ref in FORM:
        for method in METHODS[:-1]:
            p,c=predict(hw,tw,ref,method); ok=suffix&np.isfinite(p)&hw.TVT.notna().to_numpy()
            if ok.sum(): rows.append({'well_id':w,'variant':ref+'|'+method,'rmse':rmse(hw.loc[ok,'TVT'],p[ok]),'rows':int(ok.sum()),'prefix_correction':str(c)})
OOF=pd.DataFrame(rows); OOF_SCORE=OOF.groupby('variant').apply(lambda g:pd.Series({'pooled_rmse':rmse(np.repeat(0,len(g)),np.repeat(0,len(g))) if False else np.sqrt(np.average(g.rmse**2,weights=g.rows)),'mean_well_rmse':g.rmse.mean(),'median_well_rmse':g.rmse.median(),'coverage':g.rows.sum()}),include_groups=False).reset_index().sort_values('pooled_rmse')
display(OOF_SCORE.head(20))
''')
md('## Exact public train-copy forensic')
code(r'''public=[]; details=[]
for w in TEST:
    te,_=load(w,'test'); tr,tw=load(w,'train'); mask=te.TVT_input.isna().to_numpy(); truth=tr.TVT.to_numpy(float); row={'well_id':w,'rows':int(mask.sum())}
    for ref in FORM:
        for method in METHODS:
            p,c=predict(tr,tw,ref,method); ok=mask&np.isfinite(p); key=ref+'|'+method; row[key]=rmse(truth[ok],p[ok]) if ok.sum() else np.nan
            details.append({'well_id':w,'ref':ref,'method':method,'rmse':row[key],'prefix_rmse':rmse(tr.loc[tr.TVT_input.notna(),'TVT_input'],p[tr.TVT_input.notna()]) if np.isfinite(p[tr.TVT_input.notna()]).all() else np.nan})
    public.append(row)
PUBLIC=pd.DataFrame(public); DETAIL=pd.DataFrame(details); display(PUBLIC.T.head(40));
summary=[]
for ref in FORM:
    for method in METHODS:
        key=ref+'|'+method; vals=PUBLIC[key].to_numpy(float); summary.append({'variant':key,'pooled_rmse':float(np.sqrt(np.average(vals**2,weights=PUBLIC.rows))),'mean_well_rmse':float(np.mean(vals)),'max_well_rmse':float(np.max(vals))})
PUBLIC_SCORE=pd.DataFrame(summary).sort_values('pooled_rmse'); display(PUBLIC_SCORE)
''')
md('## Prefix-selected versus oracle-selected surface')
code(r'''selected=[]
for w in TEST:
    d=DETAIL[DETAIL.well_id==w]
    pref=d[d.method=='prefix_mean'].sort_values('prefix_rmse').iloc[0]
    oracle=d[d.method=='oracle_mean'].sort_values('rmse').iloc[0]
    selected.append({'well_id':w,'prefix_selected':pref.ref+'|'+pref.method,'prefix_visible_rmse':pref.prefix_rmse,'prefix_suffix_rmse':float(PUBLIC.loc[PUBLIC.well_id==w,pref.ref+'|prefix_mean'].iloc[0]),'oracle_selected':oracle.ref+'|'+oracle.method,'oracle_suffix_rmse':oracle.rmse})
SELECTED=pd.DataFrame(selected); display(SELECTED)
best=PUBLIC_SCORE.iloc[0]
summary={'best_public_variant':str(best.variant),'best_public_pooled_rmse':float(best.pooled_rmse),'public_scores':PUBLIC_SCORE.to_dict('records'),'prefix_selected':SELECTED.to_dict('records'),'oof_scores':OOF_SCORE.to_dict('records'),'notes':['oracle_mean uses full train TVT only for forensic analysis of exact public copies.','prefix methods use only visible TVT_input.','No submission is created.']}
work=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd(); OOF_SCORE.to_csv(work/'contact_v3_oof_scores.csv',index=False); PUBLIC_SCORE.to_csv(work/'contact_v3_public_scores.csv',index=False); DETAIL.to_csv(work/'contact_v3_public_detail.csv',index=False); SELECTED.to_csv(work/'contact_v3_selection.csv',index=False); (work/'contact_v3_summary.json').write_text(json.dumps(summary,indent=2)); print('BEST',best.variant,best.pooled_rmse)
''')
nb=nbf.v4.new_notebook(); nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}; nb.cells=cells; out.write_text(nbf.writes(nb),encoding='utf-8'); print(out)
