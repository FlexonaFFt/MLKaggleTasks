from pathlib import Path
import nbformat as nbf

out = Path(__file__).with_name('rogii_research_hypotheses_v1.ipynb')
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md('# ROGII research notebook - hypothesis screen v1\n\nDiagnostic only: no submission is created. It compares well-grouped pseudo-test suffix forecasts for baseline, direction, neighbor copy, midpoint hedge, and guarded blends.\n\nOutputs: `research_scores.csv`, `research_well_scores.csv`, `research_summary.json`.')
md('## Context & Methods\n\n`TVT_input` is treated as the visible prefix and missing `TVT_input` rows as the validation suffix. GroupKFold keeps complete wells together. Neighbor references are restricted to the training fold. The midpoint branch is a bounded proxy for the reported +/-15 ft near-tie hedge.')
code(r'''from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

def find_root():
    roots = [Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction'), Path('/kaggle/input/rogii-wellbore-geology-prediction'), Path.cwd()/'datasets', *(parent/'datasets' for parent in list(Path.cwd().parents)[:4])]
    for p in roots:
        if (p/'train').exists() and (p/'test').exists(): return p
    raise FileNotFoundError('dataset root not found')
ROOT=find_root(); SEED=42; N_SPLITS=5
def ids(split): return sorted(p.name.split('__')[0] for p in (ROOT/split).glob('*__horizontal_well.csv'))
def load(wid): return pd.read_csv(ROOT/'train'/f'{wid}__horizontal_well.csv'), pd.read_csv(ROOT/'train'/f'{wid}__typewell.csv')
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))
def prefix(hw):
    k=hw.TVT_input.notna()
    return (hw.loc[k].copy(),hw.loc[~k].copy()) if k.sum()>=20 and (~k).sum()>=20 else None
TRAIN=ids('train'); TEST=ids('test'); W={w:load(w) for w in TRAIN}
usable=[w for w in TRAIN if prefix(W[w][0]) is not None]
meta=[]
for w in usable:
    hw,_=W[w]; k,s=prefix(hw); t=k.tail(min(50,len(k)))
    meta.append({'well_id':w,'heel_x':hw.X.iloc[0],'heel_y':hw.Y.iloc[0],'last_md':k.MD.iloc[-1],'last_tvt':k.TVT_input.iloc[-1],'azimuth':np.arctan2(t.Y.iloc[-1]-t.Y.iloc[0],t.X.iloc[-1]-t.X.iloc[0])})
META=pd.DataFrame(meta).set_index('well_id')
print('root=',ROOT,'train=',len(TRAIN),'usable=',len(usable),'test=',len(TEST))
''')
md('## Deterministic candidates')
code(r'''def slope(k):
    t=k.tail(80); md=t.MD.diff().to_numpy(float); dt=t.TVT_input.diff().to_numpy(float); ok=(md>0)&np.isfinite(dt)
    return float(np.median(dt[ok]/md[ok])) if ok.sum() else 0.0
def candidates(w):
    hw,_=W[w]; k,s=prefix(hw); s=s.iloc[np.linspace(0,len(s)-1,min(len(s),1200),dtype=int)]; md=s.MD.to_numpy(float)-float(k.MD.iloc[-1]); last=np.full(len(s),float(k.TVT_input.iloc[-1])); sl=last+slope(k)*md; mid=(last+sl)/2
    return pd.DataFrame({'well_id':w,'target':s.TVT.to_numpy(float),'last_known':last,'recent_slope':sl,'midpoint_gate':np.where(np.abs(sl-last)<=20,mid,last),'row_index':s.index})
DET=pd.concat([candidates(w) for w in usable],ignore_index=True)
''')
md('## Nearest-well copy')
code(r'''def nearest(w,refs):
    q=META.loc[w,['heel_x','heel_y']].to_numpy(float); a=META.loc[refs,['heel_x','heel_y']].to_numpy(float); d=np.sqrt(((a-q)**2).sum(1)); return refs[int(d.argmin())],float(d.min())
def neighbor(w,refs):
    hw,_=W[w]; k,s=prefix(hw); s=s.iloc[np.linspace(0,len(s)-1,min(len(s),1200),dtype=int)]; r,dist=nearest(w,refs); rhw,_=W[r]; rk,rs=prefix(rhw); q=s.MD.to_numpy(float)-float(k.MD.iloc[-1]); x=rhw.MD.to_numpy(float)-float(rk.MD.iloc[-1]); y=rhw.TVT.to_numpy(float); p=np.interp(q,x,y,left=y[0],right=y[-1])+float(k.TVT_input.iloc[-1])-float(rk.TVT.iloc[-1]); return p,dist
''')
md('## Direction-aware residual model')
code(r'''BASE=['md_since','frac','X','Y','Z','GR','ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA','last_tvt','last_md','recent_slope']; DIR=BASE+['azimuth_sin','azimuth_cos']
def features(w):
    hw,_=W[w]; k,s=prefix(hw); s=s.iloc[np.linspace(0,len(s)-1,min(len(s),1200),dtype=int)]; last=k.iloc[-1]; md=s.MD.to_numpy(float)-float(last.MD); f=pd.DataFrame({'md_since':md,'frac':md/max(float(md[-1]),1),'last_tvt':float(last.TVT_input),'last_md':float(last.MD),'recent_slope':slope(k),'azimuth_sin':np.sin(META.loc[w,'azimuth']),'azimuth_cos':np.cos(META.loc[w,'azimuth']),'target_delta':s.TVT.to_numpy(float)-float(last.TVT_input)})
    for c in ['X','Y','Z','GR','ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA']: f[c]=s[c].to_numpy(float)
    return f
def model_predict(train_ids,valid_ids,cols):
    tr=pd.concat([features(w) for w in train_ids],ignore_index=True); tr=tr.sample(min(len(tr),100000),random_state=SEED); med=tr[cols].median().fillna(0); x=tr[cols].fillna(med).to_numpy(float); y=tr.target_delta.to_numpy(float); mu=x.mean(0); sd=x.std(0); sd[sd==0]=1; xs=(x-mu)/sd; xd=np.c_[np.ones(len(xs)),xs]; pen=np.eye(xd.shape[1])*10; pen[0,0]=0; beta=np.linalg.solve(xd.T@xd+pen,xd.T@y); out=[]
    for w in valid_ids:
        f=features(w); z=(f[cols].fillna(med).to_numpy(float)-mu)/sd; out.append(pd.DataFrame({'well_id':w,'model':f.last_tvt+np.c_[np.ones(len(z)),z]@beta,'target':f.last_tvt+f.target_delta}))
    return pd.concat(out,ignore_index=True)
''')
md('## Grouped OOF experiment')
code(r'''folds=np.arange(len(usable))%N_SPLITS; np.random.default_rng(SEED).shuffle(folds)
rows=[]
for fold in range(N_SPLITS):
    vi=[usable[i] for i in np.where(folds==fold)[0]]; tr=[w for w in usable if w not in vi]
    for w in vi:
        b=candidates(w); p,d=neighbor(w,tr); b['neighbor_copy']=p; b['neighbor_distance']=d; b['fold']=fold; rows.append(b)
    for name,cols in [('model_plain',BASE),('model_direction',DIR)]:
        q=model_predict(tr,vi,cols).rename(columns={'model':name})
        for w in vi:
            j=[i for i,b in enumerate(rows) if b.well_id.iloc[0]==w][-1]; rows[j]=rows[j].merge(q[q.well_id==w][['well_id','target',name]],on=['well_id','target'],how='left')
    print('fold',fold,'train',len(tr),'valid',len(vi))
OOF=pd.concat(rows,ignore_index=True)
OOF['neighbor_covered_150ft']=OOF.neighbor_distance<=150
OOF['blend_direction_neighbor']=.55*OOF.model_direction+.30*OOF.neighbor_copy+.15*OOF.last_known
OOF['blend_guarded']=np.where(OOF.neighbor_covered_150ft,.25*OOF.model_direction+.65*OOF.neighbor_copy+.10*OOF.last_known,.80*OOF.model_direction+.20*OOF.last_known)
VAR=['last_known','recent_slope','midpoint_gate','neighbor_copy','model_plain','model_direction','blend_direction_neighbor','blend_guarded']; out=[]; ws=[]
for name in VAR:
    per=OOF.groupby('well_id').apply(lambda x: rmse(x.target,x[name]),include_groups=False)
    out.append({'variant':name,'pooled_rmse':rmse(OOF.target,OOF[name]),'mean_well_rmse':float(per.mean()),'median_well_rmse':float(per.median()),'worst_decile_rmse':float(per.nlargest(max(1,len(per)//10)).mean()),'coverage':float(OOF[name].notna().mean())})
    ws += [{'well_id':w,'variant':name,'rmse':rmse(g.target,g[name]),'neighbor_distance':float(g.neighbor_distance.iloc[0])} for w,g in OOF.groupby('well_id')]
SCORES=pd.DataFrame(out).sort_values('pooled_rmse').reset_index(drop=True); WELL_SCORES=pd.DataFrame(ws)
display(SCORES)
display(WELL_SCORES.groupby('variant').rmse.agg(['mean','std','min','max']).sort_values('mean'))
''')
md('## Public-overlap audit and artifacts')
code(r'''overlap=[]
for w in TEST:
    tp=ROOT/'test'/f'{w}__horizontal_well.csv'; rp=ROOT/'train'/f'{w}__horizontal_well.csv'; overlap.append({'well_id':w,'train_copy_present':rp.exists(),'test_rows':len(pd.read_csv(tp)),'train_rows':len(pd.read_csv(rp)) if rp.exists() else 0})
display(pd.DataFrame(overlap))
best=SCORES.iloc[0]
summary={'best_variant':str(best.variant),'best_pooled_rmse':float(best.pooled_rmse),'scores':SCORES.to_dict('records'),'notes':['OOF is pseudo-test suffix validation, not Kaggle LB.','Neighbor references stay in the training fold.','Midpoint is a bounded proxy, not an oracle.','Exact formation-contact reconstruction and CNN/particle-filter are not implemented because their artifacts are not mounted.']}
work=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd(); SCORES.to_csv(work/'research_scores.csv',index=False); WELL_SCORES.to_csv(work/'research_well_scores.csv',index=False); (work/'research_summary.json').write_text(json.dumps(summary,indent=2)); print('BEST:',best.variant,best.pooled_rmse)
''')

nb=nbf.v4.new_notebook(); nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}; nb.cells=cells; out.write_text(nbf.writes(nb),encoding='utf-8'); print(out)
