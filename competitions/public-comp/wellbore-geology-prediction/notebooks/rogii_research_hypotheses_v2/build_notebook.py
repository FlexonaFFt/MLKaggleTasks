from pathlib import Path
import nbformat as nbf

out = Path(__file__).with_name('rogii_research_hypotheses_v2.ipynb')
cells=[]
def md(x): cells.append(nbf.v4.new_markdown_cell(x))
def code(x): cells.append(nbf.v4.new_code_cell(x))

md('''# ROGII research notebook - faithful alignment screen v2

## tl;dr

This kernel implements the core Discussion hypotheses instead of weak proxies: GR-to-typewell beam matching, two candidate trajectories, near-tie midpoint, and a geometry-aware neighbor transfer. It runs grouped pseudo-test OOF plus exact forensic scoring on the three public test wells that have train copies.

No `submission.csv` is created. Outputs: `research_v2_scores.csv`, `research_v2_public_scores.csv`, `research_v2_summary.json`.''')
md('''## Methods and assumptions

`TVT_input` is the visible prefix; missing rows are the suffix target. The beam matcher only uses observed GR, typewell GR, MD, and the visible TVT anchor. It keeps several TVT paths so repeated GR motifs can be handled as a near-tie rather than forcing one mode. Neighbor transfer is used only with a nearest heel under 150 ft and is calibrated on the visible prefix. The public forensic section uses train copies only for evaluation, never for OOF model selection.''')
code(r'''from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

def find_root():
    roots=[Path('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),Path('/kaggle/input/rogii-wellbore-geology-prediction'),Path.cwd()/'datasets',Path.cwd().parent/'datasets',Path.cwd().parents[1]/'datasets']
    return next((p for p in roots if (p/'train').exists() and (p/'test').exists()),None)
ROOT=find_root()
if ROOT is None: raise FileNotFoundError('dataset root not found')
SEED=42; N_SPLITS=5; MAX_SUFFIX=600
def ids(split): return sorted(p.name.split('__')[0] for p in (ROOT/split).glob('*__horizontal_well.csv'))
def load(wid,split='train'):
    b=ROOT/split; return pd.read_csv(b/f'{wid}__horizontal_well.csv'),pd.read_csv(b/f'{wid}__typewell.csv')
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))
def parts(hw):
    k=hw.TVT_input.notna(); return (hw.loc[k].copy(),hw.loc[~k].copy()) if k.sum()>20 and (~k).sum()>20 else None
TRAIN=ids('train'); TEST=ids('test'); W={w:load(w) for w in TRAIN}; USABLE=[w for w in TRAIN if parts(W[w][0])]
META={}
for w in USABLE:
    hw,_=W[w]; k,s=parts(hw); META[w]=(float(hw.X.iloc[0]),float(hw.Y.iloc[0]),float(k.TVT_input.iloc[-1]))
print('ROOT',ROOT,'train',len(TRAIN),'usable',len(USABLE),'test',len(TEST))
''')
md('## GR beam matcher')
code(r'''def sampled_suffix(s,max_rows=MAX_SUFFIX):
    ix=np.linspace(0,len(s)-1,min(len(s),max_rows),dtype=int); return s.iloc[ix].copy()
def beam_paths(hw,tw,max_rows=MAX_SUFFIX,beam_width=24,move_radius=3):
    k,s=parts(hw); s=sampled_suffix(s,max_rows); g=tw[['TVT','GR']].dropna().sort_values('TVT'); tvt=g.TVT.to_numpy(float); grt=g.GR.to_numpy(float)
    q=s.GR.interpolate(limit_direction='both').fillna(float(np.nanmean(grt))).to_numpy(float); start=int(np.argmin(abs(tvt-float(k.TVT_input.iloc[-1])))); beams=[(0.,start,(start,))]
    for qg in q:
        cand=[]
        for cost,idx,path in beams:
            for j in range(max(0,idx-move_radius),min(len(tvt),idx+move_radius+1)):
                move=j-idx; c=cost+(qg-grt[j])**2/80.+2.5*abs(move)
                cand.append((c,j,path+(j,)))
        cand.sort(key=lambda z:z[0]); beams=cand[:beam_width]
    best=beams[:2]; return s,best,tvt
def beam_candidates(hw,tw,max_rows=MAX_SUFFIX):
    s,beams,tvt=beam_paths(hw,tw,max_rows=max_rows)
    out=pd.DataFrame({'row_index':s.index,'target':s['TVT'].to_numpy(float) if 'TVT' in s else np.nan})
    paths=[np.array([tvt[i] for i in b[2][1:]],float) for b in beams]
    out['beam_1']=paths[0]; out['beam_2']=paths[1] if len(paths)>1 else paths[0]
    gap=(beams[1][0]-beams[0][0])/max(len(s),1) if len(beams)>1 else np.inf
    out['beam_midpoint']=np.where(gap<=12.,(out.beam_1+out.beam_2)/2.,out.beam_1)
    out['beam_cost_gap']=gap
    return out
''')
md('## Geometry-aware neighbor transfer')
code(r'''def nearest(w,refs):
    q=np.array(META[w][:2]); a=np.array([META[r][:2] for r in refs]); d=np.sqrt(((a-q)**2).sum(1)); j=int(d.argmin()); return refs[j],float(d[j])
def aligned_neighbor(w,refs,max_rows=MAX_SUFFIX):
    hw,_=W[w]; k,s=parts(hw); s=sampled_suffix(s,max_rows); r,dist=nearest(w,refs); rhw,_=W[r]; rk,rs=parts(rhw)
    qmd=s.MD.to_numpy(float)-float(k.MD.iloc[-1]); rmd=rhw.MD.to_numpy(float)-float(rk.MD.iloc[-1]); rtv=rhw.TVT.to_numpy(float); base=np.interp(qmd,rmd,rtv,left=rtv[0],right=rtv[-1])
    visible_md=k.MD.to_numpy(float)-float(k.MD.iloc[-1]); visible_ref=np.interp(visible_md,rmd,rtv,left=rtv[0],right=rtv[-1]); delta=k.TVT_input.to_numpy(float)-visible_ref
    tail=delta[-min(300,len(delta)):]; x=visible_md[-len(tail):]; slope=np.polyfit(x-x[-1],tail,1)[0] if len(tail)>3 else 0.; offset=float(np.median(tail)); pred=base+offset+slope*(qmd-visible_md[-1])
    return pd.DataFrame({'row_index':s.index,'neighbor':pred,'neighbor_distance':dist})
''')
md('## Grouped OOF and public forensic evaluation')
code(r'''fold=np.arange(len(USABLE))%N_SPLITS; np.random.default_rng(SEED).shuffle(fold); rows=[]
for f in range(N_SPLITS):
    valid=[USABLE[i] for i in np.where(fold==f)[0]]; refs=[w for w in USABLE if w not in valid]
    for w in valid:
        hw,tw=W[w]; b=beam_candidates(hw,tw); n=aligned_neighbor(w,refs); x=b.merge(n,on='row_index',how='left'); x['well_id']=w; x['fold']=f; x['last_known']=float(parts(hw)[0].TVT_input.iloc[-1]); x['neighbor_guarded']=np.where(x.neighbor_distance<=150,.65*x.neighbor+.35*x.beam_midpoint,x.beam_midpoint); rows.append(x)
    print('fold',f,'valid',len(valid))
OOF=pd.concat(rows,ignore_index=True)
VAR=['beam_1','beam_2','beam_midpoint','neighbor_guarded','last_known']; scores=[]
for v in VAR:
    per=OOF.groupby('well_id').apply(lambda z:rmse(z.target,z[v]),include_groups=False); scores.append({'variant':v,'pooled_rmse':rmse(OOF.target,OOF[v]),'mean_well_rmse':float(per.mean()),'median_well_rmse':float(per.median()),'worst_decile_rmse':float(per.nlargest(max(1,len(per)//10)).mean()),'near_tie_rate':float((OOF.beam_cost_gap<=12).mean())})
SCORES=pd.DataFrame(scores).sort_values('pooled_rmse'); display(SCORES)

public=[]
for w in TEST:
    thw,_=load(w,'test'); rhw,_=load(w,'train'); truth=rhw.TVT.to_numpy(float); mask=thw.TVT_input.isna().to_numpy(); b=beam_candidates(thw,load(w,'test')[1],max_rows=10**9); pred=b.set_index('row_index'); idx=np.flatnonzero(mask); vals={'well_id':w,'rows':len(idx)}
    for v in ['beam_1','beam_2','beam_midpoint']:
        vals[v+'_rmse']=rmse(truth[idx],pred.loc[idx,v].to_numpy(float))
    public.append(vals)
PUBLIC=pd.DataFrame(public); display(PUBLIC)
''')
md('## Artifacts and decision notes')
code(r'''best=SCORES.iloc[0]; summary={'best_variant':str(best.variant),'best_oof_rmse':float(best.pooled_rmse),'oof_scores':SCORES.to_dict('records'),'public_exact_scores':PUBLIC.to_dict('records'),'notes':['This is a research notebook, not a submission.','Beam paths use only GR/typewell/visible anchor.','Midpoint is triggered by beam cost near-tie.','Neighbor transfer is calibrated on visible prefix and gated at 150 ft.','Formation-contact artifact selector, CNN, and particle filter are intentionally separate follow-up work.']}
work=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd(); SCORES.to_csv(work/'research_v2_scores.csv',index=False); PUBLIC.to_csv(work/'research_v2_public_scores.csv',index=False); (work/'research_v2_summary.json').write_text(json.dumps(summary,indent=2)); print('BEST',best.variant,best.pooled_rmse)
''')
nb=nbf.v4.new_notebook(); nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}; nb.cells=cells; out.write_text(nbf.writes(nb),encoding='utf-8'); print(out)
