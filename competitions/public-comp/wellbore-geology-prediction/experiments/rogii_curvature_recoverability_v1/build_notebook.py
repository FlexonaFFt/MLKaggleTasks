from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).with_name("rogii_curvature_recoverability_v1.ipynb")
cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
code = lambda text: cells.append(nbf.v4.new_code_cell(text))

md("""# ROGII curvature recoverability lab v1

## tl;dr

Can legal well-level information recover missing suffix curvature? We keep the geometric Ridge path and predict five anchored smooth residual components from geometry, GR confidence and fold-safe neighbor coefficients.

No `submission.csv` is created. Promote only at **RMSE <= 13.0** with wins over spatial Ridge in at least **4/5 spatial folds**.
""")

md("""## Context & Methods

`TVT_base = TVT_PS - (Z - Z_PS)`

`TVT_ridge = TVT_base + q1*s + q2*s²`

`TVT_curvature = TVT_ridge + sum(a_k*sin(k*pi*s), k=1..5)`

### Key Assumptions

- Original `TVT_input` masks are preserved.
- Outer folds are spatial KMeans clusters at projection start.
- Validation TVT appears only in scores and oracle diagnostics.
- Neighbor labels for validation come only from outer-training wells.
- Horizontal GR, trajectory, prefix TVT and supplied typewells are legal inputs.
""")

code(r'''from pathlib import Path
import json, os, warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
VERSION, SEED, N_SPLITS, N_SINE = "v1", 42, 5, 5
SMOKE = os.environ.get("ROGII_SMOKE", "0") == "1"
H_SUFFIX, T_SUFFIX = "__horizontal_well.csv", "__typewell.csv"
LEGAL = ["MD", "X", "Y", "Z", "GR", "TVT_input"]

def find_root():
    roots = [Path(os.environ["ROGII_DATA"]) if os.environ.get("ROGII_DATA") else None,
             Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
             Path("/kaggle/input/rogii-wellbore-geology-prediction"), Path.cwd()/"datasets", Path.cwd().parent/"datasets"]
    roots += [p/"datasets" for p in list(Path.cwd().parents)[:4]]
    for root in roots:
        if root is not None and (root/"train").exists() and (root/"test").exists(): return root
    raise FileNotFoundError("ROGII dataset root not found")

ROOT, WORK = find_root(), Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
TRAIN = ROOT/"train"
ids = sorted(p.name.removesuffix(H_SUFFIX) for p in TRAIN.glob(f"*{H_SUFFIX}"))
if SMOKE: ids, N_SPLITS = ids[:30], 3
assert len(ids) >= 2*N_SPLITS

def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def shape_basis(s):
    s = np.asarray(s, float)
    return np.column_stack([s] + [np.sin(k*np.pi*s) for k in range(1, N_SINE+1)])
def q_basis(s):
    s = np.asarray(s, float)
    return np.column_stack([s, s*s])
print({"root": str(ROOT), "wells": len(ids), "folds": N_SPLITS, "smoke": SMOKE})
''')

md("## Data - legal features and coefficient targets")

code(r'''def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 and np.ptp(x[ok]) > 1e-9 else 0.0

def fill_gr(series):
    series = series.astype(float).interpolate(limit_direction="both")
    return series.fillna(float(series.median()) if series.notna().any() else 0.0).to_numpy(float)

def gr_landscape(well_id, query_gr, base_tvt):
    path = TRAIN/f"{well_id}{T_SUFFIX}"
    empty = {"gr_shift": 0., "gr_corr": 0., "gr_gap": 0., "gr_entropy": 1., "gr_valid": 0.}
    if not path.exists(): return empty
    tw = pd.read_csv(path, usecols=["TVT", "GR"]).dropna().sort_values("TVT").drop_duplicates("TVT")
    if len(tw) < 20: return empty
    take = np.linspace(0, len(base_tvt)-1, min(300, len(base_tvt))).astype(int)
    qgr, qt = query_gr[take], base_tvt[take]
    tt, tg, shifts = tw.TVT.to_numpy(float), tw.GR.to_numpy(float), np.arange(-120., 121., 10.)
    corr = []
    for shift in shifts:
        ok = (qt+shift >= tt[0]) & (qt+shift <= tt[-1])
        ref = np.interp(qt[ok]+shift, tt, tg) if ok.sum() >= 20 else np.array([])
        corr.append(float(np.corrcoef(qgr[ok], ref)[0,1]) if len(ref) and np.std(ref)>1e-9 and np.std(qgr[ok])>1e-9 else -1.)
    corr = np.nan_to_num(corr, nan=-1.); order = np.argsort(corr)[::-1]
    prob = np.exp((corr-corr.max())/.1); prob /= prob.sum()
    return {"gr_shift": float(shifts[order[0]]), "gr_corr": float(corr[order[0]]),
            "gr_gap": float(corr[order[0]]-corr[order[1]]),
            "gr_entropy": float(-np.sum(prob*np.log(prob+1e-12))/np.log(len(prob))),
            "gr_valid": float((corr>-1).mean())}

CURVES, records = {}, []
for number, well_id in enumerate(ids, 1):
    frame = pd.read_csv(TRAIN/f"{well_id}{H_SUFFIX}", usecols=LEGAL+["TVT"])
    known = frame.TVT_input.notna().to_numpy(); ki, ti = np.flatnonzero(known), np.flatnonzero(~known)
    if len(ki)<20 or len(ti)<20: continue
    ps = ki[-1]; mdv = frame.MD.to_numpy(float)
    xc, yc, z = (frame[c].to_numpy(float) for c in ["X","Y","Z"])
    tvti, truth = frame.TVT_input.to_numpy(float), frame.TVT.to_numpy(float)[ti]
    if not np.isfinite(truth).all(): continue
    span = max(float(mdv[ti[-1]]-mdv[ps]), 1.); s = (mdv[ti]-mdv[ps])/span
    base = float(tvti[ps])-(z[ti]-z[ps]); correction = truth-base
    shape = np.linalg.lstsq(shape_basis(s), correction, rcond=None)[0]
    quad = np.linalg.lstsq(q_basis(s), correction, rcond=None)[0]
    gr = fill_gr(frame.GR); u = tvti[ki]+z[ki]
    dx, dy = float(xc[ti[-1]]-xc[ps]), float(yc[ti[-1]]-yc[ps]); az = float(np.arctan2(dy,dx))
    grid = np.linspace(0,1,25)
    path_xy = np.column_stack([np.interp(grid,np.r_[0.,s],np.r_[xc[ps],xc[ti]]),
                               np.interp(grid,np.r_[0.,s],np.r_[yc[ps],yc[ti]])])
    rec = {"well_id":well_id,"ps_x":float(xc[ps]),"ps_y":float(yc[ps]),"known_rows":len(ki),
           "suffix_rows":len(ti),"suffix_span":span,"prediction_share":len(ti)/len(frame),
           "tvt_ps":float(tvti[ps]),"z_ps":float(z[ps]),"future_dx":dx,"future_dy":dy,
           "future_dxy":float(np.hypot(dx,dy)),"future_dz":float(z[ti[-1]]-z[ps]),
           "azimuth_sin":float(np.sin(az)),"azimuth_cos":float(np.cos(az)),
           "prefix_u_range":float(np.ptp(u)),"prefix_u_std":float(np.std(u)),
           "prefix_u_slope":slope(mdv[ki],u)*span,"prefix_gr_mean":float(np.mean(gr[ki])),
           "prefix_gr_std":float(np.std(gr[ki])),"suffix_gr_mean":float(np.mean(gr[ti])),
           "suffix_gr_std":float(np.std(gr[ti])),"gr_mean_change":float(np.mean(gr[ti])-np.mean(gr[ki])),
           "gr_missing_prefix":float(frame.GR.iloc[ki].isna().mean()),
           "gr_missing_suffix":float(frame.GR.iloc[ti].isna().mean()),
           **gr_landscape(well_id,gr[ti],base),
           **{f"q{i}":float(v) for i,v in enumerate(quad)},
           **{f"shape{i}":float(v) for i,v in enumerate(shape)}}
    records.append(rec); CURVES[well_id]={"s":s,"base":base,"truth":truth,"path_xy":path_xy,"azimuth":az}
    if number%100==0: print("described",number,"/",len(ids))

WELLS = pd.DataFrame(records).set_index("well_id",drop=False)
QCOLS=["q0","q1"]; SCOLS=[f"shape{i}" for i in range(N_SINE+1)]
FEATURES=[c for c in WELLS if c not in {"well_id","spatial_fold"} and not c.startswith(("q","shape"))]
xy=StandardScaler().fit_transform(WELLS[["ps_x","ps_y"]])
WELLS["spatial_fold"]=KMeans(n_clusters=N_SPLITS,random_state=SEED,n_init=10).fit_predict(xy)
assert np.isfinite(WELLS[FEATURES+QCOLS+SCOLS].to_numpy(float)).all()
print("usable",len(WELLS),"features",len(FEATURES),"folds",WELLS.spatial_fold.value_counts().sort_index().to_dict())
''')

md("## Methods - fold-safe residual and neighbor models")

code(r'''grid=np.linspace(0,1,401)
Q_TO_SHAPE=np.linalg.pinv(shape_basis(grid))@q_basis(grid)

def path_distance(left,right):
    a,b=CURVES[left],CURVES[right]
    distance=np.median(np.linalg.norm(a["path_xy"]-b["path_xy"],axis=1))
    angle=np.arccos(np.clip(np.cos(a["azimuth"]-b["azimuth"]),-1.,1.))
    return float(distance*(1+angle/np.pi)),float(angle)

def neighbor_features(query_ids,reference_ids):
    rows=[]
    for query in query_ids:
        nearest=sorted((path_distance(query,ref)[0],ref) for ref in reference_ids if ref!=query)[:3]
        distances=np.array([x[0] for x in nearest]); weights=1/np.maximum(distances,25.); weights/=weights.sum()
        values=WELLS.loc[[x[1] for x in nearest],SCOLS].to_numpy(float)
        rows.append([distances[0],path_distance(query,nearest[0][1])[1],*(weights@values)])
    return pd.DataFrame(rows,index=query_ids,columns=["neighbor_distance","neighbor_angle",*[f"neighbor_{c}" for c in SCOLS]])

def forest():
    return ExtraTreesRegressor(n_estimators=350,min_samples_leaf=8,max_features=.8,random_state=SEED,n_jobs=-1)

assert np.allclose(shape_basis([0.]),0.) and Q_TO_SHAPE.shape==(N_SINE+1,2)
''')

md("## Results - honest spatial OOF")

code(r'''frames=[]
for fold in sorted(WELLS.spatial_fold.unique()):
    train_ids=WELLS.index[WELLS.spatial_fold!=fold].tolist(); valid_ids=WELLS.index[WELLS.spatial_fold==fold].tolist()
    ridge=make_pipeline(StandardScaler(),Ridge(alpha=30.)).fit(WELLS.loc[train_ids,FEATURES],WELLS.loc[train_ids,QCOLS])
    train_q=ridge.predict(WELLS.loc[train_ids,FEATURES]); valid_q=ridge.predict(WELLS.loc[valid_ids,FEATURES])
    train_shape=train_q@Q_TO_SHAPE.T; valid_shape=valid_q@Q_TO_SHAPE.T
    residual=WELLS.loc[train_ids,SCOLS].to_numpy(float)-train_shape
    geometry=forest().fit(WELLS.loc[train_ids,FEATURES],residual)
    geometry_shape=valid_shape+geometry.predict(WELLS.loc[valid_ids,FEATURES])
    train_nf=neighbor_features(train_ids,train_ids); valid_nf=neighbor_features(valid_ids,train_ids)
    neighbor=forest().fit(np.column_stack([WELLS.loc[train_ids,FEATURES],train_nf]),residual)
    neighbor_shape=valid_shape+neighbor.predict(np.column_stack([WELLS.loc[valid_ids,FEATURES],valid_nf]))
    for row,well_id in enumerate(valid_ids):
        curve=CURVES[well_id]; s,base,truth=curve["s"],curve["base"],curve["truth"]
        frames.append(pd.DataFrame({"well_id":well_id,"spatial_fold":fold,"target":truth,"z_anchor":base,
          "spatial_ridge":base+q_basis(s)@valid_q[row],
          "curvature_geometry":base+shape_basis(s)@geometry_shape[row],
          "curvature_neighbor":base+shape_basis(s)@neighbor_shape[row],
          "oracle_6basis":base+shape_basis(s)@WELLS.loc[well_id,SCOLS].to_numpy(float)}))
    print("fold",fold,"train",len(train_ids),"valid",len(valid_ids))

OOF=pd.concat(frames,ignore_index=True)
CANDIDATES=["z_anchor","spatial_ridge","curvature_geometry","curvature_neighbor","oracle_6basis"]
SCORES=pd.DataFrame([{"candidate":c,"pooled_rmse":rmse(OOF.target,OOF[c])} for c in CANDIDATES]).sort_values("pooled_rmse")
SPATIAL=pd.DataFrame([{"spatial_fold":f,"candidate":c,"rows":len(g),"pooled_rmse":rmse(g.target,g[c])}
                      for f,g in OOF.groupby("spatial_fold") for c in CANDIDATES])
pivot=SPATIAL.pivot(index="spatial_fold",columns="candidate",values="pooled_rmse")
ridge_score=float(SCORES.set_index("candidate").loc["spatial_ridge","pooled_rmse"])
candidate_score=float(SCORES.set_index("candidate").loc["curvature_neighbor","pooled_rmse"])
fold_wins=int((pivot.curvature_neighbor<pivot.spatial_ridge).sum())
viable=bool(candidate_score<=13.0 and fold_wins>=min(4,N_SPLITS))
display(SCORES); display(pivot)
print({"improvement_ft":ridge_score-candidate_score,"fold_wins":fold_wins,"viable":viable})
''')

md("## Takeaways and artifacts")

code(r'''SCORES.to_csv(WORK/"curvature_scores_v1.csv",index=False)
SPATIAL.to_csv(WORK/"curvature_spatial_scores_v1.csv",index=False)
WELLS.reset_index(drop=True).to_csv(WORK/"curvature_wells_v1.csv",index=False)
OOF.to_parquet(WORK/"curvature_oof_predictions_v1.parquet",index=False)
summary={"version":VERSION,"wells":int(len(WELLS)),"rows":int(len(OOF)),"scores":SCORES.to_dict("records"),
 "spatial_scores":SPATIAL.to_dict("records"),"ridge_rmse":ridge_score,"curvature_neighbor_rmse":candidate_score,
 "improvement_ft":ridge_score-candidate_score,"fold_wins":fold_wins,
 "viability_rule":"curvature_neighbor <= 13.0 and beats spatial_ridge in >=4/5 folds","viable":viable,
 "decision":"continue_curvature_recovery" if viable else "stop_or_revise_curvature_recovery",
 "caveats":["Spatial clusters are harder than prior random-well folds.","Oracle uses validation TVT.","No submission is created."]}
(WORK/"curvature_summary_v1.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
''')

notebook=nbf.v4.new_notebook(metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}})
notebook.cells=cells
OUT.write_text(nbf.writes(notebook),encoding="utf-8")
print(OUT)
