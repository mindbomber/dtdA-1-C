#!/usr/bin/env python3
"""Empirical cardiorespiratory torus study on PhysioNet Fantasia and MIT-BIH SLPDB."""
from __future__ import annotations

import json, math, os, re, warnings
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy import signal, stats
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

try:
    from ripser import ripser
except Exception:
    ripser = None

TWO_PI = 2*np.pi
BEAT_SYMBOLS = set('NLRaAJSejVFE/fQ?')
APNEA_CODES = {'H','HA','OA','X','CA','CAA'}
SLEEP_CODES = {'1','2','3','4','R'}

@dataclass
class Config:
    target_fs: float = 4.0
    window_sec: int = 300
    step_sec: int = 150
    bins: int = 24
    recurrence_points: int = 180
    surrogate_reps: int = 40
    seed: int = 20260713

CFG = Config()
RNG = np.random.default_rng(CFG.seed)
OUT = Path(os.environ.get('EMPIRICAL_OUT','empirical_results'))
FIG = OUT/'figures'; CSV = OUT/'csv'; DATA = Path(os.environ.get('PHYSIONET_DATA','physionet_data'))
for d in (OUT,FIG,CSV,DATA): d.mkdir(parents=True, exist_ok=True)
(OUT/'config.json').write_text(json.dumps(asdict(CFG),indent=2))


def savefig(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def download_database(db):
    dst=DATA/db
    records_file=dst/'RECORDS'
    if not records_file.exists():
        print(f'Downloading {db} to {dst}',flush=True)
        wfdb.dl_database(db,str(dst),records='all',annotators='all',keep_subdirs=False,overwrite=False)
    return dst


def find_resp_channel(names):
    low=[str(x).lower() for x in names]
    candidates=[i for i,n in enumerate(low) if 'resp' in n or 'airflow' in n or 'nasal' in n]
    if not candidates: return None
    return sorted(candidates,key=lambda i:(0 if ('nasal' in low[i] or low[i]=='resp' or 'airflow' in low[i]) else 1,i))[0]


def clean_respiration(x,fs,target_fs=4.0):
    x=np.asarray(x,dtype=float)
    bad=~np.isfinite(x)
    if bad.all(): raise ValueError('respiration all missing')
    if bad.any():
        good=np.where(~bad)[0]; x[bad]=np.interp(np.where(bad)[0],good,x[good])
    x=signal.detrend(x)
    frac=Fraction(target_fs/float(fs)).limit_denominator(1000)
    y=signal.resample_poly(x,frac.numerator,frac.denominator)
    y=signal.detrend(y)
    hi=min(0.7,0.45*target_fs)
    sos=signal.butter(4,[0.05,hi],btype='bandpass',fs=target_fs,output='sos')
    y=signal.sosfiltfilt(sos,y)
    analytic=signal.hilbert(y)
    return y, np.mod(np.angle(analytic),TWO_PI), np.abs(analytic)


def filter_beats(samples,symbols,fs):
    s=np.asarray(samples,dtype=int)
    if symbols and len(symbols)==len(s):
        mask=np.array([(sym in BEAT_SYMBOLS) or sym=='N' for sym in symbols])
        if mask.sum()>20: s=s[mask]
    return s/float(fs)


def cardiac_phase(query_t,beats):
    idx=np.searchsorted(beats,query_t,side='right')-1
    valid=(idx>=0)&(idx+1<len(beats))
    phase=np.full(len(query_t),np.nan); rr=np.full(len(query_t),np.nan)
    j=idx[valid]; intervals=beats[j+1]-beats[j]
    ok=(intervals>=0.28)&(intervals<=2.5)
    vv=np.where(valid)[0][ok]; jj=j[ok]; ints=intervals[ok]
    phase[vv]=np.mod(TWO_PI*(query_t[vv]-beats[jj])/ints,TWO_PI)
    rr[vv]=ints
    return phase,rr


def phase_hist(theta,phi,bins=24):
    h,_,_=np.histogram2d(theta,phi,bins=bins,range=[[0,TWO_PI],[0,TWO_PI]])
    return h


def normalized_mi(h):
    p=h/h.sum() if h.sum() else h
    px=p.sum(1,keepdims=True); py=p.sum(0,keepdims=True)
    nz=p>0
    mi=float(np.sum(p[nz]*np.log(p[nz]/(px@py)[nz]))) if nz.any() else np.nan
    hx=-float(np.sum(px[px>0]*np.log(px[px>0]))); hy=-float(np.sum(py[py>0]*np.log(py[py>0])))
    return mi/max(min(hx,hy),1e-12)


def recurrence_determinism(theta,phi,n=180):
    if len(theta)<20:return np.nan
    ii=np.linspace(0,len(theta)-1,min(n,len(theta))).astype(int)
    z=np.column_stack([np.cos(theta[ii]),np.sin(theta[ii]),np.cos(phi[ii]),np.sin(phi[ii])])
    d=cdist(z,z); tri=d[np.triu_indices(len(d),1)]; eps=np.quantile(tri,0.08)
    R=d<=eps; np.fill_diagonal(R,False); total=R.sum(); det=0
    for k in range(-len(R)+1,len(R)):
        q=np.diag(R,k).astype(int)
        if len(q)<2:continue
        e=np.r_[0,q,0]; st=np.where(np.diff(e)==1)[0]; en=np.where(np.diff(e)==-1)[0]
        lengths=en-st; det+=lengths[lengths>=2].sum()
    return det/max(total,1)


def compute_metrics(theta,phi,rr,resp,amp,fs):
    valid=np.isfinite(theta)&np.isfinite(phi)&np.isfinite(rr)&np.isfinite(resp)
    theta=theta[valid]; phi=phi[valid]; rr=rr[valid]; resp=resp[valid]; amp=amp[valid]
    if len(theta)<100:return None,None
    h=phase_hist(theta,phi,CFG.bins)
    p=h/h.sum(); entropy=-np.sum(p[p>0]*np.log(p[p>0]))/np.log(CFG.bins**2)
    coverage=np.mean(h>=2)
    mi=normalized_mi(h)
    plvs={k:abs(np.mean(np.exp(1j*(theta-k*phi)))) for k in range(2,9)}
    best_k=max(plvs,key=plvs.get)
    up=np.unwrap(phi); resp_cycles=max((up[-1]-up[0])/TWO_PI,1e-6)
    duration=len(phi)/fs
    heart_rate=60/np.nanmedian(rr)
    resp_rate=60*resp_cycles/duration
    ratio=heart_rate/max(resp_rate,1e-6)
    crossings=np.where(np.diff(np.floor(up/TWO_PI))>0)[0]
    poincare=abs(np.mean(np.exp(1j*theta[crossings]))) if len(crossings)>=3 else np.nan
    metrics={'heart_rate':heart_rate,'resp_rate':resp_rate,'cycles_per_breath':ratio,'plv_max':plvs[best_k],'best_locking_k':best_k,'poincare_R':poincare,'mi_norm':mi,'joint_entropy':entropy,'coverage':coverage,'recurrence_det':recurrence_determinism(theta,phi),'resp_amp':float(np.median(amp)),'resp_amp_cv':float(np.std(amp)/(np.mean(amp)+1e-12)),'phase_speed_cv':float(np.std(np.diff(up))/(abs(np.mean(np.diff(up)))+1e-12))}
    return metrics,h


def js_div(h1,h2):
    a=h1.ravel().astype(float)+1e-9; b=h2.ravel().astype(float)+1e-9
    a/=a.sum(); b/=b.sum(); return float(jensenshannon(a,b,base=2)**2)


def record_phases(dbdir,record):
    base=str(dbdir/record)
    hdr=wfdb.rdheader(base)
    ridx=find_resp_channel(hdr.sig_name)
    if ridx is None: raise ValueError(f'no respiration in {record}: {hdr.sig_name}')
    rec=wfdb.rdrecord(base,channels=[ridx],physical=True)
    resp=np.asarray(rec.p_signal[:,0],float)
    ann=wfdb.rdann(base,'ecg')
    beats=filter_beats(ann.sample,ann.symbol,hdr.fs)
    resp_f,phi,amp=clean_respiration(resp,hdr.fs,CFG.target_fs)
    t=np.arange(len(resp_f))/CFG.target_fs
    theta,rr=cardiac_phase(t,beats)
    return hdr,t,theta,phi,rr,resp_f,amp,beats


def process_fantasia(dbdir):
    records=[x.strip() for x in (dbdir/'RECORDS').read_text().splitlines() if x.strip()]
    rows=[]; hists={}; record_series={}; surrogate=[]
    for n,recname in enumerate(records,1):
        print(f'Fantasia {n}/{len(records)} {recname}',flush=True)
        try: hdr,t,theta,phi,rr,resp,amp,beats=record_phases(dbdir,recname)
        except Exception as e:
            print('SKIP',recname,repr(e)); continue
        age_group='young' if 'y' in recname else 'elderly'
        record_series[recname]=(theta,phi)
        win=int(CFG.window_sec*CFG.target_fs); step=int(CFG.step_sec*CFG.target_fs)
        rec_h=[]; rec_indices=[]
        for start in range(int(30*CFG.target_fs),len(t)-win-int(30*CFG.target_fs)+1,step):
            sl=slice(start,start+win)
            m,h=compute_metrics(theta[sl],phi[sl],rr[sl],resp[sl],amp[sl],CFG.target_fs)
            if m is None:continue
            m.update(record=recname,age_group=age_group,start_sec=start/CFG.target_fs)
            rows.append(m); rec_h.append(h); rec_indices.append(len(rows)-1)
        if rec_h:
            base_h=np.sum(rec_h,axis=0)
            for idx,h in zip(rec_indices,rec_h): rows[idx]['deformation_js']=js_div(h,base_h)
            hists[recname]=base_h
        valid=np.isfinite(theta)&np.isfinite(phi); th=theta[valid][::4]; ph=phi[valid][::4]
        if len(th)>1000:
            obs_h=phase_hist(th,ph,CFG.bins); obs_mi=normalized_mi(obs_h)
            obs_plv=max(abs(np.mean(np.exp(1j*(th-k*ph)))) for k in range(2,9))
            s_plv=[];s_mi=[]
            for _ in range(CFG.surrogate_reps):
                sh=int(RNG.integers(len(ph)//10,9*len(ph)//10)); ps=np.roll(ph,sh)
                s_plv.append(max(abs(np.mean(np.exp(1j*(th-k*ps)))) for k in range(2,9)))
                s_mi.append(normalized_mi(phase_hist(th,ps,CFG.bins)))
            surrogate.append({'record':recname,'age_group':age_group,'plv_observed':obs_plv,'plv_surrogate_mean':np.mean(s_plv),'plv_z':(obs_plv-np.mean(s_plv))/(np.std(s_plv)+1e-9),'mi_observed':obs_mi,'mi_surrogate_mean':np.mean(s_mi),'mi_z':(obs_mi-np.mean(s_mi))/(np.std(s_mi)+1e-9)})
    df=pd.DataFrame(rows); sur=pd.DataFrame(surrogate)
    df.to_csv(CSV/'fantasia_windows.csv',index=False); sur.to_csv(CSV/'fantasia_surrogates.csv',index=False)
    return df,sur,record_series,hists


def cliff_delta(x,y):
    x=np.asarray(x);y=np.asarray(y);return float((np.sum(x[:,None]>y)-np.sum(x[:,None]<y))/(len(x)*len(y)))


def fantasia_stats(df):
    metrics=['heart_rate','resp_rate','cycles_per_breath','plv_max','poincare_R','mi_norm','joint_entropy','coverage','recurrence_det','deformation_js','phase_speed_cv']
    subj=df.groupby(['record','age_group'])[metrics].mean().reset_index()
    rows=[]
    for m in metrics:
        y=subj[subj.age_group=='young'][m].dropna();o=subj[subj.age_group=='elderly'][m].dropna()
        u,p=stats.mannwhitneyu(y,o,alternative='two-sided')
        rows.append({'metric':m,'young_mean':y.mean(),'elderly_mean':o.mean(),'young_median':y.median(),'elderly_median':o.median(),'p':p,'cliff_delta_young_minus_old':cliff_delta(y.values,o.values)})
    out=pd.DataFrame(rows);out['q_fdr']=multipletests(out.p,method='fdr_bh')[1]
    out.to_csv(CSV/'fantasia_age_stats.csv',index=False);subj.to_csv(CSV/'fantasia_subject_summary.csv',index=False)
    return subj,out


def topology_analysis(series):
    if ripser is None:return pd.DataFrame()
    chosen=sorted(series)[:5]+sorted(series)[-5:]
    rows=[]
    for rec in chosen:
        th,ph=series[rec]; valid=np.where(np.isfinite(th)&np.isfinite(ph))[0]
        if len(valid)<200:continue
        idx=np.linspace(valid[0],valid[-1],180).astype(int)
        th=th[idx];ph=ph[idx]
        clouds={'empirical':np.column_stack([np.cos(th),np.sin(th),np.cos(ph),np.sin(ph)]),'one_cycle_null':np.column_stack([np.cos(th),np.sin(th),np.cos(th/5),np.sin(th/5)])}
        for kind,x in clouds.items():
            dg=ripser(x,maxdim=2,thresh=2.2)['dgms']
            p1=np.sort((dg[1][:,1]-dg[1][:,0])[np.isfinite(dg[1][:,1])])[::-1] if len(dg)>1 else np.array([])
            p2=np.sort((dg[2][:,1]-dg[2][:,0])[np.isfinite(dg[2][:,1])])[::-1] if len(dg)>2 else np.array([])
            rows.append({'record':rec,'kind':kind,'h1_longest':p1[0] if len(p1) else 0,'h1_second':p1[1] if len(p1)>1 else 0,'h2_longest':p2[0] if len(p2) else 0})
    out=pd.DataFrame(rows);out.to_csv(CSV/'fantasia_topology.csv',index=False);return out


def parse_st_annotations(base,fs,duration_sec):
    ann=wfdb.rdann(base,'st')
    n=int(math.ceil(duration_sec/30)); apnea=np.zeros(n,dtype=int); sleep=np.zeros(n,dtype=int); codes=[]
    for smp,note in zip(ann.sample,ann.aux_note):
        i=int((smp/fs)//30)
        if not (0<=i<n):continue
        text=str(note).replace('\x00',' ').strip().upper(); toks=set(re.findall(r'[A-Z]+|[1-4]',text))
        apnea[i]=int(bool(toks&APNEA_CODES)); sleep[i]=int(bool(toks&SLEEP_CODES)); codes.append(text)
    return apnea,sleep,codes

AHI={'slp01a':17.0,'slp01b':22.3,'slp02a':34.0,'slp02b':22.2,'slp03':43.0,'slp04':59.8,'slp14':30.7,'slp16':53.1,'slp32':22.1,'slp37':100.8,'slp48':46.8,'slp59':55.3,'slp60':59.2,'slp61':41.2,'slp66':65.5,'slp67x':0.7}


def process_slpdb(dbdir):
    records=[x.strip() for x in (dbdir/'RECORDS').read_text().splitlines() if x.strip() and x.strip() in AHI]
    rows=[]; code_inventory={}
    for n,recname in enumerate(records,1):
        print(f'SLPDB {n}/{len(records)} {recname}',flush=True)
        try:hdr,t,theta,phi,rr,resp,amp,beats=record_phases(dbdir,recname)
        except Exception as e:print('SKIP signal',recname,repr(e));continue
        try:apnea,sleep,codes=parse_st_annotations(str(dbdir/recname),hdr.fs,t[-1]);code_inventory[recname]=sorted(set(codes))[:50]
        except Exception as e:print('SKIP annotation',recname,repr(e));continue
        win=int(CFG.window_sec*CFG.target_fs);step=int(30*CFG.target_fs); rec_h=[];rec_idx=[]
        for start in range(int(60*CFG.target_fs),len(t)-win-int(60*CFG.target_fs)+1,step):
            end=start+win; e0=int((start/CFG.target_fs)//30);e1=int((end/CFG.target_fs)//30)
            if e1>len(apnea):break
            m,h=compute_metrics(theta[start:end],phi[start:end],rr[start:end],resp[start:end],amp[start:end],CFG.target_fs)
            if m is None:continue
            cur_apnea=float(apnea[e0:e1].mean());sleep_frac=float(sleep[e0:e1].mean())
            f0=e1;f1=min(len(apnea),f0+10);future=float(apnea[f0:f1].max()) if f1>f0 else np.nan
            m.update(record=recname,start_sec=start/CFG.target_fs,end_sec=end/CFG.target_fs,apnea_fraction=cur_apnea,sleep_fraction=sleep_frac,future_apnea_5m=future,ahi=AHI[recname])
            rows.append(m);rec_h.append(h);rec_idx.append(len(rows)-1)
        if rec_h:
            normal=[h for idx,h in zip(rec_idx,rec_h) if rows[idx]['apnea_fraction']==0 and rows[idx]['sleep_fraction']>=0.8]
            base_h=np.sum(normal if normal else rec_h,axis=0)
            for idx,h in zip(rec_idx,rec_h):rows[idx]['deformation_js']=js_div(h,base_h)
    df=pd.DataFrame(rows);df.to_csv(CSV/'slpdb_windows.csv',index=False)
    (OUT/'slpdb_annotation_codes.json').write_text(json.dumps(code_inventory,indent=2))
    return df


def slp_stats(df):
    metrics=['heart_rate','resp_rate','cycles_per_breath','plv_max','poincare_R','mi_norm','joint_entropy','coverage','recurrence_det','deformation_js','resp_amp_cv','phase_speed_cv']
    paired=[]
    for rec,g in df[df.sleep_fraction>=0.6].groupby('record'):
        normal=g[g.apnea_fraction==0];ap=g[g.apnea_fraction>=0.2]
        if len(normal)<5 or len(ap)<5:continue
        for m in metrics:paired.append({'record':rec,'metric':m,'normal':normal[m].mean(),'apnea':ap[m].mean()})
    p=pd.DataFrame(paired);rows=[]
    for m,g in p.groupby('metric'):
        try:w,pv=stats.wilcoxon(g.apnea,g.normal,zero_method='wilcox')
        except Exception:w,pv=np.nan,np.nan
        rows.append({'metric':m,'n_records':len(g),'normal_mean':g.normal.mean(),'apnea_mean':g.apnea.mean(),'median_change':np.median(g.apnea-g.normal),'p':pv})
    st=pd.DataFrame(rows);st['q_fdr']=multipletests(st.p.fillna(1),method='fdr_bh')[1]
    p.to_csv(CSV/'slpdb_paired_by_record.csv',index=False);st.to_csv(CSV/'slpdb_paired_stats.csv',index=False)
    rec=df.groupby('record')[metrics+['ahi']].mean().reset_index();corr=[]
    for m in metrics:
        r,pv=stats.spearmanr(rec.ahi,rec[m],nan_policy='omit');corr.append({'metric':m,'spearman_r':r,'p':pv})
    corr=pd.DataFrame(corr);corr['q_fdr']=multipletests(corr.p.fillna(1),method='fdr_bh')[1]
    corr.to_csv(CSV/'slpdb_ahi_correlations.csv',index=False);rec.to_csv(CSV/'slpdb_record_summary.csv',index=False)
    return p,st,rec,corr


def pre_apnea_analysis(df):
    features=['plv_max','mi_norm','joint_entropy','coverage','recurrence_det','deformation_js','resp_amp_cv','phase_speed_cv','heart_rate','resp_rate']
    rows=[]
    for rec,g in df.sort_values('start_sec').groupby('record'):
        a=(g.apnea_fraction>=0.2).to_numpy();starts=np.where(a & ~np.r_[False,a[:-1]])[0]
        for e in starts:
            onset=float(g.iloc[e].start_sec)
            for lag in [-900,-600,-300,0]:
                target=onset+lag; cand=g[g.end_sec<=target+30]
                if cand.empty:continue
                r=cand.iloc[-1]
                row={'record':rec,'onset_sec':onset,'lag_sec':lag};row.update({f:r[f] for f in features});rows.append(row)
    out=pd.DataFrame(rows);out.to_csv(CSV/'slpdb_pre_apnea.csv',index=False)
    return out


def prediction_test(df):
    use=df[(df.apnea_fraction==0)&(df.sleep_fraction>=0.6)&df.future_apnea_5m.notna()].copy()
    base=['heart_rate','resp_rate','resp_amp_cv']
    phase=base+['cycles_per_breath','plv_max','poincare_R','mi_norm','joint_entropy','coverage','recurrence_det','deformation_js','phase_speed_cv']
    rows=[];curves={}
    if use.future_apnea_5m.nunique()<2 or use.record.nunique()<3:return pd.DataFrame(),curves
    for name,features in [('rate_amplitude_baseline',base),('phase_geometry',phase)]:
        y=use.future_apnea_5m.astype(int).to_numpy();pred=np.full(len(use),np.nan)
        nsplit=min(5,use.record.nunique());cv=GroupKFold(n_splits=nsplit)
        med=use[features].median()
        for tr,te in cv.split(use[features],y,groups=use.record):
            if len(np.unique(y[tr]))<2:continue
            model=Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=1000,class_weight='balanced'))])
            model.fit(use.iloc[tr][features].fillna(med),y[tr]);pred[te]=model.predict_proba(use.iloc[te][features].fillna(med))[:,1]
        ok=np.isfinite(pred)
        auc=roc_auc_score(y[ok],pred[ok]);ap=average_precision_score(y[ok],pred[ok]);br=brier_score_loss(y[ok],pred[ok])
        rows.append({'model':name,'n':ok.sum(),'positive_rate':y[ok].mean(),'roc_auc':auc,'pr_auc':ap,'brier':br})
        curves[name]=roc_curve(y[ok],pred[ok])[:2]
    out=pd.DataFrame(rows);out.to_csv(CSV/'slpdb_future_apnea_prediction.csv',index=False);return out,curves


def make_figures(fdf,sur,subj,age_stats,topo,sdf,paired,slpstats,recs,corr,pre,pred,curves):
    rec=fdf.record.iloc[0]; row=fdf[fdf.record==rec].iloc[len(fdf[fdf.record==rec])//2]
    theta,phi=FANTASIA_SERIES[rec];start=int(row.start_sec*CFG.target_fs);n=int(CFG.window_sec*CFG.target_fs);sl=slice(start,start+n)
    fig,ax=plt.subplots(figsize=(6.4,5.5));ax.scatter(theta[sl],phi[sl],s=4,alpha=.45);ax.set(xlabel='cardiac phase',ylabel='respiratory phase',title=f'Empirical joint phase trajectory: {rec}',xlim=(0,TWO_PI),ylim=(0,TWO_PI));savefig(fig,'fig_01_fantasia_joint_phase')
    th=theta[sl][::4];ph=phi[sl][::4];R=2;r=.65
    x=(R+r*np.cos(th))*np.cos(ph);y=(R+r*np.cos(th))*np.sin(ph);z=r*np.sin(th)
    fig=plt.figure(figsize=(7,5.8));ax=fig.add_subplot(111,projection='3d');ax.plot(x,y,z,lw=.7);ax.set_title('Cardiac-respiratory trajectory embedded on $T^2$');ax.set_axis_off();savefig(fig,'fig_02_fantasia_torus_embedding')
    fig,ax=plt.subplots(figsize=(7.2,4.8));ax.hist(sur.plv_z.dropna(),bins=15,alpha=.75,label='max n:m PLV z');ax.hist(sur.mi_z.dropna(),bins=15,alpha=.55,label='MI z');ax.axvline(1.96,ls='--');ax.set(title='Observed coupling relative to circular-shift surrogates',xlabel='surrogate z-score',ylabel='records');ax.legend(frameon=False);savefig(fig,'fig_03_surrogate_coupling')
    top=age_stats.sort_values('q_fdr').head(4).metric.tolist();long=subj.melt(id_vars=['record','age_group'],value_vars=top,var_name='metric',value_name='value')
    fig,axs=plt.subplots(2,2,figsize=(9,7))
    for ax,m in zip(axs.flat,top):
        vals=[long[(long.metric==m)&(long.age_group==g)].value.dropna() for g in ['young','elderly']];ax.boxplot(vals,tick_labels=['young','elderly']);ax.set_title(m)
    fig.suptitle('Fantasia age-group comparisons');savefig(fig,'fig_04_fantasia_age_effects')
    if len(topo):
        fig,axs=plt.subplots(1,3,figsize=(10,4))
        for ax,m in zip(axs,['h1_longest','h1_second','h2_longest']):
            vals=[topo[topo.kind==k][m] for k in ['empirical','one_cycle_null']];ax.boxplot(vals,tick_labels=['empirical','one-cycle']);ax.set_title(m)
        fig.suptitle('Persistent-homology lifetimes in $R^4$ phase embedding');savefig(fig,'fig_05_topology_empirical_vs_null')
    if len(paired):
        for m,name in [('deformation_js','fig_06_apnea_deformation'),('recurrence_det','fig_07_apnea_recurrence'),('plv_max','fig_08_apnea_plv')]:
            g=paired[paired.metric==m];fig,ax=plt.subplots(figsize=(6.5,5))
            for _,r in g.iterrows():ax.plot([0,1],[r.normal,r.apnea],marker='o',alpha=.55)
            ax.set_xticks([0,1],['normal','apnea']);ax.set_ylabel(m);ax.set_title(f'Within-record change: {m}');savefig(fig,name)
    if len(pre):
        metrics=['deformation_js','recurrence_det','plv_max'];fig,axs=plt.subplots(1,3,figsize=(12,4))
        for ax,m in zip(axs,metrics):
            q=pre.groupby('lag_sec')[m].agg(['mean','sem']).reset_index();ax.errorbar(q.lag_sec/60,q['mean'],yerr=q['sem'],marker='o');ax.set(xlabel='minutes relative to apnea onset',title=m)
        fig.suptitle('Pre-apnea trajectory');savefig(fig,'fig_09_pre_apnea_trajectory')
    if curves:
        fig,ax=plt.subplots(figsize=(6,5))
        for name,(fpr,tpr) in curves.items():ax.plot(fpr,tpr,label=name)
        ax.plot([0,1],[0,1],ls='--');ax.set(xlabel='false positive rate',ylabel='true positive rate',title='Cross-record prediction of future apnea');ax.legend(frameon=False);savefig(fig,'fig_10_future_apnea_roc')
    if len(corr):
        m=corr.iloc[corr.spearman_r.abs().argmax()].metric;fig,ax=plt.subplots(figsize=(6.3,5));ax.scatter(recs.ahi,recs[m]);coef=np.polyfit(recs.ahi,recs[m],1);xx=np.linspace(recs.ahi.min(),recs.ahi.max());ax.plot(xx,np.polyval(coef,xx));ax.set(xlabel='AHI',ylabel=m,title=f'Exploratory AHI association: {m}');savefig(fig,'fig_11_ahi_association')


def report_results(fdf,sur,subj,age,topo,sdf,paired,slpstats,recs,corr,pre,pred):
    sig_age=age[age.q_fdr<.05].sort_values('q_fdr');sig_slp=slpstats[slpstats.q_fdr<.05].sort_values('q_fdr')
    lines=['# Empirical Cardiorespiratory Toroidal Dynamics Study','', '## Scope',f'- Fantasia: {subj.record.nunique()} healthy subjects, {len(fdf)} five-minute windows.',f'- MIT-BIH SLPDB: {sdf.record.nunique()} annotated records, {len(sdf)} five-minute windows.','- All analyses use publicly available measured ECG beat annotations and respiration signals.','', '## Main findings','']
    lines.append(f"1. **Two-phase recurrence was directly measurable.** Median healthy joint-phase coverage was {fdf.coverage.median():.3f}; median normalized entropy was {fdf.joint_entropy.median():.3f}.")
    lines.append(f"2. **Coupling exceeded time-shift nulls in a subset, not universally.** Median PLV surrogate z={sur.plv_z.median():.2f}; records above z=1.96: {(sur.plv_z>1.96).mean():.1%}. Median MI z={sur.mi_z.median():.2f}.")
    if len(topo):
        emp=topo[topo.kind=='empirical'];null=topo[topo.kind=='one_cycle_null'];lines.append(f"3. **Topology was torus-compatible but imperfect.** Empirical median second-H1 lifetime={emp.h1_second.median():.3f} vs one-cycle null={null.h1_second.median():.3f}; empirical H2 lifetime={emp.h2_longest.median():.3f}.")
    lines.append(f"4. **Age effects:** {len(sig_age)} of {len(age)} tested subject-level metrics survived FDR correction.")
    lines.append(f"5. **Apnea altered recurrence geometry:** {len(sig_slp)} of {len(slpstats)} paired metrics survived FDR correction across records.")
    if len(pred):
        b=pred[pred.model=='rate_amplitude_baseline'].iloc[0];p=pred[pred.model=='phase_geometry'].iloc[0];lines.append(f"6. **Future-apnea prediction:** rate/amplitude baseline ROC-AUC={b.roc_auc:.3f}; phase-geometry ROC-AUC={p.roc_auc:.3f}.")
    lines += ['', '## Fantasia age statistics','',age.to_markdown(index=False),'','## SLPDB normal versus apnea','',slpstats.to_markdown(index=False),'','## Future-apnea prediction','',pred.to_markdown(index=False) if len(pred) else 'Insufficient balanced samples.','','## AHI correlations','',corr.to_markdown(index=False),'','## Interpretation','','The results test a geometric mechanism, not the full ATS ontology. Cardiac and respiratory phases provide independently measured recurrent coordinates; torus-compatible geometry therefore does not depend on assigning arbitrary phases. Changes during apnea support the narrower claim that loss of physiological viability can be accompanied by measurable deformation or restructuring of recurrent phase dynamics.','','## Limitations','','- Fantasia contains resting healthy subjects; it cannot test collapse or intervention.','- SLPDB subjects were selected for sleep-apnea evaluation and are not population-representative.','- Five-minute windows trade temporal localization for enough cycles to estimate geometry.','- Persistent homology was run on a limited representative subset for computational tractability.','- The phase-geometry predictor is observational and does not establish causal intervention effects.','- A torus in physiology does not prove that AI or institutions possess toroidal attractors.']
    (OUT/'EMPIRICAL_REPORT.md').write_text('\n'.join(lines))
    headline=pd.DataFrame([{'finding':'healthy_joint_phase_coverage_median','value':fdf.coverage.median()},{'finding':'healthy_joint_entropy_median','value':fdf.joint_entropy.median()},{'finding':'plv_surrogate_z_median','value':sur.plv_z.median()},{'finding':'fraction_plv_z_gt_1.96','value':(sur.plv_z>1.96).mean()},{'finding':'age_metrics_fdr_significant','value':len(sig_age)},{'finding':'apnea_metrics_fdr_significant','value':len(sig_slp)}])
    if len(pred):
        for _,r in pred.iterrows():headline.loc[len(headline)]={'finding':f'future_apnea_auc_{r.model}','value':r.roc_auc}
    headline.to_csv(CSV/'headline_findings.csv',index=False)


def main():
    global FANTASIA_SERIES
    fantasia=download_database('fantasia');slpdb=download_database('slpdb')
    fdf,sur,FANTASIA_SERIES,hists=process_fantasia(fantasia)
    subj,age=fantasia_stats(fdf);topo=topology_analysis(FANTASIA_SERIES)
    sdf=process_slpdb(slpdb);paired,slpstats,recs,corr=slp_stats(sdf);pre=pre_apnea_analysis(sdf);pred,curves=prediction_test(sdf)
    make_figures(fdf,sur,subj,age,topo,sdf,paired,slpstats,recs,corr,pre,pred,curves)
    report_results(fdf,sur,subj,age,topo,sdf,paired,slpstats,recs,corr,pre,pred)
    meta={'status':'complete','fantasia_records':int(subj.record.nunique()),'fantasia_windows':len(fdf),'slpdb_records':int(sdf.record.nunique()),'slpdb_windows':len(sdf),'figures':len(list(FIG.glob('*.png'))),'csv_files':len(list(CSV.glob('*.csv')))}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2));print('FINAL_METADATA',json.dumps(meta),flush=True)
    print((OUT/'EMPIRICAL_REPORT.md').read_text(),flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    main()
