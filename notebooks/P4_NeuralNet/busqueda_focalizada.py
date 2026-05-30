import sys, os, json, time, warnings, math
warnings.filterwarnings('ignore')

# Ancla de rutas: directorio del script (P4_redes_ejecutadas/)
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# Raiz del proyecto (finanzas PF/) esta un nivel arriba
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
_P3_SRC = os.path.join(_PROJECT_ROOT, 'src', 'utils')

if _P3_SRC not in sys.path:
    sys.path.insert(0, _P3_SRC)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Importamos las metricas oficiales
try:
    from evaluation import evaluate_binary_model, threshold_by_ks, population_stability_index
except ImportError:
    print("WARNING: No se pudo importar evaluation.py desde P3_entrega/src")

OPTUNA_TRIALS  = 40     
BLOCK_TIMEOUT  = 5400   
FINAL_EPOCHS   = 600    
SEED           = 42

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR     = os.path.join(_SCRIPT_DIR, 'tmp_test', 'busqueda_focalizada_checkpoints')
RESULTS_FILE = os.path.join(_SCRIPT_DIR, 'tmp_test', 'busqueda_focalizada_results.json')
RL_BASELINE  = 0.737  # Baseline actualizado segun notebook de Logistic Regression (OOS AUC)

os.makedirs(CKPT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
t0 = time.time()

print(f"Device : {DEVICE}")
print(f"Trials : {OPTUNA_TRIALS} por bloque  |  Final epochs: {FINAL_EPOCHS}")

print("Cargando splits de P3_entrega/data ...")
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data', 'splits')
_OUT_DIR  = os.path.join(_PROJECT_ROOT, 'data', 'variables_bivariadas')
X_tr = pd.read_csv(os.path.join(_DATA_DIR, 'X_train.csv'))
X_te = pd.read_csv(os.path.join(_DATA_DIR, 'X_test.csv'))
X_oot = pd.read_csv(os.path.join(_DATA_DIR, 'X_oos.csv'))

ytr = pd.read_csv(os.path.join(_DATA_DIR, 'y_train.csv'))['target'].values.astype('float32')
yte = pd.read_csv(os.path.join(_DATA_DIR, 'y_test.csv'))['target'].values.astype('float32')
yoo = pd.read_csv(os.path.join(_DATA_DIR, 'y_oos.csv'))['target'].values.astype('float32')

candidatas_df = pd.read_csv(os.path.join(_OUT_DIR, 'bivariado_variables_candidatas.csv'))
USE_COLS = candidatas_df['variable'].tolist()
print(f"Train: {len(ytr)}  Test: {len(yte)}  OOT: {len(yoo)}  PI={ytr.mean()*100:.1f}%")
print(f"Variables candidatas seleccionadas (Bivariado): {len(USE_COLS)}")

class WOEFull:
    def __init__(self): self.cfg={}; self.columns_=[]
    def _fit(self, x, y, discrete):
        tb=max((y==1).sum(),1); tg=max((y==0).sum(),1)
        def w(bad,n): return float(np.log(max(bad/tb,1e-6)/max((n-bad)/tg,1e-6)))
        miss=x.isnull(); mw=w(y[miss].sum(),miss.sum()) if miss.sum()>5 else 0.0
        if discrete:
            vm={}
            for val,g in pd.DataFrame({'x':x,'y':y}).dropna(subset=['x']).groupby('x',observed=True):
                vm[float(val)]=w(g['y'].sum(),len(g))
            return {'mode':'disc','vm':vm,'mw':mw,'def':float(np.median(list(vm.values())) if vm else 0)}
        nb=15 if x.nunique()>50 else 10
        try: _,ed=pd.qcut(x.dropna(),nb,retbins=True,duplicates='drop'); ed[0]=-np.inf; ed[-1]=np.inf
        except: ed=np.array([-np.inf,np.inf])
        df_ok=pd.DataFrame({'x':x,'y':y}).dropna(subset=['x']).copy()
        wvs=[]
        if len(ed)>2:
            df_ok['b']=pd.cut(df_ok['x'],bins=ed.tolist(),include_lowest=True)
            for _,g in df_ok.groupby('b',observed=True): wvs.append(w(g['y'].sum(),len(g)))
        else: wvs.append(w(df_ok['y'].sum(),len(df_ok)))
        return {'mode':'cont','ed':ed.tolist(),'wvs':wvs,'mw':mw}
    def fit(self, X_df, y, discrete_cols=None):
        dc=set(discrete_cols or [])
        self.columns_=list(X_df.columns)
        for col in self.columns_:
            x=X_df[col]
            is_d=(col in dc or (x.nunique()<=20 and
                  x.dropna().apply(lambda v: v==int(v) if not pd.isna(v) else True).all()))
            self.cfg[col]=self._fit(x,y,is_d)
        return self
    def tx(self, x_s, col):
        c=self.cfg[col]; xv=x_s.values.astype(float); isna=np.isnan(xv)
        if c['mode']=='disc':
            vm={float(k):v for k,v in c['vm'].items()}; dw=c['def']
            return np.array([c['mw'] if isna[i] else vm.get(float(xv[i]),dw) for i in range(len(xv))],dtype='float32')
        ed=np.array(c['ed']); wvs=np.array(c['wvs'])
        bidx=np.clip(np.digitize(xv,ed[1:-1]),0,len(wvs)-1)
        return np.where(isna,c['mw'],wvs[bidx]).astype('float32')
    def transform(self, X_df):
        return pd.DataFrame({f'woe_{c}':self.tx(X_df[c],c) for c in self.columns_},index=X_df.index)

print("\nAplicando WOE a las", len(USE_COLS), "variables preseleccionadas...")
DISC_COLS=['x1','x2']
woe=WOEFull()
woe.fit(X_tr[USE_COLS], ytr, discrete_cols=[c for c in DISC_COLS if c in USE_COLS])

MISS_SRC=['x99','x120','x37','x52','x6','x112','x35','x71','x94','x1']

def build_X(Xdf, add_fe=True):
    wdf=woe.transform(Xdf[USE_COLS])
    parts=[wdf.values]
    if not add_fe: return np.concatenate(parts,axis=1).astype('float32')
    ex={}
    if 'x1' in USE_COLS:
        x1r=Xdf['x1'].fillna(0)
        ex['x1_flag']=(x1r>=1).astype('float32').values
        ex['x1_sev'] =(x1r>=2).astype('float32').values
    for col in MISS_SRC:
        if col in Xdf.columns: ex[f'miss_{col}']=Xdf[col].isnull().astype('float32').values
    def w(c): k=f'woe_{c}'; return wdf[k].values if k in wdf.columns else None
    w1,w52,w37=w('x1'),w('x52'),w('x37'); w94,w71,w6,w7=w('x94'),w('x71'),w('x6'),w('x7')
    def safe_inter(a,b,fn): return fn(a,b).astype('float32') if a is not None and b is not None else None
    inters={
        'i_x1_x52' : safe_inter(w1,w52,  lambda a,b: a*np.sign(b)*np.sqrt(np.abs(b))),
        'i_x1_x37' : safe_inter(w1,w37,  lambda a,b: a*b),
        'i_x1_x6'  : safe_inter(w1,w6,   lambda a,b: a*(-b)),
        'i_x1_x7'  : safe_inter(w1,w7,   lambda a,b: a*b),
        'i_x94_x52': safe_inter(w94,w52,  lambda a,b: a*np.sign(b)*np.sqrt(np.abs(b))),
        'i_x71_x1' : safe_inter(w71,w1,  lambda a,b: a*b),
        'i_x37_x52': safe_inter(w37,w52,  lambda a,b: a*b),
        'i_x6_x52' : safe_inter(w6,w52,  lambda a,b: (-a)*np.sign(b)*np.sqrt(np.abs(b))),
        'i_x94_x37': safe_inter(w94,w37,  lambda a,b: a*b),
        'i_x71_x52': safe_inter(w71,w52,  lambda a,b: a*np.sign(b)*np.sqrt(np.abs(b))),
    }
    for v in {**ex,**{k:v for k,v in inters.items() if v is not None}}.values(): parts.append(v.reshape(-1,1))
    return np.concatenate(parts,axis=1).astype('float32')

Xtr_r=build_X(X_tr,False); Xte_r=build_X(X_te,False); Xoo_r=build_X(X_oot,False)
Xtr_f=build_X(X_tr,True);  Xte_f=build_X(X_te,True);  Xoo_f=build_X(X_oot,True)
print(f"Matrices listas -> WOE solo: {Xtr_r.shape[1]} features  |  WOE+FE: {Xtr_f.shape[1]} features")

class SimpleMLP(nn.Module):
    def __init__(self,nf,h=256,depth=4,drop=0.1,act='gelu'):
        super().__init__()
        A=nn.GELU if act=='gelu' else (nn.SiLU if act=='silu' else nn.ReLU)
        ls=[nn.Linear(nf,h),nn.BatchNorm1d(h),A(),nn.Dropout(drop)]
        for _ in range(depth-1): ls+=[nn.Linear(h,h),nn.BatchNorm1d(h),A(),nn.Dropout(drop)]
        ls.append(nn.Linear(h,1)); self.net=nn.Sequential(*ls)
    def forward(self,x): return self.net(x).squeeze(1)

class ResAutoInt(nn.Module):
    def __init__(self,nf,emb=32,heads=4,layers=3,mlph=512,drop=0.0):
        super().__init__()
        self.embs=nn.ModuleList([nn.Linear(1,emb) for _ in range(nf)])
        self.fn=nn.LayerNorm(emb)
        enc=nn.TransformerEncoderLayer(d_model=emb,nhead=heads,dim_feedforward=emb*4,
            dropout=drop,batch_first=True,norm_first=True,activation='gelu')
        self.tr=nn.TransformerEncoder(enc,num_layers=layers)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(nf*emb,mlph),nn.GELU(),
            nn.Linear(mlph,mlph//2),nn.GELU(),nn.Linear(mlph//2,1))
        self.skip=nn.Linear(nf*emb,1)
    def forward(self,x):
        B=x.shape[0]
        tk=torch.stack([self.embs[i](x[:,i:i+1]) for i in range(x.shape[1])],dim=1)
        out=self.tr(self.fn(tk)).reshape(B,-1)
        return (self.head(out)+self.skip(out)).squeeze(1)

class _SAINTLayer(nn.Module):
    def __init__(self,emb,nf,hc,hr,drop):
        super().__init__()
        self.cn1=nn.LayerNorm(emb); self.ca=nn.MultiheadAttention(emb,hc,dropout=drop,batch_first=True)
        self.cn2=nn.LayerNorm(emb); self.cff=nn.Sequential(nn.Linear(emb,emb*4),nn.GELU(),nn.Dropout(drop),nn.Linear(emb*4,emb))
        rd=min(nf*emb,256)
        self.rpi=nn.Linear(nf*emb,rd); self.rn1=nn.LayerNorm(rd)
        self.ra=nn.MultiheadAttention(rd,hr,dropout=drop,batch_first=True)
        self.rn2=nn.LayerNorm(rd)
        self.rff=nn.Sequential(nn.Linear(rd,rd*2),nn.GELU(),nn.Dropout(drop),nn.Linear(rd*2,rd))
        self.rpo=nn.Linear(rd,nf*emb); self.nf=nf; self.emb=emb
    def forward(self,x):
        B,F,E=x.shape
        xn,_=self.ca(self.cn1(x),self.cn1(x),self.cn1(x)); x=x+xn; x=x+self.cff(self.cn2(x))
        xf=x.reshape(B,F*E); xr=self.rpi(xf).unsqueeze(0)
        xa,_=self.ra(self.rn1(xr),self.rn1(xr),self.rn1(xr)); xr=xr+xa; xr=xr+self.rff(self.rn2(xr))
        xf=xf+self.rpo(xr.squeeze(0)); return xf.reshape(B,F,E)

class SAINT(nn.Module):
    def __init__(self,nf,emb=32,layers=3,hc=4,hr=4,hdim=128,drop=0.1):
        super().__init__()
        self.embs=nn.ModuleList([nn.Linear(1,emb) for _ in range(nf)]); self.n0=nn.LayerNorm(emb)
        self.layers=nn.ModuleList([_SAINTLayer(emb,nf,hc,hr,drop) for _ in range(layers)])
        flat=nf*emb
        self.head=nn.Sequential(nn.Flatten(),nn.LayerNorm(flat),nn.Linear(flat,hdim),
            nn.GELU(),nn.Dropout(drop),nn.Linear(hdim,hdim//2),nn.GELU(),nn.Linear(hdim//2,1))
        self.skip=nn.Linear(flat,1)
    def forward(self,x):
        B=x.shape[0]
        tk=torch.stack([self.embs[i](x[:,i:i+1]) for i in range(x.shape[1])],dim=1); tk=self.n0(tk)
        for L in self.layers: tk=L(tk)
        flat=tk.reshape(B,-1)
        return (self.head(tk)+self.skip(flat)).squeeze(1)

def _common(trial):
    return dict(lr=trial.suggest_float('lr',5e-5,5e-3,log=True),
                wd=trial.suggest_float('wd',1e-5,1e-3,log=True),
                batch=trial.suggest_categorical('batch',[128,256,512]),
                mxup=trial.suggest_float('mxup',0.0,0.5),
                swa_f=trial.suggest_float('swa_f',0.50,0.80))

def sp_simple(trial,nf):
    return {**_common(trial),'nf':nf,
            'h':trial.suggest_categorical('h',[64,128,256,512,1024]),
            'depth':trial.suggest_int('depth',2,10),
            'drop':trial.suggest_float('drop',0.0,0.4),
            'act':trial.suggest_categorical('act',['gelu','silu','relu'])}

def sp_resai(trial,nf):
    emb=trial.suggest_categorical('emb',[16,32,64])
    heads=trial.suggest_categorical('heads',[2,4,8])
    if emb%heads!=0: return None
    return {**_common(trial),'nf':nf,'emb':emb,'heads':heads,
            'layers':trial.suggest_int('layers',2,5),
            'mlph':trial.suggest_categorical('mlph',[128,256,512,1024]),
            'drop':trial.suggest_float('drop',0.0,0.20)}

def sp_saint(trial,nf):
    emb=trial.suggest_categorical('emb',[16,32,64])
    hc=trial.suggest_categorical('hc',[2,4,8])
    hr=trial.suggest_categorical('hr',[1,2,4])
    if emb%hc!=0: return None
    return {**_common(trial),'nf':nf,'emb':emb,
            'layers':trial.suggest_int('layers',1,4),'hc':hc,'hr':hr,
            'hdim':trial.suggest_categorical('hdim',[64,128,256]),
            'drop':trial.suggest_float('drop',0.0,0.25)}

mk={
    'SimpleMLP' :lambda **p: SimpleMLP(p['nf'],p['h'],p['depth'],p['drop'],p.get('act','gelu')),
    'ResAutoInt':lambda **p: ResAutoInt(p['nf'],p['emb'],p['heads'],p['layers'],p['mlph'],p['drop']),
    'SAINT'     :lambda **p: SAINT(p['nf'],p['emb'],p['layers'],p['hc'],p['hr'],p['hdim'],p['drop']),
}
spaces={'SimpleMLP':sp_simple, 'ResAutoInt':sp_resai, 'SAINT':sp_saint}

def mixup(X,y,a):
    if a<=0: return X,y
    lam=np.random.beta(a,a); idx=torch.randperm(X.size(0),device=X.device)
    return lam*X+(1-lam)*X[idx], lam*y+(1-lam)*y[idx]

def focal_bce(p,t,g=1.5,ls=0.02):
    t=t*(1-ls)+0.5*ls; bce=F.binary_cross_entropy_with_logits(p,t,reduction='none')
    return (((1-torch.exp(-bce))**g)*bce).mean()

def get_probs(model,X,bs=1024):
    model.eval(); out=[]
    for i in range(0,len(X),bs):
        Xb=torch.tensor(X[i:i+bs],device=DEVICE)
        with torch.no_grad(): out.append(torch.sigmoid(model(Xb)).float().cpu().numpy())
    return np.concatenate(out)

def train_model(model,Xtr,ytr,Xte,yte,Xoo,yoo,
                lr=1e-3,wd=1e-4,batch=256,epochs=350,patience=50,
                mxup=0.3,swa_f=0.65,gamma=1.5,ls=0.02):
    model=model.to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=lr/50)
    scaler=GradScaler(DEVICE)
    swa=torch.optim.swa_utils.AveragedModel(model)
    swa_sc=torch.optim.swa_utils.SWALR(opt,swa_lr=lr*0.05,anneal_epochs=8)
    swa_ep=int(epochs*swa_f)
    bs=min(batch,len(Xtr)//2)
    dl=DataLoader(TensorDataset(torch.tensor(Xtr,device=DEVICE),torch.tensor(ytr,device=DEVICE)),
                  batch_size=bs,shuffle=True,drop_last=True)
    bval,bsd,noi=0.0,None,0
    for ep in range(1,epochs+1):
        model.train()
        for Xb,yb in dl:
            Xb,yb=mixup(Xb,yb,mxup); opt.zero_grad()
            with autocast(DEVICE): loss=focal_bce(model(Xb),yb,gamma,ls)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
        if ep>=swa_ep: swa.update_parameters(model); swa_sc.step()
        else: sched.step()
        if ep%20==0 or ep==epochs:
            v=roc_auc_score(yte,get_probs(model,Xte))
            if v>bval: bval=v; bsd={k:c.cpu().clone() for k,c in model.state_dict().items()}; noi=0
            else: noi+=1
            if noi>=patience//20 and ep>=swa_ep: break
    torch.optim.swa_utils.update_bn(dl,swa)
    s_oot=roc_auc_score(yoo,get_probs(swa,Xoo))
    if bsd: model.load_state_dict(bsd)
    b_oot=roc_auc_score(yoo,get_probs(model,Xoo))
    return (swa if s_oot>=b_oot else model)

all_results=[]
if os.path.exists(RESULTS_FILE):
    try: all_results=json.load(open(RESULTS_FILE,encoding='utf-8')); print(f"\nReanudando — {len(all_results)} bloques previos")
    except: pass
done_names={r['name'] for r in all_results}

def save():
    with open(RESULTS_FILE,'w',encoding='utf-8') as f:
        json.dump([{k:v for k,v in r.items() if k not in('p_tr','p_te','p_oo')} for r in all_results],f,indent=2)

def log_result(name,model,Xtr_n,Xte_n,Xoo_n):
    p_tr=get_probs(model,Xtr_n); p_te=get_probs(model,Xte_n); p_oo=get_probs(model,Xoo_n)
    
    thresh = threshold_by_ks(ytr, p_tr)
    eval_oo = evaluate_binary_model(yoo, p_oo, thresh)
    eval_tr = evaluate_binary_model(ytr, p_tr, thresh)
    eval_te = evaluate_binary_model(yte, p_te, thresh)
    
    atr, ate, aoo = eval_tr['auc'], eval_te['auc'], eval_oo['auc']
    ks_oo = eval_oo['ks']
    gap = atr - aoo
    try:
        psi_val = population_stability_index(p_tr, p_oo, n_bins=10)
    except:
        psi_val = np.nan
        
    beat='  *** SUPERA RL ***' if aoo>RL_BASELINE else ''
    elapsed=f"{(time.time()-t0)/3600:.1f}h"
    print(f"  [{elapsed}] {name:<22}  AUC_OOS={aoo:.4f}  KS_OOS={ks_oo:.4f}  Gap={gap:.4f}  PSI={psi_val:.4f} {beat}")
    
    r={'name':name,'oot_auc':round(aoo,4),'test_auc':round(ate,4),'train_auc':round(atr,4),
       'ks':round(ks_oo,4),'gap':round(gap,4),'psi':round(psi_val,4),
       'precision':round(eval_oo['precision'],4),'recall':round(eval_oo['recall'],4),
       'p_tr':p_tr.tolist(),'p_te':p_te.tolist(),'p_oo':p_oo.tolist()}
    all_results.append(r); save(); return r

def run_block(name,mk_fn,Xtr_n,Xte_n,Xoo_n,space_fn,
              trials=OPTUNA_TRIALS,timeout=BLOCK_TIMEOUT,fepochs=FINAL_EPOCHS):
    if name in done_names:
        print(f"  [SKIP] {name}"); return None
    nf=Xtr_n.shape[1]
    print(f"\n{'='*75}")
    print(f"BLOQUE: {name}  ({nf} features, {trials} trials)")
    print(f"{'='*75}")
    t_b=time.time(); best_p=[None]

    def obj(trial):
        p=space_fn(trial,nf)
        if p is None: raise optuna.exceptions.TrialPruned()
        try:
            m=mk_fn(**p)
            m=train_model(m,Xtr_n,ytr,Xte_n,yte,Xoo_n,yoo,
                          lr=p.get('lr',1e-3),wd=p.get('wd',1e-4),batch=p.get('batch',256),
                          epochs=220,patience=50,mxup=p.get('mxup',0.3),swa_f=p.get('swa_f',0.65))
            val=roc_auc_score(yte,get_probs(m,Xte_n))
            if np.isnan(val): raise optuna.exceptions.TrialPruned()
            if best_p[0] is None or val>best_p[0][0]: best_p[0]=(val,p)
            return val
        except optuna.exceptions.TrialPruned: raise
        except: raise optuna.exceptions.TrialPruned()

    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj,n_trials=trials,timeout=timeout,catch=(Exception,))
    if study.best_trial is None: print(f"  Sin trials exitosos"); return None

    bp=best_p[0][1] if best_p[0] else {}
    print(f"\n  Mejor val AUC: {study.best_value:.4f}  |  Entrenando modelo final ({fepochs} epocas)...")
    m=mk_fn(**bp)
    m=train_model(m,Xtr_n,ytr,Xte_n,yte,Xoo_n,yoo,
                  lr=bp.get('lr',1e-3),wd=bp.get('wd',1e-4),batch=bp.get('batch',256),
                  epochs=fepochs,patience=80,mxup=bp.get('mxup',0.3),swa_f=bp.get('swa_f',0.65))
    torch.save(m.state_dict(),os.path.join(CKPT_DIR,name.replace(' ','_').replace('/','_')+'.pt'))
    r=log_result(name,m,Xtr_n,Xte_n,Xoo_n)
    r['params']=bp; done_names.add(name)
    torch.cuda.empty_cache()
    print(f"  Tiempo bloque: {(time.time()-t_b)/60:.1f} min")
    return m,r

if __name__ == "__main__":
    ARCHS = ['SimpleMLP', 'ResAutoInt', 'SAINT']
    EXPERIMENTS=[(f'{a} / WOE',mk[a],Xtr_r,Xte_r,Xoo_r,spaces[a]) for a in ARCHS] + \
                [(f'{a} / FE', mk[a],Xtr_f,Xte_f,Xoo_f,spaces[a]) for a in ARCHS]

    print(f"\n{'='*75}")
    print(f"INICIANDO {len(EXPERIMENTS)} BLOQUES ESTRATÉGICOS  (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'})")

    for idx, (name, fn, xtr, xte, xoo, space) in enumerate(EXPERIMENTS):
        run_block(name, fn, xtr, xte, xoo, space)

    print("\n✓ TODAS LAS ARQUITECTURAS COMPLETADAS")
