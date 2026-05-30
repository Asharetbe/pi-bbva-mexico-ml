"""
gran_busqueda.py  —  Busqueda de arquitecturas neuronales para PI BBVA Mexico

DATOS  (splits reproducibles, identicos a los demas modelos del proyecto):
  data/splits/X_train.csv  /  y_train.csv   — entrenamiento
  data/splits/X_test.csv   /  y_test.csv    — validacion
  data/splits/X_oos.csv    /  y_oos.csv     — out-of-sample

VARIABLES: data/variables_bivariadas/bivariado_variables_candidatas.csv (si SELECTED_VARS esta vacio)
EVALUACION: src/utils/evaluation.py  (ks_statistic, evaluate_binary_model, psi, threshold_by_ks)

INSTRUCCIONES:
  1. Pon en SELECTED_VARS las variables que quieres usar.
     Deja la lista VACIA para usar las variables candidatas del bivariado.
  2. Ejecuta desde el directorio 'finanzas PF/':
         python gran_busqueda.py
  3. Resultados: tmp_test/gran_busqueda_results.json
     Checkpoints: tmp_test/gran_busqueda_checkpoints/
     El script se puede interrumpir y reanudar — guarda progreso despues de cada bloque.
"""
import sys, os, json, time, warnings, math
warnings.filterwarnings('ignore')

# Ancla de rutas: directorio del script (P4_redes_ejecutadas/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Raiz del proyecto (finanzas PF/) esta un nivel arriba
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
_P3_SRC = os.path.join(_PROJECT_ROOT, 'src', 'utils')

if _P3_SRC not in sys.path:
    sys.path.insert(0, _P3_SRC)
from evaluation import ks_statistic, evaluate_binary_model, population_stability_index, threshold_by_ks

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
#  CONFIGURACION — UNICO LUGAR QUE DEBES MODIFICAR
# ============================================================
SELECTED_VARS = [
    # Ejemplos:  'x1', 'x52', 'x6', 'x37', 'x94', 'x71', ...
    # Deja VACIO para usar las 120 variables originales
]

OPTUNA_TRIALS  = 40     # trials de Optuna por bloque
BLOCK_TIMEOUT  = 5400   # segundos maximos por bloque (90 min)
FINAL_EPOCHS   = 600    # epocas del modelo final por bloque
CV_FOLDS       = 5      # folds para cross-validacion de top modelos
SEED           = 42

# Deja VACIO para correr todos los bloques; o lista los nombres exactos a ejecutar.
RUN_ONLY = [
    'SimpleMLP / FE',
    'ResAutoInt / FE',
    'SAINT / FE',
    'SimpleMLP / WOE',
    'ResAutoInt / WOE',
    'SAINT / WOE',
]
# ============================================================

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR     = os.path.join(_SCRIPT_DIR, 'tmp_test', 'gran_busqueda_checkpoints')
RESULTS_FILE = os.path.join(_SCRIPT_DIR, 'tmp_test', 'gran_busqueda_results.json')
RL_BASELINE  = 0.8978

os.makedirs(CKPT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
t0 = time.time()

print(f"Device : {DEVICE}  {'GPU: ' + torch.cuda.get_device_name(0) if DEVICE=='cuda' else ''}")
print(f"Trials : {OPTUNA_TRIALS} por bloque  |  Final epochs: {FINAL_EPOCHS}")

# ── Datos (splits reproducibles desde P3_entrega/data/) ──────────────────────
SCRIPT_DIR = _SCRIPT_DIR
DATA_DIR   = os.path.join(_PROJECT_ROOT, 'data', 'splits')

X_tr  = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
y_tr  = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))['target']
X_te  = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_te  = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))['target']
X_oot = pd.read_csv(os.path.join(DATA_DIR, 'X_oos.csv'))
y_oot = pd.read_csv(os.path.join(DATA_DIR, 'y_oos.csv'))['target']

# Train+test combinado — usado en cross-validacion final
X_tmp = pd.concat([X_tr, X_te], ignore_index=True)
y_tmp = pd.concat([y_tr, y_te], ignore_index=True)

ALL_FEATS = [c for c in X_tr.columns]

ytr = y_tr.values.astype('float32')
yte = y_te.values.astype('float32')
yoo = y_oot.values.astype('float32')

# Variables candidatas: desde bivariado si SELECTED_VARS esta vacio
if SELECTED_VARS:
    USE_COLS = SELECTED_VARS
else:
    _cand_path = os.path.join(_PROJECT_ROOT, 'data', 'variables_bivariadas', 'bivariado_variables_candidatas.csv')
    USE_COLS = pd.read_csv(_cand_path)['variable'].tolist()
print(f"Train: {len(ytr)}  Test: {len(yte)}  OOS: {len(yoo)}  PI={ytr.mean()*100:.1f}%")
print(f"Variables a usar: {len(USE_COLS)}  ({'seleccionadas' if SELECTED_VARS else 'bivariado candidatas'})")

# ══════════════════════════════════════════════════════════════════════════════
# WOE  (discreto por valor unico para variables enteras, qcut para continuas)
# ══════════════════════════════════════════════════════════════════════════════
class WOEFull:
    def __init__(self): self.cfg={}; self.columns_=[]

    def _fit(self, x, y, discrete):
        tb=max((y==1).sum(),1); tg=max((y==0).sum(),1)
        def w(bad,n): return float(np.log(max(bad/tb,1e-6)/max((n-bad)/tg,1e-6)))
        miss=x.isnull(); mw=w(y.values[miss.values].sum(),miss.sum()) if miss.sum()>5 else 0.0
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
        # Reset indices para compatibilidad con pandas 3.x (boolean indexing cross-Series)
        X_df = X_df.reset_index(drop=True)
        y    = y.reset_index(drop=True)
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
            return np.array([c['mw'] if isna[i] else vm.get(float(xv[i]),dw)
                             for i in range(len(xv))],dtype='float32')
        ed=np.array(c['ed']); wvs=np.array(c['wvs'])
        bidx=np.clip(np.digitize(xv,ed[1:-1]),0,len(wvs)-1)
        return np.where(isna,c['mw'],wvs[bidx]).astype('float32')

    def transform(self, X_df):
        return pd.DataFrame({f'woe_{c}':self.tx(X_df[c],c) for c in self.columns_},index=X_df.index)

print("\nFitteando WOE...")
DISC_COLS=['x1','x2']
woe=WOEFull()
woe.fit(X_tr[USE_COLS], y_tr, discrete_cols=[c for c in DISC_COLS if c in USE_COLS])

# ── Feature engineering (interacciones + missing indicators) ──────────────────
MISS_SRC=['x99','x120','x37','x52','x6','x112','x35','x71','x94','x1']

def build_X(Xdf, add_fe=True):
    wdf=woe.transform(Xdf[USE_COLS])
    parts=[wdf.values]
    if not add_fe:
        return np.concatenate(parts,axis=1).astype('float32')
    ex={}
    if 'x1' in USE_COLS:
        x1r=Xdf['x1'].fillna(0)
        ex['x1_flag']=(x1r>=1).astype('float32').values
        ex['x1_sev'] =(x1r>=2).astype('float32').values
    for col in MISS_SRC:
        if col in Xdf.columns:
            ex[f'miss_{col}']=Xdf[col].isnull().astype('float32').values
    def w(c):
        k=f'woe_{c}'; return wdf[k].values if k in wdf.columns else None
    w1,w52,w37=w('x1'),w('x52'),w('x37')
    w94,w71,w6,w7=w('x94'),w('x71'),w('x6'),w('x7')
    def safe_inter(a,b,fn):
        if a is not None and b is not None: return fn(a,b).astype('float32')
        return None
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
    for v in {**ex,**{k:v for k,v in inters.items() if v is not None}}.values():
        parts.append(v.reshape(-1,1))
    return np.concatenate(parts,axis=1).astype('float32')

Xtr_r=build_X(X_tr,False); Xte_r=build_X(X_te,False); Xoo_r=build_X(X_oot,False)
Xtr_f=build_X(X_tr,True);  Xte_f=build_X(X_te,True);  Xoo_f=build_X(X_oot,True)
nf_r,nf_f=Xtr_r.shape[1],Xtr_f.shape[1]
assert not np.isnan(Xtr_r).any() and not np.isnan(Xtr_f).any()
print(f"WOE solo: {nf_r} features  |  WOE+FE: {nf_f} features")

# ══════════════════════════════════════════════════════════════════════════════
# ARQUITECTURAS  (20 familias distintas)
# ══════════════════════════════════════════════════════════════════════════════

# 1. MLP estandar
class SimpleMLP(nn.Module):
    def __init__(self,nf,h=256,depth=4,drop=0.1,act='gelu'):
        super().__init__()
        A=nn.GELU if act=='gelu' else (nn.SiLU if act=='silu' else nn.ReLU)
        ls=[nn.Linear(nf,h),nn.BatchNorm1d(h),A(),nn.Dropout(drop)]
        for _ in range(depth-1): ls+=[nn.Linear(h,h),nn.BatchNorm1d(h),A(),nn.Dropout(drop)]
        ls.append(nn.Linear(h,1)); self.net=nn.Sequential(*ls)
    def forward(self,x): return self.net(x).squeeze(1)

# 2. MLP Residual
class ResidualMLP(nn.Module):
    def __init__(self,nf,h=256,n_blocks=4,drop=0.1):
        super().__init__()
        self.proj=nn.Linear(nf,h); self.bn0=nn.BatchNorm1d(h)
        self.blocks=nn.ModuleList([nn.Sequential(
            nn.Linear(h,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop),
            nn.Linear(h,h),nn.BatchNorm1d(h)) for _ in range(n_blocks)])
        self.head=nn.Linear(h,1)
    def forward(self,x):
        x=F.gelu(self.bn0(self.proj(x)))
        for b in self.blocks: x=F.gelu(x+b(x))
        return self.head(x).squeeze(1)

# 3. MLP Gated (GLU)
class GatedMLP(nn.Module):
    def __init__(self,nf,h=256,n_layers=4,drop=0.1):
        super().__init__()
        self.inp=nn.Linear(nf,h)
        self.gts=nn.ModuleList([nn.Linear(h,h*2) for _ in range(n_layers)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(h) for _ in range(n_layers)])
        self.drs=nn.ModuleList([nn.Dropout(drop) for _ in range(n_layers)])
        self.head=nn.Linear(h,1)
    def forward(self,x):
        x=F.gelu(self.inp(x))
        for g,b,d in zip(self.gts,self.bns,self.drs): x=d(b(F.glu(g(x),dim=-1)))
        return self.head(x).squeeze(1)

# 4. Highway Network (gated skip: y = T*H(x) + (1-T)*x)
class HighwayMLP(nn.Module):
    def __init__(self,nf,h=256,n_layers=4,drop=0.1):
        super().__init__()
        self.proj=nn.Linear(nf,h)
        self.H=nn.ModuleList([nn.Linear(h,h) for _ in range(n_layers)])
        self.T=nn.ModuleList([nn.Linear(h,h) for _ in range(n_layers)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(h) for _ in range(n_layers)])
        self.drs=nn.ModuleList([nn.Dropout(drop) for _ in range(n_layers)])
        self.head=nn.Linear(h,1)
        for t in self.T: nn.init.constant_(t.bias,-1)
    def forward(self,x):
        x=F.relu(self.proj(x))
        for H,T,bn,dr in zip(self.H,self.T,self.bns,self.drs):
            t=torch.sigmoid(T(x)); x=dr(bn(t*F.relu(H(x))+(1-t)*x))
        return self.head(x).squeeze(1)

# 5. DenseNet MLP (concat de todas las capas anteriores)
class DenseMLP(nn.Module):
    def __init__(self,nf,h=128,growth=64,n_layers=4,drop=0.1):
        super().__init__()
        self.proj=nn.Linear(nf,h)
        self.layers=nn.ModuleList()
        self.bns=nn.ModuleList()
        in_dim=h
        for _ in range(n_layers):
            self.layers.append(nn.Linear(in_dim,growth))
            self.bns.append(nn.BatchNorm1d(growth))
            in_dim+=growth
        self.drop=nn.Dropout(drop)
        self.head=nn.Linear(in_dim,1)
    def forward(self,x):
        x=F.gelu(self.proj(x))
        for L,bn in zip(self.layers,self.bns):
            out=self.drop(F.gelu(bn(L(x)))); x=torch.cat([x,out],dim=1)
        return self.head(x).squeeze(1)

# 6. SwishMLP con LayerNorm
class SwishMLP(nn.Module):
    def __init__(self,nf,h=256,depth=4,drop=0.1):
        super().__init__()
        self.proj=nn.Linear(nf,h)
        self.blocks=nn.ModuleList([nn.Sequential(
            nn.LayerNorm(h),nn.Linear(h,h*2),nn.SiLU(),nn.Dropout(drop),
            nn.Linear(h*2,h)) for _ in range(depth)])
        self.head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,1))
    def forward(self,x):
        x=F.silu(self.proj(x))
        for b in self.blocks: x=x+b(x)
        return self.head(x).squeeze(1)

# 7. Mixture of Experts
class MoEMLP(nn.Module):
    def __init__(self,nf,n_exp=8,h=128,drop=0.1):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(nf,n_exp),nn.Softmax(dim=-1))
        self.experts=nn.ModuleList([nn.Sequential(
            nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop),
            nn.Linear(h,h//2),nn.GELU(),nn.Linear(h//2,1)) for _ in range(n_exp)])
    def forward(self,x):
        g=self.gate(x)
        outs=torch.stack([e(x).squeeze(1) for e in self.experts],dim=1)
        return (g*outs).sum(dim=1)

# 8. AutoInt
class AutoInt(nn.Module):
    def __init__(self,nf,emb=32,heads=4,layers=3,hdim=128,drop=0.1):
        super().__init__()
        self.embs=nn.ModuleList([nn.Linear(1,emb) for _ in range(nf)])
        self.n0=nn.LayerNorm(emb)
        enc=nn.TransformerEncoderLayer(d_model=emb,nhead=heads,dim_feedforward=emb*4,
            dropout=drop,batch_first=True,norm_first=True,activation='gelu')
        self.tr=nn.TransformerEncoder(enc,num_layers=layers)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(nf*emb,hdim),nn.GELU(),nn.Dropout(drop),
            nn.Linear(hdim,hdim//2),nn.GELU(),nn.Linear(hdim//2,1))
        self.skip=nn.Linear(nf*emb,1)
    def forward(self,x):
        B=x.shape[0]
        tk=torch.stack([self.embs[i](x[:,i:i+1]) for i in range(x.shape[1])],dim=1)
        out=self.tr(self.n0(tk)).reshape(B,-1)
        return (self.head(out)+self.skip(out)).squeeze(1)

# 9. ResAutoInt (AutoInt con skip fuerte — ganador del mega_search)
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

# 10. FT-Transformer (CLS token)
class FTTransformer(nn.Module):
    def __init__(self,nf,emb=64,heads=8,layers=3,drop=0.1,hdrop=0.1):
        super().__init__()
        self.cls=nn.Parameter(torch.zeros(1,1,emb)); nn.init.trunc_normal_(self.cls,std=0.02)
        self.fembs=nn.ModuleList([nn.Linear(1,emb) for _ in range(nf)])
        self.fn=nn.LayerNorm(emb)
        enc=nn.TransformerEncoderLayer(d_model=emb,nhead=heads,dim_feedforward=emb*4,
            dropout=drop,batch_first=True,norm_first=True,activation='gelu')
        self.tr=nn.TransformerEncoder(enc,num_layers=layers)
        self.head=nn.Sequential(nn.LayerNorm(emb),nn.Dropout(hdrop),
            nn.Linear(emb,emb//2),nn.GELU(),nn.Linear(emb//2,1))
    def forward(self,x):
        B=x.shape[0]
        tk=torch.stack([self.fembs[i](x[:,i:i+1]) for i in range(x.shape[1])],dim=1)
        tk=self.fn(tk); cls=self.cls.expand(B,-1,-1)
        out=self.tr(torch.cat([cls,tk],dim=1))
        return self.head(out[:,0]).squeeze(1)

# 11. SAINT (column-attention + row-attention)
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

# 12. DCN-v2
class _CrossLayer(nn.Module):
    def __init__(self,dim,rank):
        super().__init__()
        self.U=nn.Linear(dim,rank,bias=False); self.V=nn.Linear(dim,rank,bias=False)
        self.b=nn.Parameter(torch.zeros(dim)); self.n=nn.LayerNorm(dim)
    def forward(self,x0,x): return self.n(x0*(self.U(x)*self.V(x0)).sum(-1,keepdim=True)+self.b+x)

class DCNv2(nn.Module):
    def __init__(self,nf,n_cross=3,h=(256,128),drop=0.1):
        super().__init__()
        rank=max(nf//4,4)
        self.cross=nn.ModuleList([_CrossLayer(nf,rank) for _ in range(n_cross)])
        ls=[]; d=nf
        for hi in h: ls+=[nn.Linear(d,hi),nn.BatchNorm1d(hi),nn.GELU(),nn.Dropout(drop)]; d=hi
        self.deep=nn.Sequential(*ls); fd=nf+d
        self.head=nn.Sequential(nn.Linear(fd,fd//2),nn.GELU(),nn.Dropout(drop),nn.Linear(fd//2,1))
    def forward(self,x):
        x0=x; xc=x
        for L in self.cross: xc=L(x0,xc)
        return self.head(torch.cat([xc,self.deep(x)],dim=1)).squeeze(1)

# 13. Wide & Deep
class WideDeep(nn.Module):
    def __init__(self,nf,h=256,depth=3,drop=0.1):
        super().__init__()
        self.wide=nn.Linear(nf,1)
        ls=[nn.Linear(nf,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop)]
        for _ in range(depth-1): ls+=[nn.Linear(h,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop)]
        ls.append(nn.Linear(h,1)); self.deep=nn.Sequential(*ls)
    def forward(self,x): return (self.wide(x)+self.deep(x)).squeeze(1)

# 14. DeepFM (Factorization Machine + Deep)
class DeepFM(nn.Module):
    def __init__(self,nf,emb=8,h=128,depth=2,drop=0.1):
        super().__init__()
        self.embs=nn.ModuleList([nn.Linear(1,emb) for _ in range(nf)])
        self.linear=nn.Linear(nf,1)
        ls=[nn.Linear(nf*emb,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop)]
        for _ in range(depth-1): ls+=[nn.Linear(h,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop)]
        ls.append(nn.Linear(h,1)); self.deep=nn.Sequential(*ls)
    def forward(self,x):
        B=x.shape[0]
        E=torch.stack([self.embs[i](x[:,i:i+1]) for i in range(x.shape[1])],dim=1)
        se=E.sum(1); fm=0.5*((se**2)-(E**2).sum(1)).sum(1,keepdim=True)
        return (fm+self.deep(E.reshape(B,-1))+self.linear(x)).squeeze(1)

# 15. MLP-Mixer (token-mixing + channel-mixing, sin attention)
class _MixerBlock(nn.Module):
    def __init__(self,nf,emb,exp=2,drop=0.1):
        super().__init__()
        self.tn=nn.LayerNorm(nf)
        self.tmix=nn.Sequential(nn.Linear(nf,nf*exp),nn.GELU(),nn.Dropout(drop),nn.Linear(nf*exp,nf))
        self.cn=nn.LayerNorm(emb)
        self.cmix=nn.Sequential(nn.Linear(emb,emb*exp),nn.GELU(),nn.Dropout(drop),nn.Linear(emb*exp,emb))
    def forward(self,x):
        xt=x.transpose(1,2); xt=xt+self.tmix(self.tn(xt)); x=xt.transpose(1,2)
        return x+self.cmix(self.cn(x))

class MLPMixer(nn.Module):
    def __init__(self,nf,emb=32,n_layers=4,exp=2,drop=0.1):
        super().__init__()
        self.proj=nn.ModuleList([nn.Linear(1,emb) for _ in range(nf)]); self.n0=nn.LayerNorm(emb)
        self.blocks=nn.ModuleList([_MixerBlock(nf,emb,exp,drop) for _ in range(n_layers)])
        flat=nf*emb
        self.head=nn.Sequential(nn.Flatten(),nn.LayerNorm(flat),nn.Linear(flat,emb*4),
            nn.GELU(),nn.Dropout(drop),nn.Linear(emb*4,1))
    def forward(self,x):
        tk=torch.stack([self.proj[i](x[:,i:i+1]) for i in range(x.shape[1])],dim=1); tk=self.n0(tk)
        for b in self.blocks: tk=b(tk)
        return self.head(tk).squeeze(1)

# 16. AutoInt+ (AutoInt + Cross Network combinados)
class AutoIntPlus(nn.Module):
    def __init__(self,nf,emb=32,heads=4,layers=3,hdim=128,n_cross=2,drop=0.1):
        super().__init__()
        self.embs=nn.ModuleList([nn.Linear(1,emb) for _ in range(nf)]); self.n0=nn.LayerNorm(emb)
        enc=nn.TransformerEncoderLayer(d_model=emb,nhead=heads,dim_feedforward=emb*4,
            dropout=drop,batch_first=True,norm_first=True,activation='gelu')
        self.tr=nn.TransformerEncoder(enc,num_layers=layers)
        rank=max(nf//4,4)
        self.cross=nn.ModuleList([_CrossLayer(nf,rank) for _ in range(n_cross)])
        combined=nf*emb+nf
        self.head=nn.Sequential(nn.Linear(combined,hdim),nn.GELU(),nn.Dropout(drop),
            nn.Linear(hdim,hdim//2),nn.GELU(),nn.Linear(hdim//2,1))
    def forward(self,x):
        B=x.shape[0]
        tk=torch.stack([self.embs[i](x[:,i:i+1]) for i in range(x.shape[1])],dim=1)
        att_out=self.tr(self.n0(tk)).reshape(B,-1)
        x0=x; xc=x
        for L in self.cross: xc=L(x0,xc)
        return self.head(torch.cat([att_out,xc],dim=1)).squeeze(1)

# 17. Bilinear Net (captura interacciones cuadraticas entre features)
class BilinearNet(nn.Module):
    def __init__(self,nf,rank=16,h=128,depth=2,drop=0.1):
        super().__init__()
        self.U=nn.Parameter(torch.randn(nf,rank)*0.01); self.V=nn.Parameter(torch.randn(nf,rank)*0.01)
        self.bn_bil=nn.BatchNorm1d(1)
        ls=[nn.Linear(nf+rank+1,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop)]
        for _ in range(depth-1): ls+=[nn.Linear(h,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop)]
        ls.append(nn.Linear(h,1)); self.mlp=nn.Sequential(*ls)
    def forward(self,x):
        xu=(x@self.U); xv=(x@self.V)
        bil=self.bn_bil((xu*xv).sum(1,keepdim=True))
        return self.mlp(torch.cat([x,xu,bil],dim=1)).squeeze(1)

# 18. NODE — Neural Oblivious Decision Ensembles
class _ObliviousTree(nn.Module):
    def __init__(self,nf,depth=4):
        super().__init__()
        self.depth=depth
        self.sw=nn.Parameter(torch.zeros(depth,nf)); nn.init.normal_(self.sw,std=0.1/nf**0.5)
        self.sb=nn.Parameter(torch.zeros(depth))
        self.lv=nn.Parameter(torch.zeros(2**depth))
    def forward(self,x):
        B=x.shape[0]
        sp=torch.sigmoid((x@self.sw.T)+self.sb)  # (B, depth)
        lp=torch.ones(B,1,device=x.device)
        for d in range(self.depth):
            gr=sp[:,d:d+1]; gl=1-gr
            lp=torch.cat([lp*gl,lp*gr],dim=1)
        return (lp*self.lv).sum(1)

class NODE(nn.Module):
    def __init__(self,nf,n_trees=64,depth=4):
        super().__init__()
        self.bn=nn.BatchNorm1d(nf)
        self.trees=nn.ModuleList([_ObliviousTree(nf,depth) for _ in range(n_trees)])
    def forward(self,x):
        x=self.bn(x)
        return torch.stack([t(x) for t in self.trees],dim=1).mean(1)

# 19. TabNet simplificado
class TabNet(nn.Module):
    def __init__(self,nf,nd=32,na=32,n_steps=5,gamma=1.3):
        super().__init__()
        self.bn=nn.BatchNorm1d(nf); self.ns=n_steps; self.gamma=gamma; self.nd=nd; self.na=na
        self.shared=nn.Linear(nf,nd+na)
        self.step_h=nn.ModuleList([nn.Linear(nf,nd+na) for _ in range(n_steps)])
        self.step_a=nn.ModuleList([nn.Linear(na,nf) for _ in range(n_steps)])
        self.bn_sh=nn.BatchNorm1d(nd+na)
        self.bn_st=nn.ModuleList([nn.BatchNorm1d(nd+na) for _ in range(n_steps)])
        self.bn_at=nn.ModuleList([nn.BatchNorm1d(nf) for _ in range(n_steps)])
        self.head=nn.Linear(nd,1)
    def forward(self,x):
        B=x.shape[0]; xb=self.bn(x)
        prior=torch.ones(B,x.shape[1],device=x.device)
        agg=torch.zeros(B,self.nd,device=x.device)
        for i in range(self.ns):
            shared=F.glu(self.bn_sh(self.shared(xb)),dim=-1)
            step  =F.glu(self.bn_st[i](self.step_h[i](xb)),dim=-1)
            m=(shared+step)*math.sqrt(0.5)
            nd2=m.shape[1]//2
            att=F.softmax(self.bn_at[i](self.step_a[i](m[:,nd2:]))*prior,dim=-1)
            prior=prior*(self.gamma-att)
            agg=agg+F.relu(m[:,:nd2])
        return self.head(agg).squeeze(1)

# 20. xDeepFM — Compressed Interaction Network + Deep
class _CINLayer(nn.Module):
    def __init__(self,n_h,n_x0,n_out):
        super().__init__()
        self.conv=nn.Conv1d(n_h*n_x0,n_out,kernel_size=1); self.bn=nn.BatchNorm1d(n_out)
    def forward(self,xk,x0):
        B=xk.shape[0]
        outer=torch.einsum('bpe,bqe->bpqe',xk,x0).reshape(B,-1,xk.shape[-1])
        return F.relu(self.bn(self.conv(outer)))

class xDeepFM(nn.Module):
    def __init__(self,nf,emb=8,cin=(16,16),h=128,depth=2,drop=0.1):
        super().__init__()
        self.embs=nn.ModuleList([nn.Linear(1,emb) for _ in range(nf)]); self.n0=nn.LayerNorm(emb)
        dims=[nf]+list(cin)
        self.cin=nn.ModuleList([_CINLayer(dims[i],nf,cin[i]) for i in range(len(cin))])
        ls=[nn.Linear(nf*emb,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop)]
        for _ in range(depth-1): ls+=[nn.Linear(h,h),nn.BatchNorm1d(h),nn.GELU(),nn.Dropout(drop)]
        self.deep=nn.Sequential(*ls)
        self.head=nn.Linear(sum(cin)+h,1)
    def forward(self,x):
        B=x.shape[0]
        tk=torch.stack([self.embs[i](x[:,i:i+1]) for i in range(x.shape[1])],dim=1); tk=self.n0(tk)
        x0=tk; xk=tk; cout=[]
        for L in self.cin: xk=L(xk,x0); cout.append(xk.sum(-1))
        cin_out=torch.cat(cout,dim=1)
        deep_out=self.deep(tk.reshape(B,-1))
        return self.head(torch.cat([cin_out,deep_out],dim=1)).squeeze(1)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
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

def ks_stat(yt, yp):
    return ks_statistic(yt, yp)

def train_model(model,Xtr,ytr,Xte,yte,Xoo,yoo,
                lr=1e-3,wd=1e-4,batch=256,epochs=350,patience=50,
                mxup=0.3,swa_f=0.65,gamma=1.5,ls=0.02):
    model=model.to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=lr/50)
    scaler=GradScaler(DEVICE) if DEVICE == 'cuda' else None
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
            if DEVICE == 'cuda':
                with autocast('cuda'): loss=focal_bce(model(Xb),yb,gamma,ls)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
            else:
                loss=focal_bce(model(Xb),yb,gamma,ls)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
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

# ══════════════════════════════════════════════════════════════════════════════
# REGISTRO
# ══════════════════════════════════════════════════════════════════════════════
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
    atr=roc_auc_score(ytr,p_tr); ate=roc_auc_score(yte,p_te); aoo=roc_auc_score(yoo,p_oo)
    ks=ks_stat(yoo,p_oo); gap=atr-aoo
    # Evaluacion completa con threshold optimo (calculado en train)
    thr = threshold_by_ks(ytr, p_tr)
    eval_tr  = evaluate_binary_model(ytr, p_tr, thr)
    eval_te  = evaluate_binary_model(yte, p_te, thr)
    eval_oos = evaluate_binary_model(yoo, p_oo, thr)
    psi_te   = population_stability_index(p_tr, p_te)
    psi_oos  = population_stability_index(p_tr, p_oo)
    beat='  *** SUPERA RL ***' if aoo>RL_BASELINE else ''
    elapsed=f"{(time.time()-t0)/3600:.1f}h"
    print(f"  [{elapsed}] {name:<52}  tr={atr:.4f}  te={ate:.4f}  oos={aoo:.4f}  ks={ks:.4f}  gap={gap:+.4f}  psi_te={psi_te:.3f}  psi_oos={psi_oos:.3f}{beat}")
    r={'name':name,'oot_auc':round(aoo,4),'test_auc':round(ate,4),'train_auc':round(atr,4),
       'ks':round(ks,4),'gap':round(gap,4),
       'gini_tr':round(eval_tr['gini'],4),'gini_te':round(eval_te['gini'],4),'gini_oos':round(eval_oos['gini'],4),
       'psi_te':round(psi_te,4),'psi_oos':round(psi_oos,4),
       'threshold':round(thr,6),
       'p_tr':p_tr.tolist(),'p_te':p_te.tolist(),'p_oo':p_oo.tolist()}
    r = {k: float(v) if isinstance(v, (np.float32, np.float64, np.number)) else v for k, v in r.items()}
    all_results.append(r); save(); return r

# ── Runner de bloques ──────────────────────────────────────────────────────────
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
        except Exception as e:
            print(f"Exception during trial: {e}")
            import traceback
            traceback.print_exc()
            raise optuna.exceptions.TrialPruned()

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

# ══════════════════════════════════════════════════════════════════════════════
# ESPACIOS DE HIPERPARAMETROS
# ══════════════════════════════════════════════════════════════════════════════
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

def sp_res(trial,nf):
    return {**_common(trial),'nf':nf,
            'h':trial.suggest_categorical('h',[64,128,256,512]),
            'n_blocks':trial.suggest_int('n_blocks',2,10),
            'drop':trial.suggest_float('drop',0.0,0.35)}

def sp_gated(trial,nf):
    return {**_common(trial),'nf':nf,
            'h':trial.suggest_categorical('h',[64,128,256,512]),
            'n_layers':trial.suggest_int('n_layers',2,10),
            'drop':trial.suggest_float('drop',0.0,0.35)}

def sp_highway(trial,nf):
    return {**_common(trial),'nf':nf,
            'h':trial.suggest_categorical('h',[64,128,256,512]),
            'n_layers':trial.suggest_int('n_layers',2,10),
            'drop':trial.suggest_float('drop',0.0,0.35)}

def sp_dense(trial,nf):
    return {**_common(trial),'nf':nf,
            'h':trial.suggest_categorical('h',[64,128,256]),
            'growth':trial.suggest_categorical('growth',[32,64,128]),
            'n_layers':trial.suggest_int('n_layers',2,8),
            'drop':trial.suggest_float('drop',0.0,0.35)}

def sp_swish(trial,nf):
    return {**_common(trial),'nf':nf,
            'h':trial.suggest_categorical('h',[64,128,256,512]),
            'depth':trial.suggest_int('depth',2,10),
            'drop':trial.suggest_float('drop',0.0,0.35)}

def sp_moe(trial,nf):
    return {**_common(trial),'nf':nf,
            'n_exp':trial.suggest_int('n_exp',4,16),
            'h':trial.suggest_categorical('h',[64,128,256]),
            'drop':trial.suggest_float('drop',0.0,0.35)}

def sp_ai(trial,nf):
    emb=trial.suggest_categorical('emb',[16,32,64,128])
    heads=trial.suggest_categorical('heads',[2,4,8])
    if emb%heads!=0: return None
    return {**_common(trial),'nf':nf,'emb':emb,'heads':heads,
            'layers':trial.suggest_int('layers',2,6),
            'hdim':trial.suggest_categorical('hdim',[64,128,256,512]),
            'drop':trial.suggest_float('drop',0.0,0.30)}

def sp_resai(trial,nf):
    emb=trial.suggest_categorical('emb',[16,32,64])
    heads=trial.suggest_categorical('heads',[2,4,8])
    if emb%heads!=0: return None
    return {**_common(trial),'nf':nf,'emb':emb,'heads':heads,
            'layers':trial.suggest_int('layers',2,5),
            'mlph':trial.suggest_categorical('mlph',[128,256,512,1024]),
            'drop':trial.suggest_float('drop',0.0,0.20)}

def sp_ft(trial,nf):
    emb=trial.suggest_categorical('emb',[32,64,128,192])
    heads=trial.suggest_categorical('heads',[2,4,8,12])
    if emb%heads!=0: return None
    return {**_common(trial),'nf':nf,'emb':emb,'heads':heads,
            'layers':trial.suggest_int('layers',2,6),
            'drop':trial.suggest_float('drop',0.0,0.25),
            'hdrop':trial.suggest_float('hdrop',0.0,0.25)}

def sp_saint(trial,nf):
    emb=trial.suggest_categorical('emb',[16,32,64])
    hc=trial.suggest_categorical('hc',[2,4,8])
    hr=trial.suggest_categorical('hr',[1,2,4])
    if emb%hc!=0: return None
    return {**_common(trial),'nf':nf,'emb':emb,
            'layers':trial.suggest_int('layers',1,4),'hc':hc,'hr':hr,
            'hdim':trial.suggest_categorical('hdim',[64,128,256]),
            'drop':trial.suggest_float('drop',0.0,0.25)}

def sp_dcn(trial,nf):
    h1=trial.suggest_categorical('h1',[64,128,256,512])
    h2=trial.suggest_categorical('h2',[32,64,128,256])
    return {**_common(trial),'nf':nf,
            'n_cross':trial.suggest_int('n_cross',2,5),'h':(h1,h2),
            'drop':trial.suggest_float('drop',0.0,0.25)}

def sp_wd(trial,nf):
    return {**_common(trial),'nf':nf,
            'h':trial.suggest_categorical('h',[64,128,256,512]),
            'depth':trial.suggest_int('depth',2,8),
            'drop':trial.suggest_float('drop',0.0,0.35)}

def sp_dfm(trial,nf):
    return {**_common(trial),'nf':nf,
            'emb':trial.suggest_categorical('emb',[4,8,16,32]),
            'h':trial.suggest_categorical('h',[64,128,256]),
            'depth':trial.suggest_int('depth',1,4),
            'drop':trial.suggest_float('drop',0.0,0.30)}

def sp_mixer(trial,nf):
    emb=trial.suggest_categorical('emb',[16,32,64])
    return {**_common(trial),'nf':nf,'emb':emb,
            'n_layers':trial.suggest_int('n_layers',2,6),
            'exp':trial.suggest_categorical('exp',[2,4]),
            'drop':trial.suggest_float('drop',0.0,0.25)}

def sp_aip(trial,nf):
    emb=trial.suggest_categorical('emb',[16,32,64])
    heads=trial.suggest_categorical('heads',[2,4,8])
    if emb%heads!=0: return None
    return {**_common(trial),'nf':nf,'emb':emb,'heads':heads,
            'layers':trial.suggest_int('layers',2,5),
            'hdim':trial.suggest_categorical('hdim',[64,128,256]),
            'n_cross':trial.suggest_int('n_cross',1,4),
            'drop':trial.suggest_float('drop',0.0,0.25)}

def sp_bilin(trial,nf):
    return {**_common(trial),'nf':nf,
            'rank':trial.suggest_categorical('rank',[8,16,32,64]),
            'h':trial.suggest_categorical('h',[64,128,256]),
            'depth':trial.suggest_int('depth',1,4),
            'drop':trial.suggest_float('drop',0.0,0.30)}

def sp_node(trial,nf):
    return {**_common(trial),'nf':nf,
            'n_trees':trial.suggest_categorical('n_trees',[32,64,128,256]),
            'depth':trial.suggest_int('depth',3,6)}

def sp_tabnet(trial,nf):
    nd=trial.suggest_categorical('nd',[16,32,64])
    return {**_common(trial),'nf':nf,'nd':nd,'na':nd,
            'n_steps':trial.suggest_int('n_steps',3,8),
            'gamma':trial.suggest_float('gamma',1.0,2.0)}

def sp_xdfm(trial,nf):
    c1=trial.suggest_categorical('c1',[8,16,32]); c2=trial.suggest_categorical('c2',[8,16,32])
    return {**_common(trial),'nf':nf,
            'emb':trial.suggest_categorical('emb',[4,8,16]),
            'cin':(c1,c2),
            'h':trial.suggest_categorical('h',[64,128,256]),
            'depth':trial.suggest_int('depth',1,3),
            'drop':trial.suggest_float('drop',0.0,0.25)}

# factories
mk={
    'SimpleMLP' :lambda **p: SimpleMLP(p['nf'],p['h'],p['depth'],p['drop'],p.get('act','gelu')),
    'ResidualMLP':lambda **p: ResidualMLP(p['nf'],p['h'],p['n_blocks'],p['drop']),
    'GatedMLP'  :lambda **p: GatedMLP(p['nf'],p['h'],p['n_layers'],p['drop']),
    'HighwayMLP':lambda **p: HighwayMLP(p['nf'],p['h'],p['n_layers'],p['drop']),
    'DenseMLP'  :lambda **p: DenseMLP(p['nf'],p['h'],p['growth'],p['n_layers'],p['drop']),
    'SwishMLP'  :lambda **p: SwishMLP(p['nf'],p['h'],p['depth'],p['drop']),
    'MoEMLP'    :lambda **p: MoEMLP(p['nf'],p['n_exp'],p['h'],p['drop']),
    'AutoInt'   :lambda **p: AutoInt(p['nf'],p['emb'],p['heads'],p['layers'],p['hdim'],p['drop']),
    'ResAutoInt':lambda **p: ResAutoInt(p['nf'],p['emb'],p['heads'],p['layers'],p['mlph'],p['drop']),
    'FTTransformer':lambda **p: FTTransformer(p['nf'],p['emb'],p['heads'],p['layers'],p['drop'],p.get('hdrop',0.1)),
    'SAINT'     :lambda **p: SAINT(p['nf'],p['emb'],p['layers'],p['hc'],p['hr'],p['hdim'],p['drop']),
    'DCNv2'     :lambda **p: DCNv2(p['nf'],p['n_cross'],p.get('h',(256,128)),p['drop']),
    'WideDeep'  :lambda **p: WideDeep(p['nf'],p['h'],p['depth'],p['drop']),
    'DeepFM'    :lambda **p: DeepFM(p['nf'],p['emb'],p['h'],p['depth'],p['drop']),
    'MLPMixer'  :lambda **p: MLPMixer(p['nf'],p['emb'],p['n_layers'],p['exp'],p['drop']),
    'AutoIntPlus':lambda **p: AutoIntPlus(p['nf'],p['emb'],p['heads'],p['layers'],p['hdim'],p['n_cross'],p['drop']),
    'BilinearNet':lambda **p: BilinearNet(p['nf'],p['rank'],p['h'],p['depth'],p['drop']),
    'NODE'      :lambda **p: NODE(p['nf'],p['n_trees'],p['depth']),
    'TabNet'    :lambda **p: TabNet(p['nf'],p['nd'],p['na'],p['n_steps'],p.get('gamma',1.3)),
    'xDeepFM'   :lambda **p: xDeepFM(p['nf'],p['emb'],p['cin'],p['h'],p['depth'],p['drop']),
}
spaces={
    'SimpleMLP':sp_simple,'ResidualMLP':sp_res,'GatedMLP':sp_gated,'HighwayMLP':sp_highway,
    'DenseMLP':sp_dense,'SwishMLP':sp_swish,'MoEMLP':sp_moe,'AutoInt':sp_ai,
    'ResAutoInt':sp_resai,'FTTransformer':sp_ft,'SAINT':sp_saint,'DCNv2':sp_dcn,
    'WideDeep':sp_wd,'DeepFM':sp_dfm,'MLPMixer':sp_mixer,'AutoIntPlus':sp_aip,
    'BilinearNet':sp_bilin,'NODE':sp_node,'TabNet':sp_tabnet,'xDeepFM':sp_xdfm,
}

# ══════════════════════════════════════════════════════════════════════════════
# BLOQUES DE EXPERIMENTOS  (20 arq × 2 variantes de features = 40 bloques)
# ══════════════════════════════════════════════════════════════════════════════
ARCHS=list(mk.keys())
EXPERIMENTS=[(f'{a} / WOE',mk[a],Xtr_r,Xte_r,Xoo_r,spaces[a]) for a in ARCHS] + \
            [(f'{a} / FE', mk[a],Xtr_f,Xte_f,Xoo_f,spaces[a]) for a in ARCHS]

print(f"\n{'='*75}")
print(f"INICIANDO {len(EXPERIMENTS)} BLOQUES  ({len(ARCHS)} arquitecturas × 2 variantes de features)")
if RUN_ONLY:
    EXPERIMENTS = [(n,mf,Xt,Xv,Xo,sp) for (n,mf,Xt,Xv,Xo,sp) in EXPERIMENTS if n in RUN_ONLY]
    print(f"FILTRO RUN_ONLY activo: {len(EXPERIMENTS)} bloques seleccionados")
    for n,*_ in EXPERIMENTS: print(f"  • {n}")
print(f"{'='*75}")

trained={}
for name,mk_fn,Xtr_b,Xte_b,Xoo_b,sp in EXPERIMENTS:
    res=run_block(name,mk_fn,Xtr_b,Xte_b,Xoo_b,sp)
    if res: trained[name]=res

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDACION — top 5 modelos
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*75}"); print("CROSS-VALIDACION 5-FOLD — TOP 5"); print(f"{'='*75}")
top5=[r for r in all_results if 'p_oo' in r and 'CV_' not in r['name']]
top5=sorted(top5,key=lambda x:-x.get('oot_auc',0))[:5]

for r in top5:
    nm=r['name']; cvnm=f'CV_{nm}'
    if cvnm in done_names: print(f"  [SKIP] {cvnm}"); continue
    params=r.get('params',{})
    mk_fn=None
    for bn,mf,Xtr_b,Xte_b,Xoo_b,sp in EXPERIMENTS:
        if bn==nm: mk_fn=mf; Xcv_raw=Xtr_b; use_fe='FE' in nm; break
    if not mk_fn or not params: continue
    print(f"\n  CV: {nm}")
    skf=StratifiedKFold(n_splits=CV_FOLDS,shuffle=True,random_state=SEED)
    X_cv=build_X(X_tmp,use_fe); oof=np.zeros(len(y_tmp))
    for fold,(itr,iva) in enumerate(skf.split(X_cv,y_tmp.values),1):
        m=mk_fn(**params)
        m=train_model(m,X_cv[itr],y_tmp.values[itr].astype('float32'),
                      X_cv[iva],y_tmp.values[iva].astype('float32'),
                      Xoo_r,yoo,lr=params.get('lr',1e-3),wd=params.get('wd',1e-4),
                      batch=params.get('batch',256),epochs=400,patience=60,
                      mxup=params.get('mxup',0.3),swa_f=params.get('swa_f',0.65))
        oof[iva]=get_probs(m,X_cv[iva])
        print(f"    Fold {fold}: {roc_auc_score(y_tmp.values[iva],oof[iva]):.4f}")
    oof_auc=roc_auc_score(y_tmp.values,oof)
    print(f"  OOF AUC={oof_auc:.4f}")
    cv_r={'name':cvnm,'oof_auc':round(oof_auc,4),'oot_auc':r['oot_auc']}
    all_results.append(cv_r); save(); done_names.add(cvnm)

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE FINAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*75}"); print("ENSEMBLE — combinando mejores modelos"); print(f"{'='*75}")
wp=[r for r in all_results if 'p_oo' in r and 'CV_' not in r['name']]
if len(wp)>=2:
    top8=sorted(wp,key=lambda x:-x.get('oot_auc',0))[:8]
    pte=[np.array(r['p_te']) for r in top8]; poo=[np.array(r['p_oo']) for r in top8]
    nms=[r['name'] for r in top8]

    p_avg=np.mean(poo,axis=0); a_avg=roc_auc_score(yoo,p_avg)
    beat='  *** SUPERA RL ***' if a_avg>RL_BASELINE else ''
    print(f"  Promedio simple top-{len(top8)}: OOT={a_avg:.4f}{beat}")

    def neg_a(w): w=np.abs(w)/np.abs(w).sum(); return -roc_auc_score(yte,sum(w[i]*pte[i] for i in range(len(pte))))
    res=minimize(neg_a,np.ones(len(pte))/len(pte),method='Nelder-Mead',options={'maxiter':3000})
    wo=np.abs(res.x)/np.abs(res.x).sum()
    p_opt=sum(wo[i]*poo[i] for i in range(len(poo))); a_opt=roc_auc_score(yoo,p_opt)
    beat='  *** SUPERA RL ***' if a_opt>RL_BASELINE else ''
    print(f"  Ensemble opt: OOT={a_opt:.4f}{beat}  pesos={[f'{nms[i][:20]}:{wo[i]:.3f}' for i in range(len(nms))]}")

    from scipy.stats import rankdata
    Xmt=np.column_stack([rankdata(p)/len(p) for p in pte]); Xmo=np.column_stack([rankdata(p)/len(p) for p in poo])
    for C in [0.01,0.1,1.0]:
        lrm=LogisticRegression(C=C,max_iter=2000,random_state=SEED).fit(Xmt,yte)
        a_lr=roc_auc_score(yoo,lrm.predict_proba(Xmo)[:,1])
        beat='  *** SUPERA RL ***' if a_lr>RL_BASELINE else ''
        print(f"  Stacking LogReg C={C}: OOT={a_lr:.4f}{beat}")
        all_results.append({'name':f'Stack_LogReg_C{C}','oot_auc':round(a_lr,4)})

    all_results.append({'name':'Ensemble_opt','oot_auc':round(a_opt,4),'avg_oot':round(a_avg,4)}); save()

# ── Reporte final ──────────────────────────────────────────────────────────────
total_h=(time.time()-t0)/3600
print(f"\n{'='*75}")
print(f"REPORTE FINAL  —  {total_h:.1f} horas")
print(f"  RL companero (referencia): {RL_BASELINE}")
print(f"{'='*75}")
sortable=[r for r in all_results if isinstance(r.get('oot_auc'),float) and 'CV_' not in r['name']]
for r in sorted(sortable,key=lambda x:-x.get('oot_auc',0))[:25]:
    beat=' ***' if r.get('oot_auc',0)>RL_BASELINE else ''
    gap=r.get('gap',float('nan')); gs=f"{gap:+.4f}" if not (isinstance(gap,float) and math.isnan(gap)) else '  N/A '
    print(f"  {r['name']:<55}  OOT={r.get('oot_auc',0):.4f}  gap={gs}{beat}")
print(f"\nResultados: {RESULTS_FILE}\nCheckpoints: {CKPT_DIR}/")
