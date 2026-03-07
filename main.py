"""
FastAPI Backend — Joint Fatigue + Activity Detection
POST /predict  →  fatigue + activity results
GET  /health   →  status check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import joblib, os, traceback

from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats  import skew, kurtosis

app = FastAPI(title="FatigueDetect API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Load models at startup ───────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

fat_rf    = fat_xgb = fat_gb = fat_sc = None
fat_feats = fat_thr = None
act_clf   = act_le  = None

@app.on_event("startup")
def load_models():
    global fat_rf, fat_xgb, fat_gb, fat_sc, fat_feats, fat_thr, act_clf, act_le
    try:
        fat_rf    = joblib.load(f"{MODEL_DIR}/fat_rf.pkl")
        fat_xgb   = joblib.load(f"{MODEL_DIR}/fat_xgb.pkl")
        fat_gb    = joblib.load(f"{MODEL_DIR}/fat_gb.pkl")
        fat_sc    = joblib.load(f"{MODEL_DIR}/fat_scaler.pkl")
        fat_feats = joblib.load(f"{MODEL_DIR}/fat_features.pkl")
        fat_thr   = joblib.load(f"{MODEL_DIR}/fat_threshold.pkl")
        act_clf   = joblib.load(f"{MODEL_DIR}/act_classifier.pkl")
        act_le    = joblib.load(f"{MODEL_DIR}/act_label_encoder.pkl")
        print("✅ All models loaded")
    except Exception as e:
        print(f"❌ Model load error: {e}")

# ── Schemas ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    ppg:        List[float] = Field(..., description="PPG signal, 1000 samples @ 50Hz (20s)")
    left_acc:   List[List[float]] = Field(..., description="Left ACC  200×3 @ 50Hz (4s)")
    left_gyro:  List[List[float]] = Field(..., description="Left GYRO 200×3 @ 50Hz (4s)")
    right_acc:  List[List[float]] = Field(..., description="Right ACC  200×3 @ 50Hz (4s)")
    right_gyro: List[List[float]] = Field(..., description="Right GYRO 200×3 @ 50Hz (4s)")

class PredictResponse(BaseModel):
    fatigue_label:  str
    fatigue_prob:   float
    fatigue_binary: int
    activity_label: str
    activity_conf:  float
    activity_probs: Dict[str, float]
    joint_label:    str

# ── Signal helpers ───────────────────────────────────────────
FS = 50

def bandpass(sig, lo, hi, order=4):
    nyq = FS/2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def _hrv_time(ibi):
    d = np.diff(ibi)
    return (np.sqrt(np.mean(d**2)), np.std(ibi), 60/np.mean(ibi),
            np.sum(np.abs(d)>0.05)/len(d), np.mean(ibi),
            float(skew(ibi)), float(kurtosis(ibi)))

def _hrv_freq(ibi, fs=4):
    if len(ibi)<5: return 0,0,0,0,0,0
    ta = np.cumsum(ibi)-np.cumsum(ibi)[0]
    ut = np.arange(0, ta[-1], 1/fs)
    if len(ut)<5: return 0,0,0,0,0,0
    interp = np.interp(ut,ta,ibi)-np.mean(np.interp(ut,ta,ibi))
    f,pxx  = welch(interp, fs=fs, nperseg=min(256,len(interp)))
    lf = np.trapezoid(pxx[(f>=0.04)&(f<=0.15)])
    hf = np.trapezoid(pxx[(f>=0.15)&(f<=0.40)])
    tot= np.trapezoid(pxx)
    return lf,hf,(lf/hf if hf>0 else 0),tot,(lf/tot if tot>0 else 0),(hf/tot if tot>0 else 0)

def _morph(sig, peaks):
    a=sig[peaks]
    return np.mean(a),np.std(a),float(skew(a)),float(kurtosis(a)),np.var(sig),np.sum(sig**2)/len(sig)

def _imu_fat(ax,ay,az,gx,gy,gz):
    acc_mag  = np.sqrt(ax**2+ay**2+az**2)
    gyro_mag = np.sqrt(gx**2+gy**2+gz**2)
    def st(s,p):
        return {f"{p}_mean":np.mean(s), f"{p}_std":np.std(s),
                f"{p}_rms":np.sqrt(np.mean(s**2)), f"{p}_range":np.ptp(s),
                f"{p}_skew":float(skew(s)), f"{p}_kurt":float(kurtosis(s))}
    f={}
    for s,n in [(ax,"acc_x"),(ay,"acc_y"),(az,"acc_z"),(acc_mag,"acc_mag"),
                (gx,"gyro_x"),(gy,"gyro_y"),(gz,"gyro_z"),(gyro_mag,"gyro_mag")]:
        f.update(st(s,n))
    jk=np.diff(acc_mag)*FS
    f.update({"jerk_mean":np.mean(np.abs(jk)),"jerk_std":np.std(jk),
              "jerk_rms":np.sqrt(np.mean(jk**2)),
              "sma":(np.sum(np.abs(ax))+np.sum(np.abs(ay))+np.sum(np.abs(az)))/len(ax),
              "acc_corr_xy":np.corrcoef(ax,ay)[0,1],
              "acc_corr_xz":np.corrcoef(ax,az)[0,1],
              "acc_corr_yz":np.corrcoef(ay,az)[0,1]})
    for s,n in [(acc_mag,"acc"),(gyro_mag,"gyro")]:
        fr,pxx=welch(s,fs=FS,nperseg=min(256,len(s)))
        p=pxx/(np.sum(pxx)+1e-10)
        f[f"dom_freq_{n}"]=fr[np.argmax(pxx)]
        f[f"spectral_entropy_{n}"]=-np.sum(p*np.log(p+1e-10))
        f[f"low_freq_power_{n}"]=np.trapezoid(pxx[(fr>=0.1)&(fr<=2.0)])
        f[f"high_freq_power_{n}"]=np.trapezoid(pxx[(fr>=2.0)&(fr<=10.0)])
        f[f"spectral_centroid_{n}"]=np.sum(fr*pxx)/np.sum(pxx) if np.sum(pxx)>0 else 0
    return f

def _stat_b(s,N):
    sk=float(skew(s)); ku=float(kurtosis(s))
    from scipy.stats import iqr as iqr_fn
    return [np.mean(s),np.std(s),np.var(s),np.median(s),
            np.min(s),np.max(s),np.ptp(s),float(iqr_fn(s)),
            np.sum(s**2),np.sqrt(np.mean(s**2)),np.sum(np.abs(s))/N,
            np.mean(s[1:-1]**2-s[:-2]*s[2:]) if N>2 else 0.,sk,ku]

def _freq_b(s):
    freqs,psd=welch(s,fs=FS,nperseg=min(128,len(s)))
    tot=np.sum(psd)+1e-10; dom=np.argmax(psd)
    pn=psd/tot; se=-np.sum(pn*np.log(pn+1e-10))
    mf=np.sum(freqs*psd)/tot
    bw=np.sqrt(np.sum(((freqs-mf)**2)*psd)/tot)
    qf=freqs[dom]/bw if bw>0 else 0.
    hp=psd[np.argmin(np.abs(freqs-2*freqs[dom]))]
    return [tot,freqs[dom],psd[dom],se,bw,qf,hp]

def _mag_jk(xyz):
    mag=np.sqrt(np.sum(xyz**2,axis=1))
    jk=np.sqrt(np.sum(np.diff(xyz,axis=0)**2,axis=1))
    return [np.mean(mag),np.std(mag),np.sqrt(np.mean(mag**2)),np.mean(jk),np.std(jk)]

def extract_activity_features(win12):
    N=win12.shape[0]; feats=[]
    for ch in range(12):
        feats.extend(_stat_b(win12[:,ch],N))
        feats.extend(_freq_b(win12[:,ch]))
    for idx in [(0,1,2),(3,4,5),(6,7,8),(9,10,11)]:
        feats.extend(_mag_jk(win12[:,list(idx)]))
    la=np.sqrt(np.sum(win12[:,0:3]**2,axis=1)); rа=np.sqrt(np.sum(win12[:,6:9]**2,axis=1))
    lg=np.sqrt(np.sum(win12[:,3:6]**2,axis=1)); rg=np.sqrt(np.sum(win12[:,9:12]**2,axis=1))
    def sc(a,b): c=np.corrcoef(a,b)[0,1]; return 0. if np.isnan(c) else c
    feats.extend([sc(la,rа),sc(lg,rg),np.mean(la-rа),np.std(la-rа),
                  np.mean(lg-rg),np.std(lg-rg),sc(la,lg),sc(rа,rg)])
    return np.array(feats,dtype=np.float32)

# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
def health():
    fat_ok = all(m is not None for m in [fat_rf,fat_xgb,fat_gb,fat_sc,fat_feats,fat_thr])
    act_ok = all(m is not None for m in [act_clf, act_le])
    return {
        "status": "running",
        "fatigue_model": fat_ok,
        "activity_model": act_ok,
        "models_loaded": fat_ok and act_ok,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if fat_rf is None or act_clf is None:
        raise HTTPException(503, "Models not loaded")

    try:
        ppg    = np.array(req.ppg,        dtype=np.float32)
        la     = np.array(req.left_acc,   dtype=np.float32)
        lg     = np.array(req.left_gyro,  dtype=np.float32)
        ra     = np.array(req.right_acc,  dtype=np.float32)
        rg     = np.array(req.right_gyro, dtype=np.float32)

        # ── Fatigue ──────────────────────────────────────────
        ppg_f  = bandpass(ppg, 0.5, 4)
        peaks, _ = find_peaks(ppg_f, distance=FS*0.4)
        fat_prob = 0.0

        if len(peaks) >= 10:
            ibi = np.diff(peaks/FS); ibi = ibi[(ibi>0.3)&(ibi<2)]
            if len(ibi) >= 10:
                r,sd,hr,p50,mi,isk,iku = _hrv_time(ibi)
                lf,hf,lfhf,tot,lfn,hfn = _hrv_freq(ibi)
                ma,sa,ask,aku,sv,se     = _morph(ppg_f, peaks)
                imu_d = _imu_fat(la[:,0],la[:,1],la[:,2],
                                  lg[:,0],lg[:,1],lg[:,2])
                row = {"rmssd":r,"sdnn":sd,"mean_hr":hr,"pnn50":p50,"mean_ibi":mi,
                       "ibi_skew":isk,"ibi_kurt":iku,
                       "lf_power":lf,"hf_power":hf,"lf_hf_ratio":lfhf,
                       "total_power":tot,"lf_norm":lfn,"hf_norm":hfn,
                       "mean_pulse_amp":ma,"std_pulse_amp":sa,
                       "amp_skew":ask,"amp_kurt":aku,
                       "signal_variance":sv,"signal_energy":se}
                row.update(imu_d)

                fv = np.array([row.get(f,0) for f in fat_feats]).reshape(1,-1)
                fv = fat_sc.transform(fv)
                pa = (0.40*fat_rf.predict_proba(fv) +
                      0.30*fat_xgb.predict_proba(fv) +
                      0.30*fat_gb.predict_proba(fv))
                fat_prob = float(pa[0,1])

        fat_binary = int(fat_prob >= fat_thr)
        fat_label  = "Fatigued" if fat_binary else "Not Fatigued"

        # ── Activity ─────────────────────────────────────────
        win12 = np.concatenate([la, lg, ra, rg], axis=1)
        af    = extract_activity_features(win12).reshape(1,-1)
        ap    = act_clf.predict_proba(af)[0]
        ai    = int(np.argmax(ap))
        al    = act_le.inverse_transform([ai])[0]

        return PredictResponse(
            fatigue_label  = fat_label,
            fatigue_prob   = round(fat_prob,3),
            fatigue_binary = fat_binary,
            activity_label = al,
            activity_conf  = round(float(ap[ai]),3),
            activity_probs = dict(zip(act_le.classes_, ap.round(3).tolist())),
            joint_label    = f"{fat_label} | {al.replace('_',' ').title()}",
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/")
def root():
    return {
        "api": "FatigueDetect v3",
        "endpoints": ["/health", "/predict"],
        "inputs": {
            "ppg":        "1000 floats (20s @ 50Hz)",
            "left_acc":   "200×3 floats (4s @ 50Hz)",
            "left_gyro":  "200×3 floats (4s @ 50Hz)",
            "right_acc":  "200×3 floats (4s @ 50Hz)",
            "right_gyro": "200×3 floats (4s @ 50Hz)",
        },
        "outputs": {
            "fatigue_label":  "Fatigued | Not Fatigued",
            "fatigue_prob":   "0.0 – 1.0",
            "activity_label": "ideal | beard_pulling | face_itching | hair_pulling | nail_biting",
            "joint_label":    "combined string",
        }
    }
