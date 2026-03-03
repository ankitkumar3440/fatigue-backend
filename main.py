"""
FatigueDetect API — FastAPI backend
Deploy on Render.com (free tier)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib, io, os, warnings
warnings.filterwarnings("ignore")

from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis

# ─────────────────────────────────────────────
app = FastAPI(title="FatigueDetect API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────────
# Globals – filled at startup
# ─────────────────────────────────────────────
RF = XGB = GB = SVM_MDL = SCALER = None
FEATURES  : list = []
THRESHOLD : float = 0.45
FS        = 50
WIN_S     = 10
STEP_S    = 5
WIN_N     = FS * WIN_S
STEP_N    = FS * STEP_S

@app.on_event("startup")
def load_models():
    global RF, XGB, GB, SVM_MDL, SCALER, FEATURES, THRESHOLD
    base = os.getenv("MODEL_DIR", "models")
    try:
        RF       = joblib.load(f"{base}/rf_final.pkl")
        XGB      = joblib.load(f"{base}/xgb_final.pkl")
        GB       = joblib.load(f"{base}/gb_final.pkl")
        SCALER   = joblib.load(f"{base}/scaler_final.pkl")
        FEATURES = joblib.load(f"{base}/feature_names.pkl")
        THRESHOLD= float(joblib.load(f"{base}/threshold.pkl"))
        svm_p    = f"{base}/svm_final.pkl"
        if os.path.exists(svm_p):
            SVM_MDL = joblib.load(svm_p)
        print(f"✅ Models loaded | features={len(FEATURES)} | threshold={THRESHOLD:.2f}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")

# ─────────────────────────────────────────────
# Signal processing helpers
# ─────────────────────────────────────────────

def bandpass(sig, lo=0.5, hi=4.0, order=4):
    nyq = 0.5 * FS
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def poincare_feats(ibi):
    if len(ibi) < 4: return 0., 0., 0.
    d = np.diff(ibi)
    sd1 = np.std(d) / np.sqrt(2)
    sd2 = np.sqrt(max(0, 2*np.std(ibi)**2 - 0.5*np.std(d)**2))
    return float(sd1), float(sd2), float(sd1/sd2 if sd2 > 0 else 0)

def approx_ent(rr, m=2, r=0.2):
    if len(rr) < 10: return 0.
    rv = r * np.std(rr)
    if rv == 0: return 0.
    N = len(rr)
    def phi(mv):
        t = np.array([rr[i:i+mv] for i in range(N-mv+1)])
        c = np.mean([np.sum(np.max(np.abs(t-t[i]),axis=1)<=rv)/len(t)
                     for i in range(len(t))])
        return np.log(c) if c > 0 else -10.
    try:    return float(phi(m) - phi(m+1))
    except: return 0.

def hrv_all(ibi):
    zero = {k:0. for k in ['rmssd','sdnn','mean_hr','pnn50','mean_ibi',
                             'ibi_skew','ibi_kurt','lf','hf','lf_hf',
                             'tot_pwr','lf_norm','hf_norm','sd1','sd2',
                             'sd1_sd2','apen']}
    if len(ibi) < 4: return zero
    d = np.diff(ibi)
    lf=hf=lf_hf=tot=lf_n=hf_n=0.
    try:
        ta = np.cumsum(ibi) - np.cumsum(ibi)[0]
        ut = np.arange(0, ta[-1], 0.25)
        if len(ut) >= 5:
            ip = np.interp(ut, ta, ibi) - np.mean(np.interp(ut, ta, ibi))
            f, p = welch(ip, fs=4, nperseg=min(256, len(ip)))
            lf  = float(np.trapezoid(p[(f>=.04)&(f<=.15)]))
            hf  = float(np.trapezoid(p[(f>=.15)&(f<=.40)]))
            tot = float(np.trapezoid(p))
            lf_hf = lf/hf  if hf>0  else 0.
            lf_n  = lf/tot if tot>0 else 0.
            hf_n  = hf/tot if tot>0 else 0.
    except: pass
    sd1, sd2, ratio = poincare_feats(ibi)
    return {'rmssd':float(np.sqrt(np.mean(d**2))),
            'sdnn':float(np.std(ibi)),
            'mean_hr':float(60/np.mean(ibi)),
            'pnn50':float(np.sum(np.abs(d)>.05)/len(d)),
            'mean_ibi':float(np.mean(ibi)),
            'ibi_skew':float(skew(ibi)), 'ibi_kurt':float(kurtosis(ibi)),
            'lf':lf, 'hf':hf, 'lf_hf':lf_hf, 'tot_pwr':tot,
            'lf_norm':lf_n, 'hf_norm':hf_n,
            'sd1':sd1, 'sd2':sd2, 'sd1_sd2':ratio, 'apen':approx_ent(ibi)}

def morph_all(seg, peaks):
    z = {k:0. for k in ['mean_amp','std_amp','amp_skew','amp_kurt',
                          'sig_var','sig_energy','pulse_width_mean','rise_time_mean']}
    if len(peaks) == 0: return z
    a = seg[peaks]; pw, rt = [], []
    for i in range(len(peaks)-1):
        beat = seg[peaks[i]:peaks[i+1]]
        if len(beat) < 5: continue
        ab = np.where(beat >= .5*np.max(beat))[0]
        if len(ab) > 1: pw.append((ab[-1]-ab[0])/FS)
        rt.append(np.argmax(beat)/FS)
    return {'mean_amp':float(np.mean(a)), 'std_amp':float(np.std(a)),
            'amp_skew':float(skew(a)),    'amp_kurt':float(kurtosis(a)),
            'sig_var':float(np.var(seg)), 'sig_energy':float(np.sum(seg**2)/len(seg)),
            'pulse_width_mean':float(np.mean(pw)) if pw else 0.,
            'rise_time_mean':float(np.mean(rt)) if rt else 0.}

def imu_all(ax, ay, az, gx, gy, gz):
    am = np.sqrt(ax**2+ay**2+az**2)
    gm = np.sqrt(gx**2+gy**2+gz**2)
    def st(s, p):
        return {f"{p}_mean":np.mean(s), f"{p}_std":np.std(s),
                f"{p}_rms":np.sqrt(np.mean(s**2)), f"{p}_range":np.ptp(s),
                f"{p}_skew":float(skew(s)), f"{p}_kurt":float(kurtosis(s))}
    f = {}
    for s, n in [(ax,"acc_x"),(ay,"acc_y"),(az,"acc_z"),(am,"acc_mag"),
                 (gx,"gyro_x"),(gy,"gyro_y"),(gz,"gyro_z"),(gm,"gyro_mag")]:
        f.update(st(s, n))
    jk = np.diff(am)*FS
    f.update({"jerk_mean":float(np.mean(np.abs(jk))), "jerk_std":float(np.std(jk)),
              "jerk_rms":float(np.sqrt(np.mean(jk**2))),
              "sma":float((np.sum(np.abs(ax))+np.sum(np.abs(ay))+np.sum(np.abs(az)))/len(ax)),
              "acc_corr_xy":float(np.corrcoef(ax,ay)[0,1]),
              "acc_corr_xz":float(np.corrcoef(ax,az)[0,1]),
              "acc_corr_yz":float(np.corrcoef(ay,az)[0,1])})
    for s, n in [(am,"acc"),(gm,"gyro")]:
        fr, p = welch(s, fs=FS, nperseg=min(256,len(s)))
        pn = p/(np.sum(p)+1e-10)
        f[f"dom_freq_{n}"]          = float(fr[np.argmax(p)])
        f[f"spectral_entropy_{n}"]  = float(-np.sum(pn*np.log(pn+1e-10)))
        f[f"low_freq_power_{n}"]    = float(np.trapezoid(p[(fr>=.1)&(fr<=2.)]))
        f[f"high_freq_power_{n}"]   = float(np.trapezoid(p[(fr>=2.)&(fr<=10.)]))
        f[f"spectral_centroid_{n}"] = float(np.sum(fr*p)/np.sum(p) if np.sum(p)>0 else 0)
    return f

# ─────────────────────────────────────────────
# CSV parsing
# ─────────────────────────────────────────────

def parse_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df.columns = df.columns.str.strip()
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    if ts_col is None:
        raise ValueError("No Timestamp column found in CSV")
    df[ts_col] = (pd.to_numeric(
        df[ts_col].astype(str).str.replace("'","").str.strip(), errors="coerce"))
    df = df.rename(columns={ts_col: "Timestamp"}).sort_values("Timestamp").reset_index(drop=True)
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "peripheral" in cl:          col_map[c] = "Peripheral ID"
        elif cl in ("value 1","value1"): col_map[c] = "Value 1"
        elif cl in ("value 2","value2"): col_map[c] = "Value 2"
        elif cl in ("value 3","value3"): col_map[c] = "Value 3"
    df = df.rename(columns=col_map)
    df["Peripheral ID"] = pd.to_numeric(df.get("Peripheral ID", np.nan), errors="coerce")
    return df

def extract_signals(df: pd.DataFrame):
    def arr(sub, col):
        return pd.to_numeric(sub[col], errors="coerce").dropna().values if col in sub.columns else np.array([])
    ppg_r = df[df["Peripheral ID"].isin([2,5])]
    acc_r = df[df["Peripheral ID"] == 0]
    gyr_r = df[df["Peripheral ID"] == 1]
    return (arr(ppg_r,"Value 3"),
            arr(acc_r,"Value 1"), arr(acc_r,"Value 2"), arr(acc_r,"Value 3"),
            arr(gyr_r,"Value 1"), arr(gyr_r,"Value 2"), arr(gyr_r,"Value 3"))

# ─────────────────────────────────────────────
# Prediction engine
# ─────────────────────────────────────────────

def run_prediction(ppg, ax, ay, az, gx, gy, gz):
    fppg = bandpass(ppg)
    rows = []
    for start in range(0, len(fppg)-WIN_N, STEP_N):
        end = start + WIN_N
        seg = fppg[start:end]
        def safe(s): return s[start:end] if end<=len(s) else s[-WIN_N:]
        peaks, _ = find_peaks(seg, distance=FS*0.4)
        if len(peaks) < 6: continue
        ibi = np.diff(peaks/FS); ibi = ibi[(ibi>.3)&(ibi<2)]
        if len(ibi) < 4: continue
        row = {"t_sec": start/FS}
        row.update(hrv_all(ibi))
        row.update(morph_all(seg, peaks))
        row.update(imu_all(safe(ax),safe(ay),safe(az),
                           safe(gx),safe(gy),safe(gz)))
        rows.append(row)
    if not rows:
        return None, None, None
    df_w = pd.DataFrame(rows)
    t_s  = df_w.pop("t_sec").values
    # Per-file normalise
    mu = df_w.mean(); sd = df_w.std().replace(0,1)
    df_n = ((df_w - mu)/sd).fillna(0)
    for feat in FEATURES:
        if feat not in df_n.columns: df_n[feat] = 0.
    X = SCALER.transform(df_n[FEATURES].values)
    # Weighted ensemble
    models = [RF, XGB, GB] + ([SVM_MDL] if SVM_MDL else [])
    weights= [.35,.25,.20,.20][:len(models)]
    proba  = sum(w*m.predict_proba(X) for w,m in zip(weights,models))
    wp     = proba[:,1]
    return int(np.mean(wp) >= THRESHOLD), float(np.mean(wp)), list(zip(t_s.tolist(), wp.tolist()))

# ─────────────────────────────────────────────
# Activity map
# ─────────────────────────────────────────────
ACTIVITY_MAP = {
    0: {"status":"Not Fatigued","emoji":"🟢","color":"#22c55e",
        "message":"Your HRV signals indicate healthy autonomic function. You're ready for high-intensity activity.",
        "recommended":[
            {"name":"🏃 Running / Jogging",     "intensity":"High",   "duration":"45–60 min"},
            {"name":"🏋️ Weight Training",       "intensity":"High",   "duration":"60 min"},
            {"name":"🚴 Cycling",               "intensity":"High",   "duration":"60 min"},
            {"name":"⚽ Team Sports",           "intensity":"High",   "duration":"60–90 min"},
            {"name":"🧗 Rock Climbing",         "intensity":"Medium", "duration":"45 min"},
        ],
        "avoid":["Prolonged sedentary periods","Skipping meals/hydration"]},
    1: {"status":"Fatigued","emoji":"🔴","color":"#ef4444",
        "message":"Fatigue detected. Reduced SD1/SD2 ratio and elevated LF/HF indicate autonomic stress. Prioritise recovery.",
        "recommended":[
            {"name":"😴 Rest / Sleep",          "intensity":"—",      "duration":"7–9 hrs"},
            {"name":"🧘 Meditation / Breathing","intensity":"Low",    "duration":"20 min"},
            {"name":"🚶 Light Walking",         "intensity":"Low",    "duration":"20–30 min"},
            {"name":"🛁 Gentle Stretching",     "intensity":"Low",    "duration":"30 min"},
            {"name":"💧 Hydration + Nutrition", "intensity":"—",      "duration":"Throughout day"},
        ],
        "avoid":["High-intensity exercise","Driving / heavy machinery",
                 "Critical decision-making","Alcohol / stimulants"]},
}

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status":"running","models_loaded":bool(RF),"features":len(FEATURES)}

@app.get("/health")
def health():
    return {"ok":True,"models":bool(RF),"features":len(FEATURES),"threshold":THRESHOLD}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = parse_csv(content)
    except Exception as e:
        raise HTTPException(400, f"CSV parse error: {e}")
    try:
        ppg,ax,ay,az,gx,gy,gz = extract_signals(df)
    except Exception as e:
        raise HTTPException(400, f"Signal error: {e}")
    if len(ppg) < WIN_N:
        raise HTTPException(422, f"PPG too short ({len(ppg)} samples, need {WIN_N})")

    pred, prob, wins = run_prediction(ppg,ax,ay,az,gx,gy,gz)
    if pred is None:
        raise HTTPException(422, "Not enough valid windows. Recording may be too noisy.")

    info = ACTIVITY_MAP[pred]
    return {
        "prediction":    pred,
        "probability":   round(prob, 4),
        "confidence":    round(abs(prob-0.5)*200, 1),
        "status":        info["status"],
        "emoji":         info["emoji"],
        "color":         info["color"],
        "message":       info["message"],
        "recommended":   info["recommended"],
        "avoid":         info["avoid"],
        "duration_sec":  round(len(ppg)/FS, 1),
        "n_windows":     len(wins),
        "windows":       [{"t":round(t,1),"prob":round(p,4)} for t,p in wins],
        "signal_info":   {"ppg":len(ppg),"acc":len(ax),"gyro":len(gx)},
    }

@app.post("/signal-stats")
async def signal_stats(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = parse_csv(content)
    except Exception as e:
        raise HTTPException(400, str(e))
    ppg,ax,ay,az,gx,gy,gz = extract_signals(df)
    step = max(1, len(ppg)//1500)
    t_chart = (np.arange(len(ppg))[::step]/FS).tolist()
    v_chart = ppg[::step].tolist()
    imu_m   = np.sqrt(ax**2+ay**2+az**2) if len(ax)>0 else np.array([0])
    return {
        "duration_sec" : round(len(ppg)/FS, 1),
        "ppg_samples"  : int(len(ppg)),
        "acc_samples"  : int(len(ax)),
        "gyro_samples" : int(len(gx)),
        "motion_pct"   : round(float(np.mean(imu_m > np.percentile(imu_m,85)))*100,1),
        "ppg_chart"    : {"t": t_chart, "v": v_chart},
    }
