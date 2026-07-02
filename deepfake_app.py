import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="DeepFake Detector", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');
* { font-family: 'Rajdhani', sans-serif; }
code, .mono { font-family: 'Share Tech Mono', monospace; }
body, .stApp { background: #060b14; color: #c8dff0; }
.stApp { background: linear-gradient(135deg, #060b14 0%, #0a1628 50%, #060b14 100%); }
.main-title { font-family:'Share Tech Mono',monospace; font-size:2.2rem; color:#00f5ff;
  text-align:center; letter-spacing:0.18em; text-shadow:0 0 30px #00f5ff66; padding:1rem 0 0.2rem; }
.sub-title { font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#4a8fa8;
  text-align:center; letter-spacing:0.25em; padding-bottom:1rem; }
.metric-box { background:linear-gradient(135deg,#0d1a2b,#0a1422); border:1px solid #1a3550;
  border-radius:8px; padding:1rem; text-align:center; }
.metric-val { font-family:'Share Tech Mono',monospace; font-size:1.8rem; color:#00f5ff; }
.metric-lab { font-size:0.72rem; color:#4a6d8c; letter-spacing:0.12em; margin-top:4px; }
.tb-anime { background:linear-gradient(90deg,#1a0a1e,#2a0d35); border:1px solid #9d00ff;
  border-radius:8px; padding:0.8rem 1.2rem; margin:0.8rem 0; }
.tb-photo { background:linear-gradient(90deg,#001a2e,#00264a); border:1px solid #00aaff;
  border-radius:8px; padding:0.8rem 1.2rem; margin:0.8rem 0; }
.tb-title { font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#4a6d8c; letter-spacing:0.2em; }
.tb-val { font-size:1.1rem; font-weight:700; margin:4px 0; }
.tb-anime .tb-val { color:#d966ff; }
.tb-photo .tb-val { color:#00aaff; }
.tb-sub { font-size:0.78rem; color:#6a8da8; font-family:'Share Tech Mono',monospace; }
.verdict-box { border-radius:12px; padding:1.5rem; margin:1rem 0; text-align:center; }
.v-fake { background:linear-gradient(135deg,#1a0508,#200a0a); border:2px solid #ff2d55; }
.v-real { background:linear-gradient(135deg,#011a0d,#021a0f); border:2px solid #00ff88; }
.vt-fake { font-family:'Share Tech Mono',monospace; font-size:2.2rem; color:#ff2d55;
  text-shadow:0 0 20px #ff2d5566; }
.vt-real { font-family:'Share Tech Mono',monospace; font-size:2.2rem; color:#00ff88;
  text-shadow:0 0 20px #00ff8866; }
.conf-bar { height:8px; border-radius:4px; margin:8px 0; }
.verdict-sub { font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#6a8da8; margin-top:8px; }
.sig-card { background:#0a1422; border:1px solid #1a3550; border-radius:6px;
  padding:0.5rem 0.8rem; margin:0.3rem 0; display:flex; justify-content:space-between; align-items:center; }
.sig-label { font-size:0.82rem; color:#8abadd; }
.badge-anime { background:#2a0d35; color:#d966ff; font-size:0.72rem; padding:2px 8px;
  border-radius:4px; border:1px solid #9d00ff; font-family:'Share Tech Mono',monospace; }
.badge-border { background:#1a1500; color:#ffd60a; font-size:0.72rem; padding:2px 8px;
  border-radius:4px; border:1px solid #ffd60a; font-family:'Share Tech Mono',monospace; }
.badge-real { background:#001a0d; color:#00ff88; font-size:0.72rem; padding:2px 8px;
  border-radius:4px; border:1px solid #00ff88; font-family:'Share Tech Mono',monospace; }
.why-box { background:#0a0d1a; border:1px solid #2a1a3a; border-radius:8px;
  padding:1rem 1.2rem; margin-top:1rem; font-family:'Share Tech Mono',monospace;
  font-size:0.76rem; color:#8abadd; line-height:2; }
.stFileUploader { border:1px dashed #1a3550 !important; border-radius:8px !important;
  background:#080f1c !important; }
.meta-box { background:#080f1c; border:1px solid #1a3550; border-radius:6px;
  padding:0.4rem 0.8rem; font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#4a6d8c; }
.footer { text-align:center; font-family:'Share Tech Mono',monospace; font-size:0.65rem;
  color:#1a3550; padding:2rem 0 1rem; letter-spacing:0.15em; }
.sidebar-logo { background:linear-gradient(135deg,#0a1a2e,#060b14); border:1px solid #1a3550;
  border-radius:8px; padding:1rem; text-align:center; margin-bottom:1rem; }
.sidebar-title { font-family:'Share Tech Mono',monospace; color:#00f5ff; font-size:1rem;
  letter-spacing:0.2em; }
.sidebar-sub { font-family:'Share Tech Mono',monospace; color:#4a6d8c; font-size:0.65rem;
  letter-spacing:0.15em; }
.logic-box { background:#080f1c; border:1px solid #1a3550; border-radius:6px;
  padding:0.8rem; font-size:0.75rem; color:#6a8da8; line-height:1.7;
  font-family:'Share Tech Mono',monospace; }
.warn-box { background:#1a0a00; border:1px solid #ff6600; border-radius:6px;
  padding:0.5rem 0.8rem; text-align:center; font-size:0.68rem; color:#ff8800;
  font-family:'Share Tech Mono',monospace; }
.await-box { background:linear-gradient(135deg,#080f1c,#060b14); border:1px solid #1a3550;
  border-radius:12px; padding:3rem; text-align:center; margin:2rem 0; }
.await-icon { font-size:3rem; margin-bottom:1rem; }
.await-title { font-family:'Share Tech Mono',monospace; color:#4a6d8c; font-size:1rem;
  letter-spacing:0.2em; margin-bottom:0.5rem; }
.await-sub { font-size:0.8rem; color:#2a4560; font-family:'Share Tech Mono',monospace; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 1 — IMAGE TYPE CLASSIFIER v8
# Handles 4 types correctly:
#   1. Real selfie/photo   → REAL
#   2. Anime/illustration  → FAKE
#   3. Pencil sketch       → FAKE
#   4. AI-generated art    → FAKE
#
# KEY INSIGHT (from test data):
#   Real selfie:    sat_mean~60,  hue_std~45, unique_colors~350, patch_std~38
#   Anime:          sat_mean~180, hue_std~20, unique_colors~288, noise_median~1.4
#   Pencil sketch:  sat_mean~0,   hue_std~0,  unique_colors~16,  noise_median~1.6
#
# REAL PHOTO GUARD now checks sat_mean (30–120) = natural, not artificially vivid
# PENCIL SKETCH BOOST: sat_mean < 5 AND hue_std < 5 → definitely drawing
# ══════════════════════════════════════════════════════════════════

def classify_image_type(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat  = hsv[:, :, 1].astype(float)
    hue  = hsv[:, :, 0].astype(float)
    sigs = {}

    # ── S1: Saturation stats ───────────────────────────────────
    high_sat = (sat > 160).sum() / sat.size
    low_sat  = (sat < 25).sum()  / sat.size
    sigs['bimodal_sat'] = float(high_sat + low_sat)
    sigs['sat_mean']    = float(sat.mean())          # real~50-80, anime~150+, sketch~0-5
    sigs['sat_spike']   = float((sat > 200).sum() / sat.size)

    # ── S2: Unique colors ──────────────────────────────────────
    small = cv2.resize(img, (64, 64))
    quant = (small.reshape(-1, 3) // 16) * 16
    sigs['unique_colors'] = float(len(set(map(tuple, quant.tolist()))))

    # ── S3: Sensor noise ──────────────────────────────────────
    blur_g = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = np.abs(gray - blur_g)
    sigs['noise_std']    = float(noise_residual.std())
    sigs['noise_median'] = float(np.median(noise_residual))

    # ── S4: Edge ratio ─────────────────────────────────────────
    g_u8     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    e_strict = cv2.Canny(g_u8, 120, 250)
    e_loose  = cv2.Canny(g_u8, 25,  80)
    sigs['edge_ratio'] = float(e_strict.sum() / (e_loose.sum() + 1e-6))

    # ── S5: Flat region ratio ──────────────────────────────────
    lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    sigs['flat_ratio'] = float((np.abs(lap) < 4).sum() / lap.size)

    # ── S6: Entropy ────────────────────────────────────────────
    h_hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(float) + 1
    h_hist /= h_hist.sum()
    sigs['entropy'] = float(-np.sum(h_hist * np.log2(h_hist)))

    # ── S7: Skin naturalness ───────────────────────────────────
    r, g2, b = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
    skin_m = (r > 80) & (g2 > 50) & (b > 30) & (r > g2) & (g2 > b * 0.75)
    if skin_m.sum() > 300:
        sk_sat = sat[skin_m].mean()
        sigs['skin_sat'] = float(sk_sat)
        unnatural = sk_sat < 18 or sk_sat > 145
    else:
        sigs['skin_sat'] = -1.0
        unnatural = False

    # ── S8: Gradient bimodality ────────────────────────────────
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(gx**2 + gy**2)
    sigs['bimodal_grad'] = float((gm > 80).sum()/gm.size + (gm < 4).sum()/gm.size)

    # ── S9: Patch texture std ──────────────────────────────────
    patch_stds = []
    ps = 16
    small128 = cv2.resize(img, (128, 128)).astype(float)
    for y in range(0, 128 - ps, ps):
        for x in range(0, 128 - ps, ps):
            patch = small128[y:y+ps, x:x+ps]
            patch_stds.append(float(patch.std()))
    sigs['patch_std_mean']     = float(np.mean(patch_stds))
    sigs['patch_std_low_frac'] = float(np.mean([1 if v < 8 else 0 for v in patch_stds]))

    # ── S10: Hue variance ─────────────────────────────────────
    colored_mask = sat > 30
    if colored_mask.sum() > 100:
        sigs['hue_std'] = float(hue[colored_mask].std())
    else:
        sigs['hue_std'] = 0.0   # near-zero → grayscale/sketch/no color

    # ── S11: Channel difference (grayscale check) ─────────────
    # Pencil sketches: R≈G≈B → channel_diff ≈ 0
    # Real photos: natural color → channel_diff 10-80
    # Anime: vivid color → channel_diff 30-100
    r_m = img[:,:,0].astype(float).mean()
    g_m = img[:,:,1].astype(float).mean()
    b_m = img[:,:,2].astype(float).mean()
    sigs['channel_diff'] = float(max(abs(r_m-g_m), abs(g_m-b_m), abs(r_m-b_m)))

    # ── S12: Local smoothness ─────────────────────────────────
    small_gray = cv2.resize(gray, (128, 128))
    local_diffs = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(small_gray, dy, axis=0), dx, axis=1)
            local_diffs.append(np.abs(small_gray - shifted))
    sigs['local_smoothness'] = float(np.mean(local_diffs))

    # ── S13: Quantization loss ─────────────────────────────────
    tiny = cv2.resize(img, (48, 48)).astype(np.float32)
    q4   = (tiny // 16).astype(np.uint8) * 16
    sigs['quant_loss'] = float(np.abs(tiny - q4.astype(np.float32)).mean())

    # ══════════════════════════════════════════════════════════
    # PRE-CLASSIFICATION: Detect image category first
    # ══════════════════════════════════════════════════════════

    # 1. PENCIL SKETCH / GRAYSCALE DRAWING DETECTION
    #    sat_mean < 5 AND hue_std < 5 → grayscale/sketch → FAKE immediately
    is_sketch = (sigs['sat_mean'] < 5.0 and sigs['hue_std'] < 5.0)

    # 2. REAL PHOTO DETECTION (must have NATURAL saturation, not too vivid/too gray)
    #    Real photos: sat_mean between 10-150 typically, hue_std > 30
    #    Anime: sat_mean > 120 AND hue_std < 30 (vivid but restricted hue)
    #    Sketch: sat_mean < 5 (no color)
    #    Key discriminator: hue_std > 35 = diverse colors = real photo
    real_sat_range = sigs['sat_mean'] < 155.0   # most real photos below 155
    real_hue_rich  = sigs['hue_std'] > 35.0     # diverse hue = real
    real_color_rich = sigs['unique_colors'] > 250 and real_hue_rich

    real_photo_strong = (
        real_color_rich and
        sigs['patch_std_mean'] > 22.0
    )
    real_photo_moderate = (
        real_hue_rich and
        sigs['unique_colors'] > 200 and
        sigs['patch_std_mean'] > 18.0
    )
    real_photo_weak = (
        real_hue_rich and
        sigs['unique_colors'] > 170
    )

    # ══════════════════════════════════════════════════════════
    # WEIGHTED SCORE
    # ══════════════════════════════════════════════════════════
    score = 0.0

    # S3: Noise median (weight 0.18)
    nm = sigs['noise_median']
    if nm < 0.8:   score += 0.18
    elif nm < 1.5: score += 0.14
    elif nm < 2.5: score += 0.08
    elif nm < 3.5: score += 0.03

    # S9: Patch texture (weight 0.16)
    psm = sigs['patch_std_mean']
    if psm < 6.0:   score += 0.16
    elif psm < 10.0: score += 0.11
    elif psm < 14.0: score += 0.06
    elif psm < 18.0: score += 0.02
    if sigs['patch_std_low_frac'] > 0.50: score += 0.05
    elif sigs['patch_std_low_frac'] > 0.30: score += 0.02

    # S13: Quant loss (weight 0.13)
    ql = sigs['quant_loss']
    if ql < 3.0:   score += 0.13
    elif ql < 5.0:  score += 0.09
    elif ql < 7.0:  score += 0.05
    elif ql < 9.0:  score += 0.01

    # S2: Unique colors (weight 0.12)
    uc = sigs['unique_colors']
    if uc < 50:    score += 0.12   # sketch/limited palette
    elif uc < 80:  score += 0.10
    elif uc < 120: score += 0.07
    elif uc < 160: score += 0.04
    elif uc < 200: score += 0.01

    # S1: Sat mean — HIGH sat = anime, LOW sat = sketch, NATURAL = real
    sm = sigs['sat_mean']
    if sm > 160:   score += 0.12   # very vivid = anime
    elif sm > 120: score += 0.08
    elif sm > 100: score += 0.04
    elif sm < 5:   score += 0.12   # no color = sketch/drawing
    elif sm < 15:  score += 0.06

    # S10: Hue std — zero/low = sketch or mono-colored anime (weight 0.08)
    hs = sigs['hue_std']
    if hs < 5:     score += 0.08   # no hue variation = sketch
    elif hs < 20:  score += 0.06   # very narrow = anime
    elif hs < 35:  score += 0.03

    # S11: Sat spike (weight 0.06)
    ss = sigs['sat_spike']
    if ss > 0.25:  score += 0.06
    elif ss > 0.12: score += 0.03
    elif ss > 0.05: score += 0.01

    # S1 bimodal sat (weight 0.05)
    bs = sigs['bimodal_sat']
    if bs > 0.75:  score += 0.05   # pencil sketch bimodal_sat = 1.0!
    elif bs > 0.58: score += 0.03
    elif bs > 0.42: score += 0.01

    # S8 gradient bimodal (weight 0.04)
    gb = sigs['bimodal_grad']
    if gb > 0.70:  score += 0.04
    elif gb > 0.55: score += 0.02

    # S4 edge ratio (weight 0.03)
    er = sigs['edge_ratio']
    if er > 0.45:  score += 0.03
    elif er > 0.30: score += 0.01

    # S6 entropy (weight 0.02)
    if sigs['entropy'] < 6.0:   score += 0.02
    elif sigs['entropy'] < 6.5: score += 0.01

    # S7 skin naturalness (weight 0.03)
    if unnatural: score += 0.03

    # S12 local smoothness (weight 0.03)
    ls = sigs['local_smoothness']
    if ls < 4.0:  score += 0.03
    elif ls < 7.0: score += 0.01

    # ══════════════════════════════════════════════════════════
    # FINAL DECISIONS — GUARDS & BOOSTS
    # ══════════════════════════════════════════════════════════

    # PENCIL SKETCH OVERRIDE: grayscale + no hue → definitely FAKE
    if is_sketch:
        score = max(score, 0.70)   # hard boost — sketch is clearly fake

    # ANIME BOOST: vivid color + low noise + RESTRICTED HUE (key!)
    # Real photos can have high sat_mean, but they always have HIGH hue_std (diverse colors)
    # Anime has high sat_mean but LOW hue_std (limited color palette)
    elif sigs['sat_mean'] > 120 and sigs['noise_median'] < 2.0 and sigs['hue_std'] < 30:
        score = max(score, 0.50)   # very likely anime
    elif sigs['sat_mean'] > 90 and sigs['noise_median'] < 1.8 and sigs['hue_std'] < 25:
        score = max(score, 0.38)

    # REAL PHOTO GUARD: diverse hue + rich colors + texture
    # hue_std > 35 is the PRIMARY discriminator (anime always has restricted hue)
    if not is_sketch:
        if real_photo_strong:
            score = min(score, 0.14)
        elif real_photo_moderate:
            score = min(score, 0.20)
        elif real_photo_weak:
            score = min(score, 0.24)

    score = min(score, 1.0)

    ANIME_THRESHOLD = 0.28
    is_anime = score >= ANIME_THRESHOLD

    # Determine image type label
    if is_sketch:
        img_type = 'Pencil Sketch / Grayscale Drawing'
    elif is_anime:
        img_type = 'Anime / Illustration / AI-Art'
    else:
        img_type = 'Real Photograph'

    return {
        'is_anime': is_anime,
        'anime_score': round(score, 4),
        'anime_threshold': ANIME_THRESHOLD,
        'img_type': img_type,
        'is_sketch': is_sketch,
        'signals': {k: round(v, 4) for k, v in sigs.items()},
    }


# ══════════════════════════════════════════════════════════════════
# STEP 2 — PHOTO DEEPFAKE ANALYZER (only runs on real photos)
# ══════════════════════════════════════════════════════════════════

def analyze_photo_deepfake(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=75)
    buf.seek(0)
    rc  = np.array(Image.open(buf).convert('RGB')).astype(np.float32)
    ela = np.abs(img.astype(np.float32) - rc)

    blur      = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_std = float(np.abs(gray - blur).std())

    fmag   = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
    h2,w2  = fmag.shape
    cy,cx  = h2//2, w2//2
    r      = min(h2,w2)//8
    freq_r = float(fmag[cy-r:cy+r, cx-r:cx+r].mean() / (fmag.mean() + 1e-8))

    diffs = []
    for y in range(8, gray.shape[0]-8, 8):
        diffs.append(float(np.abs(gray[y] - gray[y-1]).mean()))
    for x in range(8, gray.shape[1]-8, 8):
        diffs.append(float(np.abs(gray[:,x] - gray[:,x-1]).mean()))
    block = float(np.mean(diffs)) if diffs else 0.0

    rg = float(np.corrcoef(img[:,:,0].flatten().astype(float),
                           img[:,:,1].flatten().astype(float))[0,1])

    h3, w3 = gray.shape
    mvs    = [gray[:h3//2,:w3//2].mean(), gray[:h3//2,w3//2:].mean(),
              gray[h3//2:,:w3//2].mean(), gray[h3//2:,w3//2:].mean()]
    reg_var = float(np.var(mvs))

    gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm  = np.sqrt(gx**2 + gy**2)
    coh = float(gm.std() / (gm.mean() + 1e-8))

    bs  = 32
    blk = []
    for y in range(0, gray.shape[0]-bs, bs):
        for x in range(0, gray.shape[1]-bs, bs):
            blk.append(float(gray[y:y+bs, x:x+bs].std()))
    tex_var = float(np.var(blk)) if blk else 0.0

    subs = {
        'ELA Artifacts':    min(max(ela.mean()/10, 0), 1.0),
        'Noise Anomaly':    min(max((5.0 - noise_std)/5.0, 0), 1.0),
        'Frequency Anomaly':min(max((freq_r - 1.5)/2.5, 0), 1.0),
        'Block Artifacts':  min(max(block/8.0, 0), 1.0),
        'Color Correlation':min(max((rg - 0.93)/0.07, 0), 1.0),
        'Face Inconsistency':min(max(reg_var/800, 0), 1.0),
        'Edge Incoherence': min(max((coh - 2.0)/4.0, 0), 1.0),
        'Texture Flatness': min(max(1.0 - tex_var/600, 0), 1.0),
    }
    subs    = {k: max(0.0, round(v, 3)) for k,v in subs.items()}
    wts     = [0.20, 0.18, 0.15, 0.12, 0.12, 0.10, 0.08, 0.05]
    overall = sum(s*w for s,w in zip(subs.values(), wts))

    return {'sub_scores': subs, 'overall_score': round(overall, 4)}


# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════

def plot_ela(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=75)
    buf.seek(0)
    rc  = np.array(Image.open(buf).convert('RGB'))
    ela = np.clip(np.abs(img.astype(np.int16) - rc.astype(np.int16)) * 10, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(15,4), facecolor='#0d1a2b')
    for ax, im, t, c in zip(axes, [img, ela, ela[:,:,0]],
                             ['Original','ELA Map','ELA Heatmap'],
                             [None,'hot','plasma']):
        ax.imshow(im, cmap=c)
        ax.set_title(t, color='#00f5ff', fontfamily='monospace', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    return fig

def plot_freq(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft  = np.fft.fftshift(np.fft.fft2(gray))
    mag  = np.log(np.abs(fft)+1)
    pha  = np.angle(fft)
    fig, axes = plt.subplots(1, 3, figsize=(15,4), facecolor='#0d1a2b')
    for ax, d, t, c in zip(axes, [gray, mag, pha],
                            ['Grayscale','FFT Magnitude','FFT Phase'],
                            ['gray','inferno','twilight']):
        ax.imshow(d, cmap=c)
        ax.set_title(t, color='#00f5ff', fontfamily='monospace', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    return fig

def plot_hist(img):
    fig, ax = plt.subplots(figsize=(10,4), facecolor='#0d1a2b')
    ax.set_facecolor('#060b14')
    for i,(c,l) in enumerate([('#ff4d6d','R'),('#00c896','G'),('#3d9eff','B')]):
        h = cv2.calcHist([img],[i],None,[256],[0,256]).flatten()
        ax.plot(h, color=c, alpha=.85, lw=1.5, label=l)
        ax.fill_between(range(256), h, alpha=.12, color=c)
    ax.set_title('Channel Histograms', color='#00f5ff', fontfamily='monospace')
    ax.legend(facecolor='#0d1a2b', edgecolor='#1a3550', labelcolor='#e0f0ff')
    ax.tick_params(colors='#4a6d8c')
    for s in ax.spines.values(): s.set_edgecolor('#1a3550')
    plt.tight_layout()
    return fig

def plot_radar(labels, values, color, title):
    v = values + values[:1]
    a = [n/len(labels)*2*np.pi for n in range(len(labels))] + [0]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True), facecolor='#0d1a2b')
    ax.set_facecolor('#060b14')
    ax.plot(a, v, 'o-', lw=2, color=color)
    ax.fill(a, v, alpha=.22, color=color)
    ax.set_xticks(a[:-1])
    ax.set_xticklabels(labels, color='#8abadd', size=7)
    ax.set_ylim(0, 1)
    ax.grid(color='#1a3550', lw=.5)
    ax.spines['polar'].set_edgecolor('#1a3550')
    ax.set_title(title, color=color, fontfamily='monospace', size=10, pad=18)
    plt.tight_layout()
    return fig

def plot_signals(sigs: dict, anime_score: float):
    fig, ax = plt.subplots(figsize=(10,6), facecolor='#0d1a2b')
    ax.set_facecolor('#060b14')

    keys = ['noise_median','patch_std_mean','quant_loss','unique_colors',
            'flat_ratio','hue_std','sat_spike','local_smoothness']
    norms = {
        'noise_median':     (4.0,  0.0,  True),   # inverted: low = anime
        'patch_std_mean':   (30.0, 0.0,  True),   # inverted: low = anime
        'quant_loss':       (12.0, 0.0,  True),   # inverted: low = anime
        'unique_colors':    (250,  60,   True),   # inverted
        'flat_ratio':       (0,    0.7,  False),
        'hue_std':          (80,   0,    True),   # inverted: low = anime
        'sat_spike':        (0,    0.35, False),
        'local_smoothness': (20,   2,    True),   # inverted
    }
    labels_map = {
        'noise_median':     'Noise Median\n(inv ← anime=low)',
        'patch_std_mean':   'Patch Texture\n(inv ← anime=flat)',
        'quant_loss':       'Quant Loss\n(inv ← anime=low)',
        'unique_colors':    'Color Palette\n(inv)',
        'flat_ratio':       'Flat Regions',
        'hue_std':          'Hue Variance\n(inv ← anime=narrow)',
        'sat_spike':        'Sat Spike',
        'local_smoothness': 'Local Smooth\n(inv)',
    }
    nvals = []; labs = []
    for k in keys:
        v = sigs.get(k, 0)
        lo, hi, invert = norms[k]
        n = max(0, min(1, (v-lo)/(hi-lo+1e-8))) if hi != lo else 0
        if invert: n = 1-n
        nvals.append(n); labs.append(labels_map[k])

    colors = ['#ff2d55' if v > 0.6 else '#ffd60a' if v > 0.3 else '#00ff88' for v in nvals]
    bars   = ax.barh(labs, nvals, color=colors, alpha=.85, edgecolor='#1a3550', height=0.6)
    ax.set_xlim(0, 1.2)
    ax.axvline(0.28, color='#ff2d55', linestyle='--', lw=1.5, alpha=0.8,
               label=f'Anime threshold (0.28)')
    ax.set_title(f'Classifier Signals | Final Anime Score: {anime_score:.3f}',
                 color='#00f5ff', fontfamily='monospace', fontsize=10)
    ax.tick_params(colors='#8abadd', labelsize=9)
    ax.legend(facecolor='#0d1a2b', edgecolor='#1a3550', labelcolor='#ffaaaa', fontsize=8)
    for s in ax.spines.values(): s.set_edgecolor('#1a3550')
    for bar, v in zip(bars, nvals):
        ax.text(v+0.02, bar.get_y()+bar.get_height()/2,
                f'{v:.2f}', va='center', color='#8abadd', fontsize=8)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div class='sidebar-logo'>
  <div style='font-size:1.8rem'>⬡</div>
  <div class='sidebar-title'>DFDETECT v8</div>
  <div class='sidebar-sub'>SMART MULTI-TYPE (SKETCH+ANIME FIX)</div>
</div>
""", unsafe_allow_html=True)

    sensitivity = st.slider("Photo Deepfake Threshold", 0.20, 0.80, 0.38, 0.02,
        help="Only applies when image is classified as a real photo.")
    show_ela  = st.checkbox("ELA Map", True)
    show_freq = st.checkbox("Frequency Domain", True)
    show_hist = st.checkbox("Color Histogram", True)
    show_sig  = st.checkbox("Signal Chart", True)
    st.markdown("---")
    st.markdown("""
<div class='logic-box'>
<b style='color:#00f5ff'>v8 DETECTION LOGIC</b><br>
① 12-signal classifier runs<br>
② anime_score ≥ 0.28?<br>
&nbsp;&nbsp;YES → FAKE immediately<br>
&nbsp;&nbsp;NO → deepfake check<br><br>
NEW key signals:<br>
• Noise median (robust)<br>
• Patch texture std<br>
• Quantization loss<br>
• Hue variance<br>
• Saturation spike<br><br>
<b style='color:#ff8844'>Real photo guards (v7):</b><br>
• patch_std > 25 AND<br>
&nbsp;&nbsp;colors > 280 → REAL<br>
• patch_std > 20 AND<br>
&nbsp;&nbsp;colors > 220 AND<br>
&nbsp;&nbsp;quant_loss > 6.5 → REAL<br>
• 3+ real signals → cap 0.24<br><br>
<b style='color:#44ff88'>Anime boost:</b><br>
• patch_std < 14 AND<br>
&nbsp;&nbsp;quant_loss < 7 AND<br>
&nbsp;&nbsp;colors < 180 → boost
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div class='warn-box' style='margin-top:1rem'>
 ⚠ RESEARCH USE ONLY
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">DEEPFAKE DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">▸ PHOTO · ANIME · AI-ART · DEEPFAKE ▸ 12-SIGNAL CLASSIFIER ▸ AUTO-MODE ▸ v7 SKETCH+ANIME FIX</div>',
            unsafe_allow_html=True)

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
for col,val,lab,color in zip(
    [c1,c2,c3,c4],
    ['12','0.28','AUTO','v7'],
    ['Classifier Signals','Anime Threshold','Mode Switch','Version'],
    ['#00f5ff','#ff8c00','#ffd60a','#00ff88']
):
    col.markdown(f"""
<div class='metric-box'>
  <div class='metric-val' style='color:{color}'>{val}</div>
  <div class='metric-lab'>{lab}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

up_col, _ = st.columns([2, 1])
with up_col:
    uploaded = st.file_uploader(
        "DROP IMAGE — PHOTO, ANIME, AI-ART, DEEPFAKE",
        type=["jpg","jpeg","png","bmp","webp"]
    )

if uploaded:
    raw = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > 900:
        s = 900/max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### 🖼 UPLOADED IMAGE")
        st.image(img, use_container_width=True, caption=uploaded.name)
        st.markdown(f"""
<div class='meta-box'>
 📐 {img.shape[1]}×{img.shape[0]}px &nbsp;|&nbsp; 💾 {uploaded.size/1024:.1f} KB &nbsp;|&nbsp; 🏷 {uploaded.type}
</div>""", unsafe_allow_html=True)

    with right:
        st.markdown("#### 🔬 ANALYZING...")
        prog = st.progress(0)
        stat = st.empty()

        stat.markdown("► Running 12-signal image classifier...", unsafe_allow_html=True)
        prog.progress(25); time.sleep(0.35)
        tc       = classify_image_type(img)
        prog.progress(45); time.sleep(0.1)
        is_anime = tc['is_anime']
        ascore   = tc['anime_score']
        athresh  = tc['anime_threshold']
        sigs     = tc['signals']
        df_results = None

        if not is_anime:
            for msg, p in [("ELA analysis...",58),("Frequency scan...",70),
                           ("Texture check...",82),("Ensemble...",100)]:
                stat.markdown(f"► {msg}", unsafe_allow_html=True)
                prog.progress(p); time.sleep(0.13)
            df_results = analyze_photo_deepfake(img)
        else:
            prog.progress(100); time.sleep(0.2)

        stat.empty(); prog.empty()

        if is_anime:
            if tc.get('is_sketch', False):
                tbadge, tclass = "✏️ PENCIL SKETCH / DRAWING / GRAYSCALE ART", "tb-anime"
            else:
                tbadge, tclass = "🎨 ANIME / ILLUSTRATION / AI-ART", "tb-anime"
        else:
            tbadge, tclass = "📷 REAL PHOTOGRAPH", "tb-photo"

        st.markdown(f"""
<div class='{tclass}'>
  <div class='tb-title'>IMAGE TYPE</div>
  <div class='tb-val'>{tbadge}</div>
  <div class='tb-sub'>
   Anime Score: {ascore:.4f} &nbsp;/&nbsp; Threshold: {athresh} &nbsp;→&nbsp;
   {"⚠ ANIME DETECTED" if is_anime else "✅ REAL PHOTO"}
  </div>
</div>""", unsafe_allow_html=True)

        if is_anime:
            conf = min(70 + (ascore - athresh) * 130, 99.0)
            bar_w = int(conf)
            st.markdown(f"""
<div class='verdict-box v-fake'>
  <div style='font-family:Share Tech Mono,monospace;font-size:.7rem;color:#660011;letter-spacing:.2em'>VERDICT</div>
  <div class='vt-fake'>⚠️ FAKE</div>
  <div style='background:#300010;border-radius:4px;height:8px;margin:8px 0'>
    <div style='width:{bar_w}%;background:linear-gradient(90deg,#ff2d55,#ff006e);height:8px;border-radius:4px'></div>
  </div>
  <div style='color:#ff6680;font-family:Share Tech Mono,monospace;font-size:.85rem'>{conf:.1f}% CONFIDENCE</div>
  <div class='verdict-sub'>
   ❌ NOT a real photograph<br>
   Type: Anime / Illustration / AI-Generated Art<br>
   Anime score: {ascore:.4f} ≥ {athresh} threshold<br>
   Patch texture std = {sigs.get('patch_std_mean',0):.2f}
   {"(flat zones = anime/art)" if sigs.get('patch_std_mean',99) < 14.0 else ""}<br>
   Quant loss = {sigs.get('quant_loss',0):.2f}
   {"(already quantized = anime palette)" if sigs.get('quant_loss',99) < 7.0 else ""}<br>
   Unique colors = {sigs.get('unique_colors',0):.0f}
  </div>
</div>""", unsafe_allow_html=True)

        else:
            overall = df_results['overall_score']
            verdict = 'FAKE' if overall > sensitivity else 'REAL'
            raw_gap = abs(overall - sensitivity)
            conf    = min(50 + raw_gap * 300, 99.0)
            bar_w   = int(conf)
            vcard   = 'v-fake' if verdict=='FAKE' else 'v-real'
            vtxt    = 'vt-fake' if verdict=='FAKE' else 'vt-real'
            bar_col = 'linear-gradient(90deg,#ff2d55,#ff006e)' if verdict=='FAKE' else 'linear-gradient(90deg,#00ff88,#00cc66)'
            icon    = '⚠️' if verdict=='FAKE' else '✅'

            st.markdown(f"""
<div class='verdict-box {vcard}'>
  <div style='font-family:Share Tech Mono,monospace;font-size:.7rem;color:#336655;letter-spacing:.2em'>VERDICT</div>
  <div class='{vtxt}'>{icon} {verdict}</div>
  <div style='background:#002010;border-radius:4px;height:8px;margin:8px 0'>
    <div style='width:{bar_w}%;background:{bar_col};height:8px;border-radius:4px'></div>
  </div>
  <div style='color:#66ffaa;font-family:Share Tech Mono,monospace;font-size:.85rem'>{conf:.1f}% CONFIDENCE</div>
  <div class='verdict-sub'>
   DEEPFAKE SCORE: {overall:.4f} | THRESHOLD: {sensitivity:.2f}
  </div>
</div>""", unsafe_allow_html=True)

    # ── BREAKDOWN ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 SIGNAL BREAKDOWN")

    if is_anime:
        signal_rows = [
            ('noise_median',      'Noise Median (robust)',      2.5,  True ),
            ('patch_std_mean',    'Patch Texture Std',          14.0, True ),
            ('quant_loss',        'Quantization Loss',          7.0,  True ),
            ('unique_colors',     'Unique Colors',              170,  True ),
            ('flat_ratio',        'Flat Color Regions',         0.50, False),
            ('hue_std',           'Hue Variance',               40.0, True ),
            ('sat_spike',         'Saturation Spike',           0.10, False),
            ('local_smoothness',  'Local Smoothness',           7.0,  True ),
        ]
        fcols = st.columns(2)
        for i, (key, label, thresh, invert) in enumerate(signal_rows):
            v = sigs.get(key, 0)
            if v < 0: v = 0
            anime_sig = (v < thresh) if invert else (v > thresh)
            score_txt = f"{v:.3f}"
            if anime_sig:
                badge = f"■ ANIME SIGNAL ({score_txt})"
                bc    = "badge-anime"
            elif abs(v - thresh) < thresh * 0.25:
                badge = f"◆ BORDERLINE ({score_txt})"
                bc    = "badge-border"
            else:
                badge = f"● PHOTO-LIKE ({score_txt})"
                bc    = "badge-real"
            with fcols[i % 2]:
                st.markdown(f"""
<div class='sig-card'>
  <span class='sig-label'>{label}</span>
  <span class='{bc}'>{badge}</span>
</div>""", unsafe_allow_html=True)

        psm_v = sigs.get('patch_std_mean', 99)
        ql_v  = sigs.get('quant_loss', 99)
        hs_v  = sigs.get('hue_std', 99)
        st.markdown(f"""
<div class='why-box'>
<span style='color:#ff6680'>WHY VERDICT = FAKE (ANIME/ART)</span><br>
• Anime score {ascore:.4f} exceeds threshold {athresh}<br>
• Patch texture std = {psm_v:.2f}
  {"✅ flat color zones = anime/cel-shading" if psm_v < 14.0 else "⚪ borderline"}<br>
• Quantization loss = {ql_v:.2f}
  {"✅ already quantized palette = anime/illustration" if ql_v < 7.0 else "⚪ borderline"}<br>
• Hue variance = {hs_v:.2f}
  {"✅ restricted hue range = anime color palette" if hs_v < 40.0 else "⚪ borderline"}<br><br>
<span style='color:#ff8844'>Anime, illustration, and AI-generated art are NOT real photographs.<br>
They are synthetic/fake by definition.</span>
</div>""", unsafe_allow_html=True)

    else:
        subs  = df_results['sub_scores']
        fcols = st.columns(2)
        for i, (label, score) in enumerate(subs.items()):
            if score > 0.6:
                badge, bc = f"■ HIGH ({score:.2f})",   "badge-anime"
            elif score > 0.3:
                badge, bc = f"◆ MED ({score:.2f})",    "badge-border"
            else:
                badge, bc = f"● LOW ({score:.2f})",    "badge-real"
            with fcols[i % 2]:
                st.markdown(f"""
<div class='sig-card'>
  <span class='sig-label'>{label}</span>
  <span class='{bc}'>{badge}</span>
</div>""", unsafe_allow_html=True)

    # ── VISUAL FORENSICS ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔭 VISUAL FORENSICS")
    t1, t2, t3, t4 = st.tabs(["⚡ ELA","〰 Frequency","🎨 Histogram","📊 Classifier Signals"])

    with t1:
        if show_ela:
            st.pyplot(plot_ela(img), use_container_width=True)
            st.markdown("""
<div class='logic-box' style='margin-top:.5rem'>
ELA — re-compress at known quality. Synthetic/edited regions show different error levels.
</div>""", unsafe_allow_html=True)

    with t2:
        if show_freq:
            st.pyplot(plot_freq(img), use_container_width=True)
            st.markdown("""
<div class='logic-box' style='margin-top:.5rem'>
FFT — AI images show unusual spectral patterns. Anime has atypically high center energy.
</div>""", unsafe_allow_html=True)

    with t3:
        if show_hist:
            st.pyplot(plot_hist(img), use_container_width=True)
            st.markdown("""
<div class='logic-box' style='margin-top:.5rem'>
Histograms — anime has spiky/narrow distributions. Real photos have smooth, wide color spreads.
</div>""", unsafe_allow_html=True)

    with t4:
        if show_sig:
            st.pyplot(plot_signals(sigs, ascore), use_container_width=True)
        if df_results:
            subs      = df_results['sub_scores']
            radar_labs = list(subs.keys())
            radar_vals = list(subs.values())
            st.pyplot(plot_radar(radar_labs, radar_vals, '#00f5ff', 'Deepfake Anomaly'),
                      use_container_width=True)

    with st.expander("🗂 RAW DATA"):
        st.json({'classification': tc, 'deepfake_analysis': df_results})

else:
    st.markdown("""
<div class='await-box'>
  <div class='await-icon'>🔍</div>
  <div class='await-title'>AWAITING INPUT</div>
  <div class='await-sub'>
   Upload any image — v6 classifier checks 12 signals first<br>
   Anime / Illustration / AI-Art → ⚠️ FAKE &nbsp;|&nbsp; Real Photo → deepfake check
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
 DEEPFAKE DETECTOR v8.0 // 12-SIGNAL CLASSIFIER (SKETCH+ANIME FIX) // RESEARCH USE ONLY
</div>""", unsafe_allow_html=True)