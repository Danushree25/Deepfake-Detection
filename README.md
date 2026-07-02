# 🔍 DeepFake Detector — Forensic Image Analysis Tool

A **Streamlit-based deepfake detection app** using classical computer vision + ML heuristics.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🧠 Detection Methods (7 Modules)

| Module | Technique | What It Detects |
|--------|-----------|-----------------|
| **Noise Pattern** | Gaussian residual + ELA | Over-smoothed regions, re-compression artifacts |
| **Frequency Domain** | FFT magnitude + phase | GAN checkerboard artifacts, unnatural spectral energy |
| **Compression** | DCT block boundaries | JPEG artifacts, unnatural sharpness |
| **Color Statistics** | Channel histograms + correlation | Unnatural inter-channel correlations |
| **Edge Coherence** | Sobel + Canny | Inconsistent gradient directions |
| **Texture Uniformity** | Block std analysis | GAN-typical over-smooth texture |
| **Face Region** | Quadrant statistics | Compositing inconsistencies, face swap boundaries |

---

## 📊 ML Ensemble Scoring

Each module produces an `anomaly_score ∈ [0, 1]`. The final verdict uses a **weighted ensemble**:

```
score = 0.20×noise + 0.18×frequency + 0.12×compression +
        0.15×color + 0.15×edge + 0.10×texture + 0.10×face_region
```

- **score > threshold** → FAKE  
- **score ≤ threshold** → REAL  
- **Confidence** = |score − 0.5| × 200%

---

## ⚙️ Sidebar Settings

- **Detection Sensitivity** (0.3–0.9): Adjust false positive/negative tradeoff
- **Toggle** individual visualization modules

---

## 📁 Project Structure

```
deepfake_detector/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## ⚠️ Disclaimer

> This tool is for **research and educational purposes only**.  
> Results are probabilistic estimates based on image forensics heuristics.  
> Not suitable for legal proceedings. Always consult domain experts.

---

## 🔬 Visualization Tabs

1. **ELA Analysis** — Error Level Analysis maps showing re-compression differences
2. **Frequency Domain** — FFT magnitude and phase spectra
3. **Color Histogram** — RGB channel distributions
4. **Radar Chart** — Multi-dimensional anomaly score visualization
