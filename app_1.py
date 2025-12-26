import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SynthDetect Ultra | AI Forensics",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .verdict-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        font-family: 'Arial', sans-serif;
    }
    .v-real { background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; }
    .v-fake { background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7; }
    .v-warn { background-color: #fff3cd; color: #664d03; border: 1px solid #ffecb5; }
    h1, h2, h3 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# --- CORE MATH FUNCTIONS ---

def get_luminance(img_bgr):
    """Converts BGR to Luminance (BT.709)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    R, G, B = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return L, img_rgb

def compute_gradient_pca(L):
    """
    Module 1: Gradient Field Analysis
    Returns: Rho (Anisotropy), Kappa (Coherence)
    """
    Gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    G_mag = np.sqrt(Gx**2 + Gy**2)
    
    M = np.stack((Gx.flatten(), Gy.flatten()), axis=1)
    N = M.shape[0]
    C = (1/N) * np.dot(M.T, M)
    
    evals, _ = np.linalg.eigh(C)
    evals = evals[::-1]
    l1, l2 = max(evals[0], 1e-10), max(evals[1], 1e-10)
    
    rho = l1 / l2
    kappa = ((l1 - l2)/(l1 + l2))**2
    return rho, kappa, G_mag, l1, l2

def compute_frequency_stats(L, beta_trim=0.1):
    """
    Module 2: Frequency Domain Analysis
    Returns: Beta (Spectral Slope), Eta (HF Ratio)
    """
    rows, cols = L.shape
    F = np.fft.fft2(L)
    Fshift = np.fft.fftshift(F)
    power_spec = np.abs(Fshift)**2
    
    y, x = np.indices((rows, cols))
    center = np.array([rows//2, cols//2])
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    tbin = np.bincount(r.ravel(), power_spec.ravel())
    nr = np.bincount(r.ravel())
    rad_prof = tbin / np.maximum(nr, 1)
    
    # Beta Calculation
    freqs = np.arange(len(rad_prof))
    start = 2
    end = int(len(freqs) * (1 - beta_trim))
    if end > start + 10:
        log_f = np.log10(freqs[start:end])
        log_s = np.log10(rad_prof[start:end] + 1e-10)
        slope, intercept, _, _, _ = stats.linregress(log_f, log_s)
        beta = -slope
    else:
        beta, slope, intercept, log_f, log_s = 0, 0, 0, [], []

    # Eta Calculation
    cutoff = len(rad_prof) // 4
    total_E = np.sum(rad_prof[1:])
    hf_E = np.sum(rad_prof[cutoff:])
    eta = hf_E / total_E if total_E > 0 else 0
    
    return beta, eta, rad_prof, power_spec, log_f, log_s, intercept, slope

def compute_advanced_forensics(img_bgr, img_gray):
    """
    Module 3: Advanced Stats (Benford, CFA, Texture)
    """
    stats_out = {}
    
    # 1. Benford's Law (DCT)
    im_f = img_gray.astype(float)
    # Ensure even dimensions for DCT
    h, w = im_f.shape
    if h % 2 != 0:
        im_f = im_f[:-1, :]
    if w % 2 != 0:
        im_f = im_f[:, :-1]
    dct = cv2.dct(im_f)
    coeffs = np.abs(dct.flatten())
    coeffs = coeffs[coeffs > 0]
    
    first_digits = np.floor(coeffs / (10 ** np.floor(np.log10(coeffs)))).astype(int)
    counts = np.bincount(first_digits, minlength=10)[1:10]
    total = np.sum(counts)
    
    if total > 0:
        actual_dist = counts / total
        benford_dist = np.log10(1 + 1 / np.arange(1, 10))
        # KL Divergence-ish score
        stats_out['benford_div'] = np.sum((actual_dist - benford_dist)**2 / benford_dist)
        stats_out['benford_actual'] = actual_dist
        stats_out['benford_ideal'] = benford_dist
    else:
        stats_out['benford_div'] = 0
        stats_out['benford_actual'] = np.zeros(9)
        stats_out['benford_ideal'] = np.zeros(9)

    # 2. CFA Artifacts (Color Correlation)
    B, G, R = cv2.split(img_bgr)
    diff_GR = G.astype(float) - R.astype(float)
    diff_GB = G.astype(float) - B.astype(float)
    
    if np.std(diff_GR) > 0 and np.std(diff_GB) > 0:
        stats_out['cfa_score'] = np.corrcoef(diff_GR.flatten(), diff_GB.flatten())[0, 1]
    else:
        stats_out['cfa_score'] = 0

    # 3. Texture Analysis (GLCM)
    # Ensure uint8
    if img_gray.dtype != 'uint8':
        img_gray = (img_gray * 255).astype(np.uint8)
        
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    stats_out['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    stats_out['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    
    return stats_out

# --- SIDEBAR ---
st.sidebar.title("🎛️ Parameters")

with st.sidebar.expander("Analysis Settings", expanded=True):
    resize_opt = st.checkbox("Resize Large Images", value=True, help="Standardizes analysis to 1024px")
    
with st.sidebar.expander("Detection Thresholds", expanded=False):
    t_rho = st.slider("Min Anisotropy (Real)", 0.8, 1.5, 1.15)
    t_beta_min = st.number_input("Min Beta (Real)", 1.6)
    t_beta_max = st.number_input("Max Beta (Real)", 2.6)
    t_benford = st.number_input("Max Benford Div (Real)", 0.005, 0.02, 0.005, format="%.4f")
    t_cfa = st.number_input("Min CFA Score (Real)", 0.5, 0.95, 0.85)

# --- MAIN APP ---

st.title("SynthDetect Ultra")
st.markdown("**Advanced Synthetic Image Forensics** | Gradient • Spectral • Statistical")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])

if uploaded:
    # Load
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_raw = cv2.imdecode(file_bytes, 1)
    
    if resize_opt:
        h, w = img_raw.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img = cv2.resize(img_raw, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            img = img_raw
    else:
        img = img_raw

    # Process
    L, img_rgb = get_luminance(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    with st.spinner("Running Multi-Stage Forensics..."):
        # 1. Gradient
        rho, kappa, G_mag, _, _ = compute_gradient_pca(L)
        # 2. Spectral
        beta, eta, rad_prof, pow_spec, log_f, log_s, intercept, slope = compute_frequency_stats(L)
        # 3. Advanced
        adv_stats = compute_advanced_forensics(img, img_gray)

    # --- DECISION ENGINE ---
    score = 0
    flags = []
    
    # Physics Check
    if t_beta_min <= beta <= t_beta_max: 
        score += 1
    else:
        flags.append(f"Unnatural Spectral Slope (β={beta:.2f})")
        
    # Geometry Check
    if rho > t_rho: 
        score += 1
    else:
        flags.append(f"Isotropic Gradient Field (ρ={rho:.2f})")
        
    # Statistics Check
    if adv_stats['benford_div'] < t_benford:
        score += 1
    else:
        flags.append(f"Violates Benford's Law (Div={adv_stats['benford_div']:.4f})")
        
    # Sensor Check (CFA)
    if adv_stats['cfa_score'] > t_cfa:
        score += 1
    else:
        flags.append(f"Low Color Correlation (CFA={adv_stats['cfa_score']:.2f})")

    # Compression Check
    is_compressed = eta < 0.01
    if is_compressed:
        flags.append("High Compression Detected (May mask AI artifacts)")

    # Final Verdict
    if is_compressed and score >= 2:
        verdict = "REAL (Compressed)"
        style = "v-real"
    elif score >= 3:
        verdict = "LIKELY REAL"
        style = "v-real"
    elif score == 2:
        verdict = "UNCERTAIN / HYBRID"
        style = "v-warn"
    else:
        verdict = "LIKELY SYNTHETIC"
        style = "v-fake"

    # --- DASHBOARD ---
    
    col_img, col_res = st.columns([1, 1.5])
    
    with col_img:
        st.image(img_rgb, use_container_width=True, caption=f"Resolution: {img.shape[1]}x{img.shape[0]}")
        
    with col_res:
        st.markdown(f"""
        <div class="verdict-box {style}">
            <h2>{verdict}</h2>
            <p>Score: {score}/4 Passing Metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        if flags:
            st.warning("**Anomalies:** " + ", ".join(flags))
        else:
            st.success("Image consistency checks passed.")
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anisotropy", f"{rho:.2f}", delta="> 1.15" if rho>t_rho else "Low")
        c2.metric("Beta Slope", f"{beta:.2f}", delta="Normal" if t_beta_min<=beta<=t_beta_max else "Abnormal")
        c3.metric("Benford Div", f"{adv_stats['benford_div']:.4f}", delta="Low" if adv_stats['benford_div']<t_benford else "High", delta_color="inverse")
        c4.metric("CFA Score", f"{adv_stats['cfa_score']:.2f}", delta="High" if adv_stats['cfa_score']>t_cfa else "Low")

    st.markdown("---")
    
    # --- VISUALIZATION TABS ---
    
    tab1, tab2, tab3 = st.tabs(["📐 Gradient Geometry", "🌌 Spectral Physics", "📊 Statistical Forensics"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.imshow(G_mag, cmap='inferno', vmax=np.percentile(G_mag, 99))
            ax.axis('off')
            ax.set_title("Gradient Magnitude")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            # Viz logic
            minor = 1/rho
            el = plt.matplotlib.patches.Ellipse((0,0), 2, 2*minor, color='blue', alpha=0.4, label='Covariance')
            ax.add_artist(el)
            ax.add_artist(plt.Circle((0,0), 1, fill=False, ls='--', color='gray', label='Isotropic'))
            ax.legend()
            ax.set_title(f"PCA Shape (ρ={rho:.2f})")
            st.pyplot(fig)
            st.caption("Real images (Blue) usually stretch along a dominant light direction. AI images (Gray) are often circular.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.imshow(np.log10(pow_spec+1), cmap='jet')
            ax.axis('off')
            ax.set_title("2D Power Spectrum")
            st.pyplot(fig)
            st.caption("Look for grid-like artifacts (checkerboard patterns) indicating GAN/Diffusion upsampling.")
        with c2:
            fig, ax = plt.subplots()
            ax.scatter(log_f, log_s, s=1, alpha=0.3)
            ax.plot(log_f, intercept + slope*log_f, 'r--', label=f'Fit (β={beta:.2f})')
            ax.set_xlabel("Log Freq"); ax.set_ylabel("Log Power")
            ax.legend()
            st.pyplot(fig)

    with tab3:
        # Benford Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        digits = np.arange(1, 10)
        width = 0.35
        ax.bar(digits - width/2, adv_stats['benford_actual'], width, label='Image DCT', color='#3498db')
        ax.bar(digits + width/2, adv_stats['benford_ideal'], width, label='Benford Ideal', color='#95a5a6', alpha=0.6)
        ax.set_xticks(digits)
        ax.set_title(f"Benford's Law Analysis (Div: {adv_stats['benford_div']:.4f})")
        ax.legend()
        st.pyplot(fig)
        
        st.write("#### Texture Metrics (GLCM)")
        st.json({
            "Homogeneity": f"{adv_stats['homogeneity']:.4f} (AI often smoother/higher)",
            "Contrast": f"{adv_stats['contrast']:.4f}"
        })