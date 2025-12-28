import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import pywt
from PIL import Image
from PIL.ExifTags import TAGS
import io
import sqlite3
from datetime import datetime
import hashlib

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
    Module 3: Advanced Stats (Benford, CFA, Texture, Chromatic Aberration, Noise Kurtosis, ELA)
    """
    from scipy.stats import kurtosis
    stats_out = {}
    
    # --- 1. Benford's Law (DCT) ---
    im_f = img_gray.astype(float)
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
        stats_out['benford_div'] = np.sum((actual_dist - benford_dist)**2 / benford_dist)
        stats_out['benford_actual'] = actual_dist
        stats_out['benford_ideal'] = benford_dist
    else:
        stats_out['benford_div'] = 0
        stats_out['benford_actual'] = np.zeros(9)
        stats_out['benford_ideal'] = np.zeros(9)

    # --- 2. CFA Artifacts (Color Correlation) ---
    B, G, R = cv2.split(img_bgr)
    diff_GR = G.astype(float) - R.astype(float)
    diff_GB = G.astype(float) - B.astype(float)
    if np.std(diff_GR) > 0 and np.std(diff_GB) > 0:
        stats_out['cfa_score'] = np.corrcoef(diff_GR.flatten(), diff_GB.flatten())[0, 1]
    else:
        stats_out['cfa_score'] = 0

    # --- 3. Texture Analysis (GLCM) ---
    if img_gray.dtype != 'uint8':
        img_gray = (img_gray * 255).astype(np.uint8)
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    stats_out['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    stats_out['contrast'] = graycoprops(glcm, 'contrast')[0, 0]

    # --- 4. Chromatic Aberration (Lens Physics) ---
    h, w = img_gray.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask_center = (x - center_x)**2 + (y - center_y)**2 <= (min(h,w)//4)**2
    mask_edge = (x - center_x)**2 + (y - center_y)**2 >= (min(h,w)//2.5)**2
    diff_map = np.abs(diff_GR)
    mean_diff_center = np.mean(diff_map[mask_center]) + 1e-6
    mean_diff_edge = np.mean(diff_map[mask_edge]) + 1e-6
    stats_out['chromatic_aberration'] = mean_diff_edge / mean_diff_center

    # --- 5. Noise Residual Kurtosis ---
    blurred = cv2.GaussianBlur(img_gray, (3,3), 0)
    residual = img_gray.astype(float) - blurred.astype(float)
    stats_out['noise_kurtosis'] = kurtosis(residual.flatten())

    # --- 6. ELA (Error Level Analysis) ---
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    ela_diff = cv2.absdiff(img_bgr, decimg)
    stats_out['ela_mean'] = np.mean(ela_diff)

    return stats_out

def detect_grid_artifacts(power_spec):
    """
    Module 4: Detect GAN grid artifacts in frequency domain
    """
    rows, cols = power_spec.shape
    center_row, center_col = rows//2, cols//2
    
    # Check for horizontal/vertical lines in spectrum
    horz_line = power_spec[center_row, :]
    vert_line = power_spec[:, center_col]
    
    # Calculate periodicity using autocorrelation
    def check_periodicity(line):
        if len(line) < 10:
            return False, 0
        correlation = np.correlate(line, line, mode='full')
        half_idx = len(line)//2
        peaks, properties = find_peaks(correlation[half_idx:], height=np.mean(correlation)*1.5)
        return len(peaks) > 3, len(peaks)
    
    horz_periodic, horz_peaks = check_periodicity(horz_line)
    vert_periodic, vert_peaks = check_periodicity(vert_line)
    
    return {
        'has_grid_artifacts': horz_periodic or vert_periodic,
        'grid_score': (horz_peaks + vert_peaks) / 2.0
    }

def analyze_noise_consistency(img_bgr):
    """
    Module 5: Check for unnatural noise patterns in different color channels
    """
    # Split into channels
    B, G, R = cv2.split(img_bgr.astype(float))
    
    # Calculate noise variance in local patches
    def patch_noise_variance(channel, patch_size=32):
        h, w = channel.shape
        variances = []
        for i in range(0, h-patch_size, patch_size):
            for j in range(0, w-patch_size, patch_size):
                patch = channel[i:i+patch_size, j:j+patch_size]
                variances.append(np.var(patch))
        return np.array(variances)
    
    # Compare noise patterns across channels
    var_B = patch_noise_variance(B)
    var_G = patch_noise_variance(G)
    var_R = patch_noise_variance(R)
    
    # Real cameras have correlated noise across channels
    if len(var_B) > 1 and len(var_G) > 1 and len(var_R) > 1:
        corr_BG = np.corrcoef(var_B, var_G)[0,1] if np.std(var_B) > 0 and np.std(var_G) > 0 else 0
        corr_BR = np.corrcoef(var_B, var_R)[0,1] if np.std(var_B) > 0 and np.std(var_R) > 0 else 0
        corr_GR = np.corrcoef(var_G, var_R)[0,1] if np.std(var_G) > 0 and np.std(var_R) > 0 else 0
    else:
        corr_BG, corr_BR, corr_GR = 0, 0, 0
    
    return {
        'noise_correlation_avg': (corr_BG + corr_BR + corr_GR) / 3.0,
        'noise_inconsistency': np.std([corr_BG, corr_BR, corr_GR]),
        'noise_var_map': var_R  # For visualization
    }

def detect_color_space_anomalies(img_bgr):
    """
    Module 6: Check for unnatural color distributions
    """
    # Convert to LAB and HSV
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Check for unnatural a/b channel distributions in LAB
    a_channel = img_lab[:,:,1].astype(float) - 128  # Center around 0
    b_channel = img_lab[:,:,2].astype(float) - 128
    
    # Real images have certain distributions
    a_skew = stats.skew(a_channel.flatten())
    b_skew = stats.skew(b_channel.flatten())
    a_kurt = stats.kurtosis(a_channel.flatten())
    b_kurt = stats.kurtosis(b_channel.flatten())
    
    # Check for unnatural saturation/value distributions
    saturation = img_hsv[:,:,1].astype(float) / 255.0
    hist, _ = np.histogram(saturation.flatten(), bins=50)
    hist_normalized = hist / (np.sum(hist) + 1e-10)
    saturation_entropy = stats.entropy(hist_normalized + 1e-10)
    
    return {
        'lab_a_skew': abs(a_skew),
        'lab_b_skew': abs(b_skew),
        'lab_a_kurtosis': a_kurt,
        'lab_b_kurtosis': b_kurt,
        'saturation_entropy': saturation_entropy,
        'color_anomaly_score': (abs(a_skew) + abs(b_skew)) / 2.0
    }

def micro_texture_analysis(img_gray):
    """
    Module 7: Analyze fine texture patterns using wavelet transforms
    """
    # Ensure proper dtype
    if img_gray.dtype == np.uint8:
        img_gray = img_gray.astype(float)
    
    # Perform 2-level wavelet decomposition
    coeffs = pywt.wavedec2(img_gray, 'haar', level=2)
    
    # Extract detail coefficients (horizontal, vertical, diagonal)
    cH1, cV1, cD1 = coeffs[1]  # Level 1 details
    cH2, cV2, cD2 = coeffs[2]  # Level 2 details
    
    # Calculate kurtosis of detail coefficients
    # Real images have leptokurtic distributions (heavy tails)
    features = {}
    wavelet_kurtosis = []
    
    for name, coeff in [('H1', cH1), ('V1', cV1), ('D1', cD1), 
                        ('H2', cH2), ('V2', cV2), ('D2', cD2)]:
        data = coeff.flatten()
        kurt = stats.kurtosis(data)
        skew = stats.skew(data)
        features[f'kurtosis_{name}'] = kurt
        features[f'skew_{name}'] = skew
        wavelet_kurtosis.append(abs(kurt))
    
    # Check for unnatural symmetry (AI images often too symmetric)
    if np.std(cH1.flatten()) > 0 and np.std(cV1.flatten()) > 0:
        h_v_symmetry = np.corrcoef(cH1.flatten(), cV1.flatten())[0,1]
    else:
        h_v_symmetry = 0
        
    features['wavelet_symmetry'] = abs(h_v_symmetry)
    features['wavelet_kurtosis_avg'] = np.mean(wavelet_kurtosis)
    
    return features

def analyze_edge_profiles(img_gray, threshold=30):
    """
    Module 8: Analyze edge sharpness and consistency
    """
    # Ensure uint8
    if img_gray.dtype != np.uint8:
        img_gray = (img_gray * 255).clip(0, 255).astype(np.uint8)
    
    # Detect edges
    edges = cv2.Canny(img_gray, threshold, threshold*2)
    
    # Get edge coordinates
    edge_coords = np.column_stack(np.where(edges > 0))
    
    if len(edge_coords) < 100:  # Not enough edges
        return {
            'edge_sharpness': 0, 
            'edge_consistency': 0,
            'edge_uniformity': 0,
            'edge_count': len(edge_coords)
        }
    
    # Calculate gradient perpendicular to edges
    Gx = cv2.Sobel(img_gray.astype(float), cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img_gray.astype(float), cv2.CV_64F, 0, 1, ksize=3)
    
    edge_gradients = []
    sample_size = min(500, len(edge_coords))
    indices = np.random.choice(len(edge_coords), sample_size, replace=False)
    
    for idx in indices:
        y, x = edge_coords[idx]
        if 1 < y < img_gray.shape[0]-1 and 1 < x < img_gray.shape[1]-1:
            # Get gradient magnitude at edge
            mag = np.sqrt(Gx[y,x]**2 + Gy[y,x]**2)
            edge_gradients.append(mag)
    
    if len(edge_gradients) < 10:
        return {
            'edge_sharpness': 0, 
            'edge_consistency': 0,
            'edge_uniformity': 0,
            'edge_count': len(edge_coords)
        }
    
    # Analyze distribution
    edge_gradients = np.array(edge_gradients)
    median_grad = np.median(edge_gradients)
    mean_grad = np.mean(edge_gradients)
    
    return {
        'edge_sharpness': median_grad,
        'edge_consistency': np.std(edge_gradients) / (mean_grad + 1e-10),
        'edge_uniformity': 1.0 - (np.percentile(edge_gradients, 90) - np.percentile(edge_gradients, 10)) / (median_grad + 1e-10),
        'edge_count': len(edge_coords)
    }

def analyze_metadata(file_bytes):
    """
    Module 9: Extract and analyze EXIF metadata
    """
    metadata = {}
    
    try:
        image = Image.open(io.BytesIO(file_bytes))
        exifdata = image.getexif()
        
        raw_metadata = {}
        if exifdata:
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                raw_metadata[tag] = str(value)[:100]  # Limit length
                
        # Check for suspicious metadata
        software = str(raw_metadata.get('Software', '')).lower()
        ai_terms = ['dall', 'midjourney', 'stable', 'diffusion', 'adobe firefly', 
                    'imagen', 'flux', 'leonardo', 'playground']
        
        metadata_checks = {
            'has_software': 'Software' in raw_metadata,
            'has_camera_model': 'Model' in raw_metadata,
            'has_exif': len(raw_metadata) > 0,
            'software_is_ai': any(ai_term in software for ai_term in ai_terms),
            'metadata_count': len(raw_metadata),
            'has_gps': any('GPS' in key for key in raw_metadata.keys())
        }
        
        return {**metadata_checks, 'raw_metadata': raw_metadata}
        
    except Exception as e:
        return {
            'has_metadata': False, 
            'error': str(e),
            'has_exif': False,
            'metadata_count': 0
        }

class EnsembleDetector:
    """
    Module 10: Ensemble Decision System with weighted metrics
    """
    def __init__(self):
        self.metrics = []
        
    def add_metric(self, name, value, weight=1.0, threshold=None, higher_is_real=True):
        """Add metric to ensemble"""
        self.metrics.append({
            'name': name,
            'value': value,
            'weight': weight,
            'threshold': threshold,
            'higher_is_real': higher_is_real
        })
        
    def compute_score(self):
        """Compute weighted score (0-1 scale, higher = more real)"""
        total_score = 0
        total_weight = 0
        passed_checks = 0
        total_checks = 0
        
        for metric in self.metrics:
            val = metric['value']
            weight = metric['weight']
            
            if metric['threshold'] is not None:
                # Binary threshold check
                total_checks += 1
                if metric['higher_is_real']:
                    if val >= metric['threshold']:
                        total_score += weight
                        passed_checks += 1
                else:
                    if val <= metric['threshold']:
                        total_score += weight
                        passed_checks += 1
            else:
                # Continuous contribution (assume normalized 0-1)
                total_score += val * weight
                
            total_weight += weight
            
        confidence = total_score / total_weight if total_weight > 0 else 0
        
        return confidence, passed_checks, total_checks
    
    def get_detailed_report(self):
        """Generate detailed report"""
        report = []
        for metric in self.metrics:
            contribution = metric['value'] * metric['weight']
            report.append({
                'name': metric['name'],
                'value': metric['value'],
                'weight': metric['weight'],
                'contribution': contribution
            })
        return pd.DataFrame(report)

class DetectionDatabase:
    """
    Module 11: Database for tracking detections
    """
    def __init__(self, db_path="detections.db"):
        self.db_path = db_path
        self.conn = None
        self.create_tables()
        
    def create_tables(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    filename TEXT,
                    verdict TEXT,
                    confidence REAL,
                    image_hash TEXT,
                    metrics TEXT
                )
            ''')
            self.conn.commit()
        except Exception as e:
            st.sidebar.warning(f"DB init warning: {e}")
        
    def save_detection(self, filename, verdict, confidence, metrics_dict):
        try:
            if self.conn is None:
                return
            cursor = self.conn.cursor()
            img_hash = hashlib.md5(filename.encode()).hexdigest()
            cursor.execute('''
                INSERT INTO detections (timestamp, filename, verdict, confidence, image_hash, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), filename, verdict, confidence, img_hash, str(metrics_dict)))
            self.conn.commit()
        except Exception as e:
            pass  # Silent fail for demo
        
    def get_statistics(self):
        try:
            if self.conn is None:
                return []
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    verdict,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence
                FROM detections
                GROUP BY verdict
            ''')
            return cursor.fetchall()
        except:
            return []

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
    t_noise_corr = st.number_input("Min Noise Correlation", 0.3, 0.9, 0.6)
    t_grid_score = st.number_input("Max Grid Artifacts", 0.0, 10.0, 3.0)

st.sidebar.markdown("---")
with st.sidebar.expander("🗄️ Detection History", expanded=False):
    db = DetectionDatabase()
    detection_stats = db.get_statistics()
    if detection_stats:
        for verdict, count, avg_conf in detection_stats:
            st.write(f"**{verdict}**: {count} images (avg conf: {avg_conf:.2f})")
    else:
        st.info("No detections logged yet")

# --- MAIN APP ---

st.title("SynthDetect Ultra")
st.markdown("**Advanced Synthetic Image Forensics** | Gradient • Spectral • Statistical")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])

if uploaded:
    # Load
    file_bytes_array = uploaded.read()
    file_bytes = np.asarray(bytearray(file_bytes_array), dtype=np.uint8)
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
        # 1. Gradient Analysis
        rho, kappa, G_mag, _, _ = compute_gradient_pca(L)
        
        # 2. Spectral Analysis
        beta, eta, rad_prof, pow_spec, log_f, log_s, intercept, slope = compute_frequency_stats(L)
        
        # 3. Basic Advanced Stats
        adv_stats = compute_advanced_forensics(img, img_gray)
        
        # 4. Grid Artifacts Detection
        grid_stats = detect_grid_artifacts(pow_spec)
        
        # 5. Noise Consistency
        noise_stats = analyze_noise_consistency(img)
        
        # 6. Color Space Anomalies
        color_stats = detect_color_space_anomalies(img)
        
        # 7. Micro-texture Analysis
        wavelet_stats = micro_texture_analysis(img_gray)
        
        # 8. Edge Profile Analysis
        edge_stats = analyze_edge_profiles(img_gray)
        
        # 9. Metadata Analysis
        metadata_stats = analyze_metadata(file_bytes_array)

    # --- ENHANCED ENSEMBLE DECISION ENGINE ---
    ensemble = EnsembleDetector()
    flags = []
    
    # Add all metrics to ensemble (higher value = more real for most)
    # Core Metrics (higher weight)
    ensemble.add_metric("Gradient Anisotropy", rho, weight=2.0, threshold=t_rho, higher_is_real=True)
    beta_in_range = 1.0 if t_beta_min <= beta <= t_beta_max else 0.0
    ensemble.add_metric("Spectral Slope", beta_in_range, weight=2.0, threshold=0.5, higher_is_real=True)
    ensemble.add_metric("Benford Law", adv_stats['benford_div'], weight=1.5, threshold=t_benford, higher_is_real=False)
    ensemble.add_metric("CFA Score", adv_stats['cfa_score'], weight=1.5, threshold=t_cfa, higher_is_real=True)
    

    # New Advanced Metrics
    ensemble.add_metric("Noise Correlation", noise_stats['noise_correlation_avg'], weight=1.0, threshold=t_noise_corr, higher_is_real=True)
    ensemble.add_metric("Grid Artifacts", grid_stats['grid_score'], weight=1.5, threshold=t_grid_score, higher_is_real=False)
    ensemble.add_metric("Wavelet Kurtosis", wavelet_stats['wavelet_kurtosis_avg'], weight=1.0, threshold=3.0, higher_is_real=True)
    ensemble.add_metric("Edge Consistency", edge_stats['edge_consistency'], weight=0.8, threshold=2.0, higher_is_real=False)

    # Color anomalies (lower is more natural)
    color_anomaly_normalized = 1.0 / (1.0 + color_stats['color_anomaly_score'])
    ensemble.add_metric("Color Naturalness", color_anomaly_normalized, weight=0.8, threshold=0.4, higher_is_real=True)

    # --- NEW: Chromatic Aberration ---
    ensemble.add_metric("Chromatic Aberration", adv_stats.get('chromatic_aberration', 1.0), weight=1.2, threshold=1.05, higher_is_real=True)

    # --- NEW: Noise Kurtosis ---
    ensemble.add_metric("Noise Kurtosis", adv_stats.get('noise_kurtosis', 0.0), weight=1.0, threshold=1.0, higher_is_real=True)

    # --- NEW: ELA Mean ---
    ensemble.add_metric("ELA Mean", adv_stats.get('ela_mean', 0.0), weight=0.8, threshold=5.0, higher_is_real=True)

    # Metadata checks
    if metadata_stats.get('software_is_ai', False):
        ensemble.add_metric("Metadata Check", 0.0, weight=3.0, threshold=0.5, higher_is_real=True)
        flags.append("⚠️ AI Software Detected in EXIF")
    elif metadata_stats.get('has_camera_model', False):
        ensemble.add_metric("Metadata Check", 1.0, weight=1.0, threshold=0.5, higher_is_real=True)

    # Compute final score
    confidence, passed_checks, total_checks = ensemble.compute_score()

    # Generate flags based on failed checks
    if not (t_beta_min <= beta <= t_beta_max):
        flags.append(f"❌ Unnatural Spectral Slope (β={beta:.2f})")
    if rho <= t_rho:
        flags.append(f"❌ Isotropic Gradient Field (ρ={rho:.2f})")
    if adv_stats['benford_div'] >= t_benford:
        flags.append(f"❌ Violates Benford's Law (Div={adv_stats['benford_div']:.4f})")
    if adv_stats['cfa_score'] <= t_cfa:
        flags.append(f"❌ Low Color Correlation (CFA={adv_stats['cfa_score']:.2f})")
    if noise_stats['noise_correlation_avg'] < t_noise_corr:
        flags.append(f"❌ Inconsistent Noise Pattern ({noise_stats['noise_correlation_avg']:.2f})")
    if grid_stats['has_grid_artifacts']:
        flags.append(f"❌ GAN Grid Artifacts Detected (Score={grid_stats['grid_score']:.1f})")
    if edge_stats['edge_consistency'] > 2.0:
        flags.append(f"❌ Unnatural Edge Patterns")
    # Chromatic Aberration Check
    if adv_stats.get('chromatic_aberration', 1.0) <= 1.05:
        flags.append(f"No Lens Aberration Detected (Ratio={adv_stats.get('chromatic_aberration', 1.0):.2f})")
    # Noise Kurtosis Check
    if adv_stats.get('noise_kurtosis', 0.0) <= 1.0:
        flags.append(f"Gaussian/Artificial Noise Profile (Kurtosis={adv_stats.get('noise_kurtosis', 0.0):.2f})")

    # Compression Check
    is_compressed = eta < 0.01
    if is_compressed:
        flags.append("ℹ️ High Compression (May mask artifacts)")
        confidence *= 0.9  # Reduce confidence for compressed images

    # Final Verdict based on confidence score
    if confidence >= 0.70:
        verdict = "LIKELY REAL" if not is_compressed else "REAL (Compressed)"
        style = "v-real"
    elif confidence >= 0.50:
        verdict = "UNCERTAIN / HYBRID"
        style = "v-warn"
    else:
        verdict = "LIKELY SYNTHETIC"
        style = "v-fake"
    
    # Save to database
    db = DetectionDatabase()
    db.save_detection(uploaded.name, verdict, confidence, {
        'rho': rho, 'beta': beta, 'benford': adv_stats['benford_div'],
        'noise_corr': noise_stats['noise_correlation_avg'],
        'grid_score': grid_stats['grid_score']
    })

    # --- DASHBOARD ---
    
    col_img, col_res = st.columns([1, 1.5])
    
    with col_img:
        st.image(img_rgb, use_container_width=True, caption=f"Resolution: {img.shape[1]}x{img.shape[0]}")
        
    with col_res:
        st.markdown(f"""
        <div class="verdict-box {style}">
            <h2>{verdict}</h2>
            <p>Confidence: {confidence:.1%} | {passed_checks}/{total_checks} Checks Passed</p>
        </div>
        """, unsafe_allow_html=True)
        
        if flags:
            for flag in flags:
                if "❌" in flag:
                    st.error(flag)
                elif "⚠️" in flag:
                    st.warning(flag)
                else:
                    st.info(flag)
        else:
            st.success("✅ All consistency checks passed.")
            
        st.markdown("### 📊 Core Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anisotropy (ρ)", f"{rho:.2f}", delta="✓" if rho>t_rho else "✗")
        c2.metric("Beta Slope (β)", f"{beta:.2f}", delta="✓" if t_beta_min<=beta<=t_beta_max else "✗")
        c3.metric("Benford Div", f"{adv_stats['benford_div']:.4f}", delta="✓" if adv_stats['benford_div']<t_benford else "✗")
        c4.metric("CFA Score", f"{adv_stats['cfa_score']:.2f}", delta="✓" if adv_stats['cfa_score']>t_cfa else "✗")
        
        st.markdown("### 🔬 Advanced Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Noise Corr", f"{noise_stats['noise_correlation_avg']:.2f}", delta="✓" if noise_stats['noise_correlation_avg']>t_noise_corr else "✗")
        c2.metric("Grid Score", f"{grid_stats['grid_score']:.1f}", delta="✓" if grid_stats['grid_score']<t_grid_score else "✗")
        c3.metric("Wavelet Kurt", f"{wavelet_stats['wavelet_kurtosis_avg']:.2f}", delta="✓" if wavelet_stats['wavelet_kurtosis_avg']>3.0 else "✗")
        c4.metric("Edge Consist", f"{edge_stats['edge_consistency']:.2f}", delta="✓" if edge_stats['edge_consistency']<2.0 else "✗")
        # New metrics
        st.markdown("### 🧬 Deep Forensics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Chromatic Aberration", f"{adv_stats.get('chromatic_aberration', 1.0):.3f}", delta="✓" if adv_stats.get('chromatic_aberration', 1.0)>1.05 else "✗")
        c2.metric("Noise Kurtosis", f"{adv_stats.get('noise_kurtosis', 0.0):.2f}", delta="✓" if adv_stats.get('noise_kurtosis', 0.0)>1.0 else "✗")
        c3.metric("ELA Mean", f"{adv_stats.get('ela_mean', 0.0):.2f}")

    st.markdown("---")
    
    # --- VISUALIZATION TABS ---
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📐 Gradient Geometry", 
        "🌌 Spectral Physics", 
        "📊 Statistical Forensics",
        "🔬 Advanced Analysis",
        "📋 Detailed Report"
    ])
    
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
    
    with tab4:
        st.markdown("### 🎨 Noise & Color Analysis")
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**Noise Consistency Check**")
            st.metric("Channel Correlation", f"{noise_stats['noise_correlation_avg']:.3f}")
            st.metric("Inconsistency", f"{noise_stats['noise_inconsistency']:.3f}")
            st.caption("Real cameras have correlated noise patterns across RGB channels. AI images often lack this correlation.")
            
            # Noise variance heatmap
            if 'noise_var_map' in noise_stats and len(noise_stats['noise_var_map']) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                noise_map_2d = noise_stats['noise_var_map'].reshape(-1, 1)
                ax.hist(noise_stats['noise_var_map'], bins=30, color='purple', alpha=0.7)
                ax.set_title("Noise Variance Distribution")
                ax.set_xlabel("Variance")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
        
        with c2:
            st.write("**Color Space Anomalies**")
            st.metric("LAB a* Skewness", f"{color_stats['lab_a_skew']:.3f}")
            st.metric("LAB b* Skewness", f"{color_stats['lab_b_skew']:.3f}")
            st.metric("Saturation Entropy", f"{color_stats['saturation_entropy']:.3f}")
            st.caption("Natural images follow specific color distributions in LAB space. Deviations suggest synthesis.")
        
        st.markdown("---")
        st.markdown("### 🌊 Wavelet Texture Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Wavelet Coefficients Kurtosis**")
            wavelet_data = {
                'Level': ['H1', 'V1', 'D1', 'H2', 'V2', 'D2'],
                'Kurtosis': [
                    wavelet_stats.get('kurtosis_H1', 0),
                    wavelet_stats.get('kurtosis_V1', 0),
                    wavelet_stats.get('kurtosis_D1', 0),
                    wavelet_stats.get('kurtosis_H2', 0),
                    wavelet_stats.get('kurtosis_V2', 0),
                    wavelet_stats.get('kurtosis_D2', 0)
                ]
            }
            df_wavelet = pd.DataFrame(wavelet_data)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_wavelet['Level'], df_wavelet['Kurtosis'], color='teal')
            ax.axhline(y=3.0, color='r', linestyle='--', label='Natural threshold')
            ax.set_title("Wavelet Detail Coefficient Kurtosis")
            ax.set_ylabel("Kurtosis")
            ax.legend()
            st.pyplot(fig)
            
        with c2:
            st.write("**Edge Profile Analysis**")
            st.metric("Edge Sharpness", f"{edge_stats['edge_sharpness']:.2f}")
            st.metric("Edge Consistency", f"{edge_stats['edge_consistency']:.2f}")
            st.metric("Edge Uniformity", f"{edge_stats['edge_uniformity']:.2f}")
            st.metric("Total Edges", f"{edge_stats['edge_count']}")
            st.caption("AI images often have unnaturally consistent or sharp edges due to synthesis artifacts.")
        
        st.markdown("---")
        st.markdown("### 🔍 Metadata Analysis")
        
        if metadata_stats.get('has_exif', False):
            st.success(f"✓ EXIF data found ({metadata_stats.get('metadata_count', 0)} fields)")
            
            if metadata_stats.get('software_is_ai', False):
                st.error("⚠️ AI generation software detected in metadata!")
            elif metadata_stats.get('has_camera_model', False):
                st.info("✓ Camera model information present")
            
            if metadata_stats.get('has_gps', False):
                st.info("✓ GPS data present")
            
            with st.expander("Raw Metadata"):
                st.json(metadata_stats.get('raw_metadata', {}))
        else:
            st.warning("No EXIF metadata found (common for AI images or edited photos)")
    
    with tab5:
        st.markdown("### 📊 Comprehensive Detection Report")
        
        # Ensemble metrics breakdown
        st.write("#### Weighted Metric Contributions")
        report_df = ensemble.get_detailed_report()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if v > 0.5 else 'red' for v in report_df['value']]
        ax.barh(report_df['name'], report_df['contribution'], color=colors, alpha=0.6)
        ax.set_xlabel("Weighted Contribution")
        ax.set_title("Feature Importance in Detection Decision")
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        st.pyplot(fig)
        
        st.write("#### Detailed Metrics Table")
        st.dataframe(report_df.style.format({
            'value': '{:.4f}',
            'weight': '{:.2f}',
            'contribution': '{:.4f}'
        }), use_container_width=True)
        
        st.markdown("---")
        st.write("#### Radar Chart: Multi-Dimensional Analysis")
        
        # Create radar chart
        categories = ['Gradient', 'Spectral', 'Statistical', 'Noise', 'Color', 'Texture', 'Edges']
        values = [
            min(rho / 2.0, 1.0),  # Normalize
            beta_in_range,
            1.0 - min(adv_stats['benford_div'] / 0.01, 1.0),
            noise_stats['noise_correlation_avg'],
            color_anomaly_normalized,
            min(wavelet_stats['wavelet_kurtosis_avg'] / 10.0, 1.0),
            1.0 - min(edge_stats['edge_consistency'] / 3.0, 1.0)
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', label='Image')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f"Multi-Dimensional Authenticity Profile\nOverall Confidence: {confidence:.1%}", size=14, pad=20)
        ax.grid(True)
        st.pyplot(fig)
        
        st.markdown("---")
        st.write("#### Summary Statistics")
        
        summary_data = {
            'Category': ['Core Physics', 'Statistical', 'Advanced', 'Overall'],
            'Status': [
                '✓' if (rho > t_rho and beta_in_range) else '✗',
                '✓' if (adv_stats['benford_div'] < t_benford) else '✗',
                '✓' if (noise_stats['noise_correlation_avg'] > t_noise_corr and not grid_stats['has_grid_artifacts']) else '✗',
                '✓' if confidence >= 0.6 else '✗'
            ],
            'Confidence': [
                f"{((rho/2.0 + beta_in_range)/2):.1%}",
                f"{(1.0 - min(adv_stats['benford_div']/0.01, 1.0)):.1%}",
                f"{noise_stats['noise_correlation_avg']:.1%}",
                f"{confidence:.1%}"
            ]
        }
        
        st.table(pd.DataFrame(summary_data))
        
        st.markdown("---")
        st.info(f"""
        **Final Verdict: {verdict}**
        
        This image was analyzed using {total_checks} independent forensic tests across multiple domains:
        - ✓ Passed: {passed_checks} tests
        - ✗ Failed: {total_checks - passed_checks} tests
        - Overall Confidence: {confidence:.1%}
        
        **Recommendation:** {'This image appears to be authentic with natural physical and statistical properties.' if confidence >= 0.6 else 'This image exhibits characteristics consistent with AI synthesis or heavy manipulation.'}
        """)