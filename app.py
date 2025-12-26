import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
from PIL import Image
from io import BytesIO

# --- CORE ANALYSIS FUNCTIONS ---

def get_luminance(img_bgr):
    """
    Converts BGR image to Luminance using BT.709 coefficients.
    L = 0.2126*R + 0.7152*G + 0.0722*B
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return L, img_rgb

def process_image_gradients(pil_image):
    """
    Implements the 4-step logic from Tarek Rahman's post:
    1. RGB -> Luminance
    2. Compute Gradients (dL/dx, dL/dy)
    3. Treat pixels as 2D vectors
    4. Apply PCA
    """
    # Convert PIL image to OpenCV format (numpy array)
    img_array = np.array(pil_image)
    
    # 1. Convert RGB -> Luminance (Grayscale)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Normalize pixel values to 0-1 for numerical stability
    gray = gray.astype(np.float32) / 255.0

    # 2. Compute Spatial Gradients (dL/dx, dL/dy)
    # ksize=3 uses a standard 3x3 Sobel kernel
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # 3. Flatten and Stack to create (N_pixels, 2) dataset
    # Each row is a vector [dx, dy] describing the gradient at that pixel
    gradients = np.stack((grad_x.flatten(), grad_y.flatten()), axis=1)

    # 4. Apply PCA on the gradient covariance
    pca = PCA(n_components=2)
    
    # We sample a subset for PCA fitting if image is huge to keep it fast
    # (Natural image stats are consistent, so 50k pixels is usually enough)
    if gradients.shape[0] > 50000:
        indices = np.random.choice(gradients.shape[0], 50000, replace=False)
        pca.fit(gradients[indices])
    else:
        pca.fit(gradients)
        
    return pca, gradients, grad_x, grad_y

def compute_gradient_features(L):
    """
    Calculates features based on Gradient PCA.
    Returns: Anisotropy (rho), Coherence (kappa), and gradient components
    """
    # Gradientes Sobel
    Gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    
    # Aplanar
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    M = np.stack((Gx_flat, Gy_flat), axis=1)
    
    # Matriz de Covarianza
    N = M.shape[0]
    C = (1/N) * np.dot(M.T, M)
    
    # Valores Propios
    eigenvalues, _ = np.linalg.eigh(C)
    # Ordenar descendente
    eigenvalues = eigenvalues[::-1]
    l1, l2 = eigenvalues[0], eigenvalues[1]
    
    # Evitar división por cero
    l2 = max(l2, 1e-10)
    l1 = max(l1, 1e-10)
    
    # Características
    rho = l1 / l2  # Ratio de Anisotropía
    kappa = ((l1 - l2) / (l1 + l2))**2 # Coherencia
    energy = l1 + l2  # Gradient Energy
    
    return rho, kappa, energy, l1, l2, Gx, Gy

def compute_frequency_features(L):
    """
    Calculates features in the Frequency Domain.
    Returns: Spectral Slope (beta), High Frequency Ratio (eta)
    """
    rows, cols = L.shape
    
    # FFT (Transformada Rápida de Fourier)
    F = np.fft.fft2(L)
    Fshift = np.fft.fftshift(F)
    magnitude_spectrum = np.abs(Fshift)
    power_spectrum = magnitude_spectrum**2
    
    # Perfil Radial
    center_x, center_y = cols // 2, rows // 2
    y, x = np.ogrid[-center_y:rows-center_y, -center_x:cols-center_x]
    r = np.sqrt(x*x + y*y)
    r = r.astype(int)
    
    # Suma de potencia por radio
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    
    # 1. Pendiente Espectral (beta)
    r_axis = np.arange(len(radial_profile))
    mask = (r_axis > 0) & (r_axis < min(rows, cols)//2)
    
    log_r = np.log10(r_axis[mask])
    log_S = np.log10(radial_profile[mask] + 1e-10)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_S)
    beta = -slope
    
    # 2. Ratio de Alta Frecuencia (eta)
    cutoff = min(rows, cols) / 4
    
    total_energy = np.sum(radial_profile[1:])
    high_freq_energy = np.sum(radial_profile[r_axis > cutoff])
    
    if total_energy > 0:
        eta = high_freq_energy / total_energy
    else:
        eta = 0.0
    
    return beta, eta, radial_profile, log_r, log_S, slope, intercept, power_spectrum

def analyze_image_comprehensive(img, beta_min=1.6, beta_max=2.6, rho_threshold=1.15, 
                                eta_compressed=0.01, eta_noise=0.25):
    """Comprehensive image analysis combining all features"""
    
    # Preprocess: Resize to standard size (max 1024px)
    height, width = img.shape[:2]
    max_dim = 1024
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    L, img_rgb = get_luminance(img)
    
    # Extract Features
    rho, kappa, energy, l1, l2, Gx, Gy = compute_gradient_features(L)
    beta, eta, radial_prof, log_r, log_S, slope, intercept, power_spectrum = compute_frequency_features(L)
    
    # Decision Logic (detect_ai.py style)
    is_natural_physics = beta_min <= beta <= beta_max
    is_anisotropic = rho > rho_threshold
    is_compressed = eta < eta_compressed
    is_high_freq_noise = eta > eta_noise

    score = 0
    reasons = []
    if is_natural_physics:
        score += 1
    else:
        reasons.append("Spectral Slope (Beta) deviates from natural light physics.")
    if is_anisotropic:
        score += 1
    else:
        reasons.append("Gradient field is Isotropic (lacks directional coherence).")
    if is_high_freq_noise:
        score -= 10
        reasons.append("High-frequency energy is excessive (Diffusion artifacts).")

    # Label mapping (detect_ai.py style)
    if is_compressed and is_natural_physics:
        verdict_text = "REAL (Compressed)"
        verdict_reason = "Light physics (Beta) is correct. The lack of fine details is likely due to compression, not AI generation."
    elif score == 2:
        verdict_text = "LIKELY REAL"
        verdict_reason = "Image passes all physical and structural tests."
    elif score == 1:
        verdict_text = "UNCERTAIN / HYBRID"
        verdict_reason = "Image is ambiguous: passes some but not all natural image tests."
    else:
        verdict_text = "LIKELY SYNTHETIC (AI)"
        verdict_reason = "Image fails key natural image tests. " + (" ".join(reasons) if reasons else "")

    return {
        'verdict': verdict_text,
        'reason': verdict_reason,
        'beta': beta,
        'rho': rho,
        'eta': eta,
        'kappa': kappa,
        'energy': energy,
        'l1': l1,
        'l2': l2,
        'is_natural_physics': is_natural_physics,
        'is_anisotropic': is_anisotropic,
        'is_compressed': is_compressed,
        'is_high_freq_noise': is_high_freq_noise,
        'img_rgb': img_rgb,
        'L': L,
        'Gx': Gx,
        'Gy': Gy,
        'log_r': log_r,
        'log_S': log_S,
        'intercept': intercept,
        'slope': slope,
        'power_spectrum': power_spectrum,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'rho_threshold': rho_threshold,
        'eta_compressed': eta_compressed,
        'eta_noise': eta_noise
    }

def plot_gradient_distribution(gradients, title, color):
    """
    Helper to create a scatter plot of the gradients
    """
    # Sample points for plotting so the chart doesn't crash browser
    if gradients.shape[0] > 5000:
        indices = np.random.choice(gradients.shape[0], 5000, replace=False)
        sample = gradients[indices]
    else:
        sample = gradients
        
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sample[:, 0], sample[:, 1], alpha=0.1, s=1, c=color)
    ax.set_xlim([-0.5, 0.5]) # Limit view to center to see the structure
    ax.set_ylim([-0.5, 0.5])
    ax.set_title(title)
    ax.set_xlabel("Gradient X (dL/dx)")
    ax.set_ylabel("Gradient Y (dL/dy)")
    ax.grid(True, alpha=0.3)
    return fig

def create_comprehensive_visualizations(results):
    """Create comprehensive analysis visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Original
    axes[0, 0].imshow(results['img_rgb'])
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # 2. Gradient Magnitude
    G_mag = np.sqrt(results['Gx']**2 + results['Gy']**2)
    im2 = axes[0, 1].imshow(G_mag, cmap='inferno')
    axes[0, 1].set_title(f"Gradients (Anisotropy: {results['rho']:.2f})")
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Power Spectrum (Log)
    power_log = np.log10(np.fft.fftshift(results['power_spectrum']) + 1)
    im3 = axes[1, 0].imshow(power_log, cmap='jet')
    axes[1, 0].set_title("Power Spectrum (Log)")
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 4. Radial Profile (Log-Log)
    axes[1, 1].plot(results['log_r'], results['log_S'], label='Data', linewidth=2, alpha=0.7)
    axes[1, 1].plot(results['log_r'], results['intercept'] - results['beta']*results['log_r'], 
                    'r--', label=f'Fit (beta={results["beta"]:.2f})', linewidth=2)
    axes[1, 1].set_xlabel("Log Frequency")
    axes[1, 1].set_ylabel("Log Power")
    axes[1, 1].set_title("Radial Profile (Spectral Slope)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    return fig

def create_pca_ellipse_plot(rho):
    """Create PCA ellipse visualization"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    
    scale = 0.8
    ellipse = plt.matplotlib.patches.Ellipse(
        (0, 0), 
        width=scale * 1.0, 
        height=scale * (1/rho), 
        color='blue', 
        alpha=0.3,
        label='Gradient Distribution'
    )
    ax.add_patch(ellipse)
    ax.legend()
    ax.set_title(f"PCA Ellipse (ρ = {rho:.2f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig

def create_downloadable_report(results, filename="report"):
    """Create comprehensive report image for download"""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14), dpi=300)
    gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.3)
    
    # Title and Verdict
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    verdict_color = {
        "REAL": "green",
        "REAL (Compressed)": "green",
        "REAL (High Texture)": "green",
        "AI-GENERATED": "red",
        "SUSPICIOUS / POSSIBLE AI": "orange"
    }
    
    color = verdict_color.get(results['verdict'], "gray")
    
    ax_title.text(0.5, 0.7, f"AI Detection Report", 
                  ha='center', va='center', fontsize=28, weight='bold')
    ax_title.text(0.5, 0.4, f"Verdict: {results['verdict']}", 
                  ha='center', va='center', fontsize=22, weight='bold', color=color)
    ax_title.text(0.5, 0.1, results['reason'], 
                  ha='center', va='center', fontsize=14, style='italic', wrap=True)
    
    # Original Image
    ax1 = fig.add_subplot(gs[1, 0:2])
    ax1.imshow(results['img_rgb'])
    ax1.set_title("Original Image", fontsize=14, weight='bold')
    ax1.axis('off')
    
    # Metrics Box
    ax_metrics = fig.add_subplot(gs[1, 2:])
    ax_metrics.axis('off')
    ax_metrics.set_ylim(0, 1)
    # Get thresholds from results if available, otherwise use defaults
    beta_range = f"{results.get('beta_min', 1.5):.1f} - {results.get('beta_max', 2.8):.1f}"
    rho_thresh = results.get('rho_threshold', 1.10)
    
    metrics_text = f"""KEY METRICS
{'='*40}

Beta (Spectral Slope): {results['beta']:.4f}
Range: {beta_range} (Natural)
Status: {'✓ PASS' if results['is_natural_physics'] else '✗ FAIL'}

Rho (Anisotropy): {results['rho']:.4f}
Threshold: > {rho_thresh:.2f}
Status: {'✓ PASS' if results['is_anisotropic'] else '✗ FAIL'}

Eta (HF Ratio): {results['eta']:.4f}
Status: {'Compressed' if results['is_compressed'] else ('High Noise' if results['is_high_freq_noise'] else 'Normal')}

Kappa (Coherence): {results['kappa']:.4f}
Energy: {results['energy']:.4f}

Eigenvalue λ1: {results['l1']:.6f}
Eigenvalue λ2: {results['l2']:.6f}"""
    ax_metrics.text(0.1, 0.90, metrics_text, 
                    fontsize=10, verticalalignment='top', 
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Gradient Magnitude
    ax2 = fig.add_subplot(gs[2, 0])
    G_mag = np.sqrt(results['Gx']**2 + results['Gy']**2)
    im2 = ax2.imshow(G_mag, cmap='inferno')
    ax2.set_title(f"Gradient Magnitude", fontsize=12, weight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Power Spectrum
    ax3 = fig.add_subplot(gs[2, 1])
    power_log = np.log10(np.fft.fftshift(results['power_spectrum']) + 1)
    im3 = ax3.imshow(power_log, cmap='jet')
    ax3.set_title("Power Spectrum (Log)", fontsize=12, weight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Radial Profile
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.plot(results['log_r'], results['log_S'], 'b-', label='Data', linewidth=2, alpha=0.7)
    ax4.plot(results['log_r'], results['intercept'] - results['beta']*results['log_r'], 
             'r--', label=f'Fit (β={results["beta"]:.2f})', linewidth=2)
    ax4.set_xlabel("Log Frequency", fontsize=11)
    ax4.set_ylabel("Log Power", fontsize=11)
    ax4.set_title("Radial Profile (Spectral Slope)", fontsize=12, weight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, which="both", ls="-", alpha=0.3)
    
    # PCA Ellipse
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.set_xlim(-1, 1)
    ax5.set_ylim(-1, 1)
    ax5.set_aspect('equal')
    scale = 0.8
    ellipse = plt.matplotlib.patches.Ellipse(
        (0, 0), width=scale * 1.0, height=scale * (1/results['rho']), 
        color='blue', alpha=0.3, label='Gradient Distribution'
    )
    ax5.add_patch(ellipse)
    ax5.legend(fontsize=9)
    ax5.set_title(f"PCA Ellipse (ρ={results['rho']:.2f})", fontsize=12, weight='bold')
    ax5.set_xlabel("PC1", fontsize=10)
    ax5.set_ylabel("PC2", fontsize=10)
    ax5.grid(True, linestyle='--', alpha=0.3)
    
    # Physical Tests Summary
    ax6 = fig.add_subplot(gs[3, 1:])
    ax6.axis('off')
    
    tests_summary = f"""
    PHYSICAL TESTS SUMMARY
    {'='*50}
    
    ✓ Light Physics (Beta):  {'PASS ✓' if results['is_natural_physics'] else 'FAIL ✗'}
       Natural images follow ~1/f² law (Beta ≈ 2.0)
       Measured: {results['beta']:.3f}
    
    ✓ Gradient Structure (Rho):  {'PASS ✓' if results['is_anisotropic'] else 'FAIL ✗'}
       Real images have directional gradients (Rho > 1.15)
       Measured: {results['rho']:.3f}
    
    ✓ Frequency Analysis (Eta):  {'Normal' if not results['is_compressed'] and not results['is_high_freq_noise'] else ('Compressed' if results['is_compressed'] else 'High Noise')}
       Detects compression artifacts or diffusion noise
       Measured: {results['eta']:.3f}
    
    CONCLUSION:
    {results['verdict']}
    {results['reason']}
    """
    
    ax6.text(0.05, 0.95, tests_summary, 
             fontsize=10, verticalalignment='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Convert to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def create_downloadable_comparison_report(results1, results2, var1, var2, filename="comparison_report"):
    """Create comprehensive comparison report for two images"""
    
    fig = plt.figure(figsize=(24, 16), dpi=300)
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
    
    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, "Comparative AI Detection Report", 
                  ha='center', va='center', fontsize=32, weight='bold')
    
    verdict_color = {
        "REAL": "green",
        "REAL (Compressed)": "green",
        "REAL (High Texture)": "green",
        "AI-GENERATED": "red",
        "SUSPICIOUS / POSSIBLE AI": "orange"
    }
    
    # Image 1 Section
    ax_img1 = fig.add_subplot(gs[1, 0:2])
    ax_img1.imshow(results1['img_rgb'])
    color1 = verdict_color.get(results1['verdict'], "gray")
    ax_img1.set_title(f"IMAGE 1: {results1['verdict']}", fontsize=16, weight='bold', color=color1)
    ax_img1.axis('off')
    
    # Image 2 Section
    ax_img2 = fig.add_subplot(gs[1, 2:])
    ax_img2.imshow(results2['img_rgb'])
    color2 = verdict_color.get(results2['verdict'], "gray")
    ax_img2.set_title(f"IMAGE 2: {results2['verdict']}", fontsize=16, weight='bold', color=color2)
    ax_img2.axis('off')
    
    # Metrics Comparison
    ax_metrics = fig.add_subplot(gs[2, :])
    ax_metrics.axis('off')
    
    metrics_table = f"""
    {'METRIC':<25} | {'IMAGE 1':<20} | {'IMAGE 2':<20} | {'INTERPRETATION'}
    {'-'*110}
    Beta (Spectral Slope)     | {results1['beta']:>18.4f} | {results2['beta']:>18.4f} | Natural: 1.6-2.6
    Rho (Anisotropy)          | {results1['rho']:>18.4f} | {results2['rho']:>18.4f} | Real: > 1.15
    Eta (HF Ratio)            | {results1['eta']:>18.4f} | {results2['eta']:>18.4f} | Compressed: <0.01, Noisy: >0.25
    Kappa (Coherence)         | {results1['kappa']:>18.4f} | {results2['kappa']:>18.4f} | Higher = More coherent
    Energy                    | {results1['energy']:>18.4f} | {results2['energy']:>18.4f} | Total gradient energy
    PCA Variance Ratio        | {var1[0]/var1[1]:>18.2f} | {var2[0]/var2[1]:>18.2f} | Higher = More directional
    """
    
    ax_metrics.text(0.05, 0.95, metrics_table, 
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Gradient Magnitude - Image 1
    ax_g1 = fig.add_subplot(gs[3, 0])
    G_mag1 = np.sqrt(results1['Gx']**2 + results1['Gy']**2)
    im_g1 = ax_g1.imshow(G_mag1, cmap='inferno')
    ax_g1.set_title("Image 1: Gradients", fontsize=12, weight='bold')
    ax_g1.axis('off')
    plt.colorbar(im_g1, ax=ax_g1, fraction=0.046, pad=0.04)
    
    # Gradient Magnitude - Image 2
    ax_g2 = fig.add_subplot(gs[3, 1])
    G_mag2 = np.sqrt(results2['Gx']**2 + results2['Gy']**2)
    im_g2 = ax_g2.imshow(G_mag2, cmap='inferno')
    ax_g2.set_title("Image 2: Gradients", fontsize=12, weight='bold')
    ax_g2.axis('off')
    plt.colorbar(im_g2, ax=ax_g2, fraction=0.046, pad=0.04)
    
    # Power Spectrum - Image 1
    ax_p1 = fig.add_subplot(gs[3, 2])
    power_log1 = np.log10(np.fft.fftshift(results1['power_spectrum']) + 1)
    im_p1 = ax_p1.imshow(power_log1, cmap='jet')
    ax_p1.set_title("Image 1: Power Spectrum", fontsize=12, weight='bold')
    ax_p1.axis('off')
    plt.colorbar(im_p1, ax=ax_p1, fraction=0.046, pad=0.04)
    
    # Power Spectrum - Image 2
    ax_p2 = fig.add_subplot(gs[3, 3])
    power_log2 = np.log10(np.fft.fftshift(results2['power_spectrum']) + 1)
    im_p2 = ax_p2.imshow(power_log2, cmap='jet')
    ax_p2.set_title("Image 2: Power Spectrum", fontsize=12, weight='bold')
    ax_p2.axis('off')
    plt.colorbar(im_p2, ax=ax_p2, fraction=0.046, pad=0.04)
    
    # Radial Profile - Image 1
    ax_r1 = fig.add_subplot(gs[4, 0:2])
    ax_r1.plot(results1['log_r'], results1['log_S'], 'b-', label='Data', linewidth=2)
    ax_r1.plot(results1['log_r'], results1['intercept'] - results1['beta']*results1['log_r'], 
               'r--', label=f'Fit (β={results1["beta"]:.2f})', linewidth=2)
    ax_r1.set_xlabel("Log Frequency")
    ax_r1.set_ylabel("Log Power")
    ax_r1.set_title("Image 1: Radial Profile", fontsize=12, weight='bold')
    ax_r1.legend()
    ax_r1.grid(True, alpha=0.3)
    
    # Radial Profile - Image 2
    ax_r2 = fig.add_subplot(gs[4, 2:])
    ax_r2.plot(results2['log_r'], results2['log_S'], 'b-', label='Data', linewidth=2)
    ax_r2.plot(results2['log_r'], results2['intercept'] - results2['beta']*results2['log_r'], 
               'r--', label=f'Fit (β={results2["beta"]:.2f})', linewidth=2)
    ax_r2.set_xlabel("Log Frequency")
    ax_r2.set_ylabel("Log Power")
    ax_r2.set_title("Image 2: Radial Profile", fontsize=12, weight='bold')
    ax_r2.legend()
    ax_r2.grid(True, alpha=0.3)
    
    # Convert to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# --- STREAMLIT UI ---

st.set_page_config(page_title="Gradient Field Forensics - Complete Analysis", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("Gradient Field Forensics - Complete Analysis")
st.markdown("""
**Comprehensive AI Image Detection using Physics & Math.** This tool combines multiple methods:
* **Gradient PCA Analysis** - Analyzes the "Luminance Gradients" structure
* **Frequency Domain Analysis** - Examines spectral characteristics and 1/f law
* **Physical Consistency** - Verifies if the image follows natural light physics

Choose your analysis mode below:
""")

st.sidebar.title("Configuration")
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Single Image Analysis", "Comparison Mode (2 Images)"],
    help="Choose between analyzing a single image in detail or comparing two images"
)

resize_opt = st.sidebar.checkbox("Resize large images", value=True, help="Resizes to 1024px max dimension for consistent analysis")

st.sidebar.markdown("---")
st.sidebar.header("Detection Parameters")
st.sidebar.markdown("Adjust thresholds to fine-tune detection sensitivity:")

# Beta (Spectral Slope) parameters
st.sidebar.subheader("Beta (Spectral Slope)")
beta_min = st.sidebar.slider("Beta Min", 1.0, 2.0, 1.6, 0.1, 
                              help="Minimum beta for natural images")
beta_max = st.sidebar.slider("Beta Max", 2.0, 3.5, 2.6, 0.1,
                              help="Maximum beta for natural images")

# Rho (Anisotropy) threshold
st.sidebar.subheader("Rho (Anisotropy)")
rho_threshold = st.sidebar.slider("Rho Threshold", 1.0, 1.5, 1.15, 0.05,
                                   help="Minimum rho for directional gradients")

# Eta (High-Frequency Ratio) thresholds
st.sidebar.subheader("Eta (HF Ratio)")
eta_compressed = st.sidebar.slider("Compression Threshold", 0.001, 0.05, 0.01, 0.001,
                                    help="Max eta for compressed images")
eta_noise = st.sidebar.slider("Noise Threshold", 0.15, 0.50, 0.25, 0.05,
                               help="Min eta for high-frequency noise")

st.sidebar.markdown("---")
st.sidebar.header("Reference Ranges")
st.sidebar.markdown(f"""
**Current Settings:**
- Beta: {beta_min} - {beta_max}
- Rho: > {rho_threshold}
- Eta: {eta_compressed} - {eta_noise}

**Defaults:**
- Beta: 1.6 - 2.6
- Rho: > 1.15
- Eta: 0.01 - 0.25
""")



 # --- SINGLE IMAGE ANALYSIS MODE ---
if analysis_mode == "Single Image Analysis":
    st.header("Single Image Deep Analysis")
    
    uploaded_file = st.file_uploader("Upload image for analysis", type=["jpg", "png", "jpeg", "webp"])
    
    if uploaded_file is not None:
        # Load Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            st.error("Error loading image.")
        else:
            # Preprocess if needed
            height, width = img_bgr.shape[:2]
            if resize_opt and max(height, width) > 1024:
                scale = 1024 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
                st.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            
            # Analyze
            with st.spinner("Performing comprehensive analysis..."):
                results = analyze_image_comprehensive(img_bgr)
            
            # Display Results
            col_img, col_verdict = st.columns([1, 1])
            
            with col_img:
                st.image(results['img_rgb'], caption=uploaded_file.name, use_container_width=True)
            
            with col_verdict:
                st.markdown("### Verdict")
                
                verdict_color = {
                    "REAL": "green",
                    "REAL (Compressed)": "green",
                    "REAL (High Texture)": "green",
                    "AI-GENERATED": "red",
                    "SUSPICIOUS / POSSIBLE AI": "orange"
                }
                
                color = verdict_color.get(results['verdict'], "gray")
                
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="margin:0; color: white;">{results['verdict']}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(results['reason'])
                
                # Download Report Button
                st.markdown("---")
                report_buffer = create_downloadable_report(results, uploaded_file.name)
                st.download_button(
                    label="Download Complete Report (PNG - 300 DPI)",
                    data=report_buffer,
                    file_name=f"ai_detection_report_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric(
                    label="Beta (Spectral Slope)",
                    value=f"{results['beta']:.3f}",
                    delta="Natural" if results['is_natural_physics'] else "Abnormal",
                    delta_color="normal" if results['is_natural_physics'] else "inverse",
                    help="Natural images follow ~1/f^2 law (Beta≈2)"
                )
            
            with m2:
                st.metric(
                    label="Rho (Anisotropy)",
                    value=f"{results['rho']:.3f}",
                    delta="Coherent" if results['is_anisotropic'] else "Low",
                    delta_color="normal" if results['is_anisotropic'] else "inverse",
                    help="Ratio of eigenvalues. Real images have directional gradients"
                )
            
            with m3:
                st.metric(
                    label="Eta (HF Ratio)",
                    value=f"{results['eta']:.3f}",
                    delta="Compressed" if results['is_compressed'] else ("Noisy" if results['is_high_freq_noise'] else "Normal"),
                    delta_color="inverse" if results['is_high_freq_noise'] else "off",
                    help="High frequency energy ratio"
                )
            
            with m4:
                st.metric(
                    label="Energy",
                    value=f"{results['energy']:.3f}",
                    help="Total gradient energy"
                )
            
            # Tabbed Visualizations
            st.markdown("### Detailed Visualizations")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Comprehensive View", "Gradient Analysis", "Frequency Analysis", "Raw Data"])
            
            with tab1:
                fig_comp = create_comprehensive_visualizations(results)
                st.pyplot(fig_comp)
                plt.close(fig_comp)
            
            with tab2:
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    fig_g, ax_g = plt.subplots(figsize=(6, 6))
                    G_mag = np.sqrt(results['Gx']**2 + results['Gy']**2)
                    im = ax_g.imshow(G_mag, cmap='inferno')
                    ax_g.set_title(f"Gradient Magnitude (Energy={results['energy']:.2f})")
                    ax_g.axis('off')
                    plt.colorbar(im, ax=ax_g, fraction=0.046, pad=0.04)
                    st.pyplot(fig_g)
                    plt.close(fig_g)
                    st.caption("Real images usually have sparse, connected edges")
                
                with col_g2:
                    fig_ellipse = create_pca_ellipse_plot(results['rho'])
                    st.pyplot(fig_ellipse)
                    plt.close(fig_ellipse)
                    st.caption(f"Flatter ellipse (ρ={results['rho']:.2f}) indicates directional lighting")
            
            with tab3:
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    fig_s, ax_s = plt.subplots(figsize=(6, 6))
                    power_log = np.log10(np.fft.fftshift(results['power_spectrum']) + 1)
                    im_s = ax_s.imshow(power_log, cmap='jet')
                    ax_s.set_title("Log Power Spectrum")
                    ax_s.axis('off')
                    plt.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
                    st.pyplot(fig_s)
                    plt.close(fig_s)
                    st.caption("Look for grid artifacts (checkerboard patterns)")
                
                with col_f2:
                    fig_p, ax_p = plt.subplots(figsize=(6, 6))
                    ax_p.plot(results['log_r'], results['log_S'], label='Radial Data', alpha=0.7, linewidth=2)
                    fit_line = results['intercept'] - results['beta'] * results['log_r']
                    ax_p.plot(results['log_r'], fit_line, 'r--', label=f'Fit (β={results["beta"]:.2f})', linewidth=2)
                    ax_p.set_xlabel("Log Frequency")
                    ax_p.set_ylabel("Log Power")
                    ax_p.set_title("Radial Profile & Spectral Slope")
                    ax_p.legend()
                    ax_p.grid(True, which="both", linestyle='--', alpha=0.5)
                    st.pyplot(fig_p)
                    plt.close(fig_p)
                    st.caption("Linear fit indicates 1/f scaling")
            
            with tab4:
                st.write("#### Mathematical Details")
                st.json({
                    "Eigenvalue λ1": f"{results['l1']:.6f}",
                    "Eigenvalue λ2": f"{results['l2']:.6f}",
                    "Coherence κ": f"{results['kappa']:.6f}",
                    "Spectral Intercept": f"{results['intercept']:.4f}",
                    "Spectral Slope": f"{results['slope']:.4f}",
                    "Image Dimensions": f"{img_bgr.shape}"
                })
                
                st.markdown("#### Physical Interpretation")
                st.write(f"**✓ Light Physics:** {'PASS' if results['is_natural_physics'] else 'FAIL'}")
                st.write(f"**✓ Gradient Structure:** {'PASS' if results['is_anisotropic'] else 'FAIL'}")
                st.write(f"**✓ Compression Check:** {'Compressed' if results['is_compressed'] else 'Normal'}")
                st.write(f"**✓ Noise Check:** {'High Noise' if results['is_high_freq_noise'] else 'Normal'}")
    
    else:
        st.info("👆 Upload an image to begin deep analysis")

# --- COMPARISON MODE ---
else:
    st.header("⚖️ Comparison Mode: Two Images")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image 1 (Reference/Real)")
        file1 = st.file_uploader("Upload first image", type=["jpg", "png", "jpeg"], key="img1")

    with col2:
        st.subheader("Image 2 (Suspect/AI)")
        file2 = st.file_uploader("Upload second image", type=["jpg", "png", "jpeg"], key="img2")

    if file1 and file2:
        image1 = Image.open(file1)
        image2 = Image.open(file2)

        # Display Images
        c1, c2 = st.columns(2)
        c1.image(image1, use_container_width=True, caption="Image 1")
        c2.image(image2, use_container_width=True, caption="Image 2")

        if st.button("🚀 Run Comparative Forensic Analysis"):
            with st.spinner("Analyzing both images..."):
                # Simple PCA Analysis
                pca1, grads1, gx1, gy1 = process_image_gradients(image1)
                var1 = pca1.explained_variance_ratio_
                
                pca2, grads2, gx2, gy2 = process_image_gradients(image2)
                var2 = pca2.explained_variance_ratio_
                
                # Comprehensive Analysis
                img1_array = np.array(image1)
                img2_array = np.array(image2)

                if len(img1_array.shape) == 3:
                    img1_bgr = cv2.cvtColor(img1_array, cv2.COLOR_RGB2BGR)
                else:
                    img1_bgr = cv2.cvtColor(img1_array, cv2.COLOR_GRAY2BGR)

                if len(img2_array.shape) == 3:
                    img2_bgr = cv2.cvtColor(img2_array, cv2.COLOR_RGB2BGR)
                else:
                    img2_bgr = cv2.cvtColor(img2_array, cv2.COLOR_GRAY2BGR)

                results1 = analyze_image_comprehensive(
                    img1_bgr, beta_min, beta_max, rho_threshold, eta_compressed, eta_noise
                )
                results2 = analyze_image_comprehensive(
                    img2_bgr, beta_min, beta_max, rho_threshold, eta_compressed, eta_noise
                )

            st.divider()
            st.subheader("📊 Comparative Analysis Results")

            # --- Verdicts ---
            v1, v2 = st.columns(2)
            
            verdict_color = {
                "REAL": "green",
                "REAL (Compressed)": "green",
                "REAL (High Texture)": "green",
                "AI-GENERATED": "red",
                "SUSPICIOUS / POSSIBLE AI": "orange"
            }
            
            with v1:
                color1 = verdict_color.get(results1['verdict'], "gray")
                st.markdown(f"""
                <div style="background-color: {color1}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="margin:0; color: white;">{results1['verdict']}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.caption(results1['reason'])
            
            with v2:
                color2 = verdict_color.get(results2['verdict'], "gray")
                st.markdown(f"""
                <div style="background-color: {color2}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="margin:0; color: white;">{results2['verdict']}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.caption(results2['reason'])

            # --- Metrics Comparison ---
            st.markdown("### Metrics Comparison")
            
            m1, m2 = st.columns(2)
            
            with m1:
                st.info("**Image 1 Metrics**")
                st.metric("Beta (Spectral Slope)", f"{results1['beta']:.4f}")
                st.metric("Rho (Anisotropy)", f"{results1['rho']:.4f}")
                st.metric("Eta (HF Ratio)", f"{results1['eta']:.4f}")
                st.metric("PCA Variance Ratio", f"{var1[0]/var1[1]:.2f}")
                
            with m2:
                st.info("**Image 2 Metrics**")
                st.metric("Beta (Spectral Slope)", f"{results2['beta']:.4f}")
                st.metric("Rho (Anisotropy)", f"{results2['rho']:.4f}")
                st.metric("Eta (HF Ratio)", f"{results2['eta']:.4f}")
                st.metric("PCA Variance Ratio", f"{var2[0]/var2[1]:.2f}")

            # --- Gradient Scatter Plots (Original Method) ---
            st.markdown("### Gradient Scatter Plots")
            st.caption("Look for the 'shape' of the cloud. Real images have 'spikier' structured clouds. AI noise creates rounder/fuzzier clouds.")
            
            p1, p2 = st.columns(2)
            with p1:
                fig1 = plot_gradient_distribution(grads1, "Image 1 Gradient Field", "blue")
                st.pyplot(fig1)
                plt.close(fig1)
            
            with p2:
                fig2 = plot_gradient_distribution(grads2, "Image 2 Gradient Field", "red")
                st.pyplot(fig2)
                plt.close(fig2)

            # --- Comprehensive Visualizations ---
            st.markdown("### Comprehensive Visual Analysis")
            
            comp1, comp2 = st.columns(2)
            
            with comp1:
                st.markdown("#### Image 1")
                fig_comp1 = create_comprehensive_visualizations(results1)
                st.pyplot(fig_comp1)
                plt.close(fig_comp1)
            
            with comp2:
                st.markdown("#### Image 2")
                fig_comp2 = create_comprehensive_visualizations(results2)
                st.pyplot(fig_comp2)
                plt.close(fig_comp2)

            # --- Interpretation ---
            st.success("""
            **How to interpret the comparison:**
            1. **Verdict Colors:** Green = Real, Red = AI-Generated, Orange = Suspicious
            2. **Beta (Spectral Slope):** Natural images: 1.6-2.6. Outside this range suggests artificial generation.
            3. **Rho (Anisotropy):** Higher values indicate directional gradients (real photos). Lower values suggest isotropic noise (AI).
            4. **Eta (HF Ratio):** Very low (<0.01) = compression. Very high (>0.25) = AI diffusion noise.
            5. **Gradient Scatter:** Star/cross shape = real. Diffuse cloud = AI.
            6. **Power Spectrum:** Look for grid artifacts and unnatural patterns in AI images.
            """)
            
            # Download Comparison Report Button
            st.markdown("---")
            comparison_buffer = create_downloadable_comparison_report(results1, results2, var1, var2)
            st.download_button(
                label="Download Comparison Report (PNG - 300 DPI)",
                data=comparison_buffer,
                file_name="ai_detection_comparison_report.png",
                mime="image/png",
                use_container_width=True
            )