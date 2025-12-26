import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st
from io import BytesIO
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="AI Image Detection",
    page_icon="🔍",
    layout="wide"
)

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

def compute_gradient_features(L):
    """
    Calculates features based on Gradient PCA
    Returns: Anisotropy (rho), Coherence (kappa)
    """
    # Gradientes Sobel
    Gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    
    # Aplanar
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    M = np.stack((Gx_flat, Gy_flat), axis=1)
    
    # Matriz de Covarianza
    # C = (1/N) * M.T * M
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
    
    return rho, kappa, C, Gx, Gy

def compute_frequency_features(L):
    """
    Calculates features in the Frequency Domain
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
    # log(S(r)) = -beta * log(r) + c
    # Ignorar componente DC (r=0) y ruido de muy alta frecuencia
    r_axis = np.arange(len(radial_profile))
    mask = (r_axis > 0) & (r_axis < min(rows, cols)//2)
    
    log_r = np.log10(r_axis[mask])
    log_S = np.log10(radial_profile[mask])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_S)
    beta = -slope
    
    # 2. Ratio de Alta Frecuencia (eta)
    # Ratio de energía sobre frecuencia de corte vs energía total
    cutoff = min(rows, cols) / 4
    
    # CORRECCIÓN: Excluir componente DC (r=0) de la energía total, de lo contrario domina
    # y eta se vuelve ~0.0
    total_energy = np.sum(radial_profile[1:])
    high_freq_energy = np.sum(radial_profile[r_axis > cutoff])
    
    if total_energy > 0:
        eta = high_freq_energy / total_energy
    else:
        eta = 0.0
    
    return beta, eta, radial_profile, log_r, log_S, slope, intercept, power_spectrum

def analyze_image(img):
    """Analyze image and return results"""
    
    # 0. Preprocesamiento: Redimensionar a tamaño estándar (máx 1024px)
    height, width = img.shape[:2]
    max_dim = 1024
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    L, img_rgb = get_luminance(img)
    
    # --- Extracción de Características ---
    rho, kappa, C, Gx, Gy = compute_gradient_features(L)
    beta, eta, radial_prof, log_r, log_S, slope, intercept, power_spectrum = compute_frequency_features(L)
    
    # --- Umbrales y Refinamiento Lógico (v2.0) ---
    is_natural_physics = 1.6 <= beta <= 2.6
    T_rho = 1.15
    is_anisotropic = rho > T_rho
    is_compressed = eta < 0.01
    is_high_freq_noise = eta > 0.25
    
    # Lógica de Decisión
    verdict_text = "DESCONOCIDO"
    verdict_reason = ""
    
    if is_natural_physics:
        if is_compressed:
            verdict_text = "REAL (Compressed)"
            verdict_reason = "Light physics (Beta) is correct. The lack of fine details is likely due to compression, not AI generation."
        elif is_anisotropic:
            verdict_text = "REAL"
            verdict_reason = "Passes all physical and structural tests."
        else:
            verdict_text = "REAL (High Texture)"
            verdict_reason = "Light physics is correct. Low anisotropy is normal in nature photos (grass, trees) or complex textures."
    else:
        if is_high_freq_noise:
            verdict_text = "AI-GENERATED"
            verdict_reason = "Excessive high-frequency noise and incorrect light physics."
        else:
            verdict_text = "SUSPICIOUS / POSSIBLE AI"
            verdict_reason = "Does not follow natural light statistics (Beta out of range)."
    
    return {
        'verdict': verdict_text,
        'reason': verdict_reason,
        'beta': beta,
        'rho': rho,
        'eta': eta,
        'kappa': kappa,
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
        'power_spectrum': power_spectrum
    }

def create_visualizations(results):
    """Create analysis visualizations"""
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Original
    axes[0, 0].imshow(results['img_rgb'])
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # 2. Magnitud del Gradiente
    G_mag = np.sqrt(results['Gx']**2 + results['Gy']**2)
    im2 = axes[0, 1].imshow(G_mag, cmap='inferno')
    axes[0, 1].set_title(f"Gradients (Anisotropy: {results['rho']:.2f})")
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Espectro de Potencia (Log)
    power_log = np.log10(np.fft.fftshift(results['power_spectrum']) + 1)
    im3 = axes[1, 0].imshow(power_log, cmap='jet')
    axes[1, 0].set_title("Power Spectrum (Log)")
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 4. Perfil Radial (Log-Log)
    axes[1, 1].plot(results['log_r'], results['log_S'], label='Data', linewidth=2)
    axes[1, 1].plot(results['log_r'], results['intercept'] - results['beta']*results['log_r'], 
                    'r--', label=f'Fit (beta={results["beta"]:.2f})', linewidth=2)
    axes[1, 1].set_xlabel("Log Frequency")
    axes[1, 1].set_ylabel("Log Power")
    axes[1, 1].set_title("Radial Profile (Spectral Slope)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    return fig

# Streamlit UI
st.title("🔍 AI-Generated Image Detector")
st.markdown("""
This application analyzes images using gradient and frequency analysis techniques 
to determine if an image is real or AI-generated.

### Features analyzed:
- **Light Physics (Beta)**: Spectral slope - Measures if the image follows natural light statistics
- **Gradient Structure (Rho)**: Anisotropy - Analyzes gradient coherence
- **Fine Details (Eta)**: High frequency ratio - Detects noise or compression
""")

st.sidebar.header("📊 Information")
st.sidebar.markdown("""
**Framework v2.0**  
Nature/Compression Aware

**Reference Ranges:**
- Natural Beta: 1.6 - 2.6
- Rho threshold: > 1.15
- Compressed Eta: < 0.01
- High noise Eta: > 0.25
""")

# File uploader
uploaded_files = st.file_uploader(
    "Upload one or more images to analyze",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)

if uploaded_files:
    st.header("📋 Analysis Results")
    
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")
        st.subheader(f"Analysis {idx+1}: {uploaded_file.name}")
        
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error(f"Error: Could not load image {uploaded_file.name}")
            continue
        
        # Analyze image
        with st.spinner("Analyzing image..."):
            results = analyze_image(img)
        
        # Display results in columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display verdict
            verdict_color = {
                "REAL": "green",
                "REAL (Compressed)": "green",
                "REAL (High Texture)": "green",
                "AI-GENERATED": "red",
                "SUSPICIOUS / POSSIBLE AI": "orange"
            }
            
            color = verdict_color.get(results['verdict'], "gray")
            
            st.markdown(f"### Verdict")
            st.markdown(f"<h2 style='color: {color};'>{results['verdict']}</h2>", unsafe_allow_html=True)
            st.info(results['reason'])
            
            st.markdown("### Metrics")
            
            # Beta (Física de la Luz)
            st.metric(
                label="🌟 Beta (Pendiente Espectral)",
                value=f"{results['beta']:.4f}",
                delta="Natural" if results['is_natural_physics'] else "Anormal",
                delta_color="normal" if results['is_natural_physics'] else "inverse"
            )
            
            # Rho (Anisotropía)
            st.metric(
                label="📐 Rho (Anisotropía)",
                value=f"{results['rho']:.4f}",
                delta="Coherente" if results['is_anisotropic'] else "Bajo",
                delta_color="normal" if results['is_anisotropic'] else "inverse"
            )
            
            # Eta (Alta Frecuencia)
            eta_status = "Compressed" if results['is_compressed'] else ("High Noise" if results['is_high_freq_noise'] else "Normal")
            st.metric(
                label="🔊 Eta (High Frequency Ratio)",
                value=f"{results['eta']:.4f}",
                delta=eta_status,
                delta_color="normal" if eta_status == "Normal" else "inverse"
            )
        
        with col2:
            st.markdown("### Detailed Analysis")
            
            # Create and display visualizations
            fig = create_visualizations(results)
            st.pyplot(fig)
            plt.close(fig)
        
        # Detailed analysis expander
        with st.expander("📊 View Detailed Analysis"):
            st.markdown("#### [1] LIGHT PHYSICS (Spectral Slope - Beta)")
            st.write(f"- **Value:** {results['beta']:.4f}")
            st.write(f"- **Natural Range:** 1.6 - 2.6")
            if results['is_natural_physics']:
                st.success("✅ The image follows natural light statistics (1/f Law).")
            else:
                st.warning("⚠️ The frequency distribution does not appear natural.")
            
            st.markdown("#### [2] GRADIENT STRUCTURE (Anisotropy - Rho)")
            st.write(f"- **Value:** {results['rho']:.4f}")
            st.write(f"- **Adjusted Threshold:** > 1.15")
            if results['is_anisotropic']:
                st.success("✅ Coherent gradient structure.")
            else:
                st.warning("⚠️ Chaotic gradients (common in AI or high vegetation/texture).")
            
            st.markdown("#### [3] FINE DETAILS (High Frequency Ratio - Eta)")
            st.write(f"- **Value:** {results['eta']:.4f}")
            if results['is_compressed']:
                st.info("ℹ️ Very low value. Indicates STRONG COMPRESSION (e.g. WhatsApp/Web).")
            elif results['is_high_freq_noise']:
                st.warning("⚠️ Excess high frequency energy (possible diffusion noise).")
            else:
                st.success("✅ Normal detail level.")

else:
    st.info("👆 Upload one or more images to begin analysis")
    
    # Example section
    st.markdown("---")
    st.markdown("### 💡 How does it work?")
    st.markdown("""
    This detector uses advanced mathematical analysis to examine:
    
    1. **Gradient Domain**: Analyzes how pixels change in the image
    2. **Frequency Domain**: Examines spatial frequencies using FFT
    3. **Light Physics**: Verifies if the image follows natural light distribution laws
    
    AI-generated images (especially diffusion models) typically have:
    - Excessive high-frequency noise
    - Chaotic or overly smooth gradients
    - Spectral distributions that don't follow natural 1/f statistics
    """)
