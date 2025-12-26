import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SynthDetect | Gradient Field Analysis",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
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
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# --- MATH & PROCESSING FUNCTIONS ---

def get_luminance(img_bgr):
    """
    Converts BGR image to Luminance using BT.709 standard.
    Equation (1) in Technical Documentation.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]
    # L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return L, img_rgb

def compute_gradient_pca(L):
    """
    Computes Gradient PCA features.
    See Section 2 of Technical Documentation.
    Returns: Anisotropy (rho), Coherence (kappa), Covariance Matrix, Eigenvalues
    """
    # 1. Compute Gradients (Sobel) - Section 2.3.1
    Gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient Magnitude for visualization
    G_mag = np.sqrt(Gx**2 + Gy**2)
    
    # 2. Flatten to Matrix M - Section 2.4
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    M = np.stack((Gx_flat, Gy_flat), axis=1)
    
    # 3. Covariance Matrix C - Section 2.5
    N = M.shape[0]
    # Center the data? Document implies E[Gx]~0, but let's be safe
    # M_centered = M - np.mean(M, axis=0) 
    # Using raw second moments per Equation (10)
    C = (1/N) * np.dot(M.T, M)
    
    # 4. Eigendecomposition - Section 2.6
    eigenvalues, _ = np.linalg.eigh(C)
    # Sort descending (lambda1 >= lambda2)
    eigenvalues = eigenvalues[::-1]
    l1, l2 = eigenvalues[0], eigenvalues[1]
    
    # Avoid division by zero
    l2 = max(l2, 1e-10)
    l1 = max(l1, 1e-10)
    
    # 5. Features - Section 2.7
    rho = l1 / l2  # Anisotropy Ratio
    kappa = ((l1 - l2) / (l1 + l2))**2  # Coherence
    energy = l1 + l2 # Gradient Energy
    
    return {
        "rho": rho,
        "kappa": kappa,
        "energy": energy,
        "l1": l1,
        "l2": l2,
        "G_mag": G_mag
    }

def compute_frequency_analysis(L):
    """
    Computes Frequency Domain features (DFT).
    See Section 3.1 of Technical Documentation.
    Returns: Spectral Slope (beta), High-Freq Ratio (eta)
    """
    rows, cols = L.shape
    
    # 1. FFT and Power Spectrum - Section 3.1.1 - 3.1.3
    F = np.fft.fft2(L)
    Fshift = np.fft.fftshift(F)
    magnitude_spectrum = np.abs(Fshift)
    power_spectrum = magnitude_spectrum**2
    
    # 2. Radial Power Spectrum - Section 3.1.4
    center_x, center_y = cols // 2, rows // 2
    y, x = np.ogrid[-center_y:rows-center_y, -center_x:cols-center_x]
    r = np.sqrt(x*x + y*y).astype(int)
    
    # Sum power per radius
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    # Avoid division by zero
    radial_profile = tbin / np.maximum(nr, 1)
    
    # 3. Spectral Slope (Beta) - Section 3.1.5
    # Natural images follow 1/f^beta, so log(S) ~ -beta * log(f)
    r_axis = np.arange(len(radial_profile))
    
    # Mask DC component (0) and very high frequencies (corner artifacts)
    # Typically fit between r=1 and Nyquist/2 or min dimension
    mask = (r_axis > 2) & (r_axis < min(rows, cols)//2)
    
    if np.sum(mask) > 10:
        log_r = np.log10(r_axis[mask])
        log_S = np.log10(radial_profile[mask] + 1e-10) # Avoid log(0)
        
        slope, intercept, r_value, _, _ = stats.linregress(log_r, log_S)
        beta = -slope
    else:
        beta = 0.0
        slope, intercept = 0, 0
        log_r, log_S = np.array([]), np.array([])
        
    # 4. High-Frequency Ratio (Eta) - Section 3.1.5
    # High frequency typically > Nyquist/4
    cutoff = min(rows, cols) / 4
    
    # Calculate energy excluding DC component
    total_energy = np.sum(radial_profile[1:])
    high_freq_energy = np.sum(radial_profile[r_axis > cutoff])
    
    eta = (high_freq_energy / total_energy) if total_energy > 0 else 0.0
    
    return {
        "beta": beta,
        "eta": eta,
        "radial_profile": radial_profile,
        "log_r": log_r,
        "log_S": log_S,
        "slope": slope,
        "intercept": intercept,
        "log_spectrum": np.log10(power_spectrum + 1)
    }

# --- UI LAYOUT ---

st.sidebar.title("🛠️ Configuration")
st.sidebar.info("This tool implements the 'Synthetic Image Detection using Gradient Fields' framework (Nov 2025).")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

resize_opt = st.sidebar.checkbox("Resize large images", value=True, help="Resizes to 1024px max dimension for consistent gradient statistics.")

st.title("Synthetic Image Detection")
st.subheader("Gradient Field & Frequency Analysis")

if uploaded_file is not None:
    # Load Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    if img_bgr is None:
        st.error("Error loading image.")
    else:
        # Preprocessing
        height, width = img_bgr.shape[:2]
        
        if resize_opt and max(height, width) > 1024:
            scale = 1024 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_proc = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            st.sidebar.warning(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        else:
            img_proc = img_bgr

        # Computation
        with st.spinner("Analyzing Gradient Fields and Frequency Spectrum..."):
            L, img_rgb = get_luminance(img_proc)
            grad_stats = compute_gradient_pca(L)
            freq_stats = compute_frequency_analysis(L)

        # --- DASHBOARD ---
        
        col_img, col_verdict = st.columns([1, 1])
        
        with col_img:
            st.image(img_rgb, caption="Input Image", use_container_width=True)

        with col_verdict:
            st.markdown("### Analysis Verdict")
            
            # --- DECISION LOGIC (Based on Document & Code) ---
            beta = freq_stats['beta']
            rho = grad_stats['rho']
            eta = freq_stats['eta']
            
            # Thresholds
            # Beta: Real images ~ 2.0 (1/f^2 law). AI often < 1.6 (noisy) or > 2.6 (waxy)
            is_natural_physics = 1.6 <= beta <= 2.6
            
            # Rho: Real images are anisotropic (directional light). AI is isotropic (noise).
            # Lowered slightly to 1.15 to account for highly textured natural scenes.
            is_anisotropic = rho > 1.15
            
            # Eta: High freq ratio. Very low (<0.01) = JPEG. Very high (>0.25) = Diffusion Noise.
            is_compressed = eta < 0.01
            is_diffusion_noise = eta > 0.25
            
            score = 0
            reasons = []
            
            if is_natural_physics: score += 1
            else: reasons.append("Spectral Slope (Beta) deviates from natural light physics.")
            
            if is_anisotropic: score += 1
            else: reasons.append("Gradient field is Isotropic (lacks directional coherence).")
            
            if is_diffusion_noise: 
                score -= 10 # Strong indicator
                reasons.append("High-frequency energy is excessive (Diffusion artifacts).")
            
            # Final Label
            if is_compressed and is_natural_physics:
                label = "REAL (Compressed)"
                color = "green"
                confidence = "High"
            elif score == 2:
                label = "LIKELY REAL"
                color = "green"
                confidence = "High"
            elif score == 1:
                label = "UNCERTAIN / HYBRID"
                color = "orange"
                confidence = "Low"
            else:
                label = "LIKELY SYNTHETIC (AI)"
                color = "red"
                confidence = "High"

            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="margin:0; color: white;">{label}</h2>
                <p style="margin:0;">Confidence: {confidence}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if reasons:
                st.warning("**Anomalies Detected:**\n" + "\n".join([f"- {r}" for r in reasons]))
            else:
                st.success("**Physics Consistency:** Image adheres to natural image statistics.")

        # --- METRICS ROW ---
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric(
                label="Anisotropy (ρ)",
                value=f"{rho:.3f}",
                delta="Real > 1.15" if rho > 1.15 else "Low Directionality",
                help="Ratio of eigenvalues (λ1/λ2). Real images have directional gradients due to physical lighting. AI images tend to be isotropic."
            )
        
        with m2:
            st.metric(
                label="Spectral Slope (β)",
                value=f"{beta:.3f}",
                delta="Normal" if 1.6 <= beta <= 2.6 else "Abnormal",
                delta_color="normal" if 1.6 <= beta <= 2.6 else "inverse",
                help="Slope of the log-log power spectrum. Natural images follow ~1/f^2 (Beta=2). Deviation implies artificial generation."
            )
            
        with m3:
            st.metric(
                label="HF Energy Ratio (η)",
                value=f"{eta:.3f}",
                delta="Compressed" if eta < 0.01 else ("Noisy" if eta > 0.25 else "Normal"),
                delta_color="inverse" if eta > 0.25 else "off",
                help="Ratio of high-frequency energy. Very low indicates JPEG compression. Very high indicates diffusion denoising artifacts."
            )

        # --- VISUALIZATIONS ---
        st.markdown("### Detailed Visualization")
        
        tab1, tab2, tab3 = st.tabs(["Gradient Analysis", "Frequency Analysis", "Raw Data"])
        
        with tab1:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_g, ax_g = plt.subplots()
                im = ax_g.imshow(grad_stats['G_mag'], cmap='inferno')
                ax_g.set_title(f"Gradient Magnitude (Energy={grad_stats['energy']:.1f})")
                ax_g.axis('off')
                plt.colorbar(im, ax=ax_g, fraction=0.046, pad=0.04)
                st.pyplot(fig_g)
                st.caption("Visualizes edge strength. Real images usually have sparse, connected edges.")
                
            with col_g2:
                # Covariance ellipse visualization concept
                fig_e, ax_e = plt.subplots()
                ax_e.set_xlim(-1, 1)
                ax_e.set_ylim(-1, 1)
                ax_e.set_aspect('equal')
                
                # Draw eigenvectors
                # This is a simplified abstract representation
                origin = [0, 0]
                # Scale for visualization
                scale = 0.8
                
                # Eigenvector 1 (Dominant) - Fixed to x-axis for relative viz, or calculated angle
                angle = 0 # In PCA space, we just care about the ratio
                
                ellipse = plt.matplotlib.patches.Ellipse(
                    (0, 0), 
                    width=scale * 1.0, 
                    height=scale * (1/rho), 
                    color='blue', 
                    alpha=0.3,
                    label='Gradient Distribution'
                )
                ax_e.add_patch(ellipse)
                ax_e.legend()
                ax_e.set_title(f"PCA Ellipse (ρ = {rho:.2f})")
                ax_e.grid(True, linestyle='--')
                st.pyplot(fig_e)
                st.caption(f"Geometric representation of covariance. A flatter ellipse (higher ρ) indicates real physical lighting.")

        with tab2:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                fig_s, ax_s = plt.subplots()
                im_s = ax_s.imshow(freq_stats['log_spectrum'], cmap='jet')
                ax_s.set_title("Log Power Spectrum")
                ax_s.axis('off')
                st.pyplot(fig_s)
                st.caption("2D Frequency content. Look for grid artifacts (checkerboard patterns) often found in GANs/Diffusion upsampling.")
                
            with col_f2:
                fig_p, ax_p = plt.subplots()
                ax_p.plot(freq_stats['log_r'], freq_stats['log_S'], label='Radial Data', alpha=0.6)
                
                # Regression line
                if len(freq_stats['log_r']) > 0:
                    fit_line = freq_stats['intercept'] - beta * freq_stats['log_r']
                    ax_p.plot(freq_stats['log_r'], fit_line, 'r--', label=f'Fit (β={beta:.2f})')
                
                ax_p.set_xlabel("Log Frequency")
                ax_p.set_ylabel("Log Power")
                ax_p.set_title("Radial Profile & Spectral Slope")
                ax_p.legend()
                ax_p.grid(True, which="both", linestyle='--', alpha=0.5)
                st.pyplot(fig_p)
                st.caption("Linear fit indicates 1/f scaling. Deviations suggest artificial generation.")

        with tab3:
            st.write("#### Mathematical Internals")
            st.json({
                "Eigenvalue 1 (λ1)": f"{grad_stats['l1']:.6f}",
                "Eigenvalue 2 (λ2)": f"{grad_stats['l2']:.6f}",
                "Gradient Coherence (κ)": f"{grad_stats['kappa']:.6f}",
                "Regression Intercept": f"{freq_stats['intercept']:.4f}",
                "Image Dimensions": f"{img_proc.shape}"
            })
            st.markdown("""
            **Reference:** 
            *Abeywardhana, K. (2025). Synthetic Image Detection using Gradient Fields.*
            """)

else:
    st.markdown("""
    ### 👋 Welcome
    
    Upload an image in the sidebar to begin analysis.
    
    **How it works:**
    1. **Luminance Conversion:** Converts image to grayscale using BT.709.
    2. **Gradient PCA:** Calculates horizontal/vertical gradients and finds the principal components of their covariance.
       - *Real photos* typically have **High Anisotropy** (directional lighting).
       - *AI images* tend to be more **Isotropic** (uniform noise distribution).
    3. **Frequency Analysis:** Performs FFT to check the "1/f" natural image statistic law.
    
    **Privacy Note:** Images are processed in RAM and not saved to disk.
    """)