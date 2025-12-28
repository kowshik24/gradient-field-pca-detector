# SynthDetect Ultra: AI Image Forensics

SynthDetect Ultra is an advanced forensic tool for detecting AI-generated (synthetic) images using a combination of physics-inspired, statistical, and deep forensic features. It is built with Streamlit for an interactive web-based analysis experience.

## Features

- **Gradient Field Analysis**: Measures anisotropy and coherence in image gradients to detect unnatural patterns.
- **Frequency Domain Analysis**: Analyzes spectral slope and high-frequency ratios to spot GAN/Diffusion artifacts.
- **Benford's Law (DCT)**: Checks for naturalness in DCT coefficient distributions.
- **CFA Artifacts**: Examines color filter array correlations typical of real camera sensors.
- **Texture Analysis (GLCM)**: Extracts homogeneity and contrast from gray-level co-occurrence matrices.
- **Chromatic Aberration**: Detects lens-induced color fringing absent in AI images.
- **Noise Residual Kurtosis**: Measures the "tailedness" of noise, distinguishing real sensor noise from AI noise.
- **ELA (Error Level Analysis)**: Assesses JPEG compression error patterns for inconsistencies.
- **Grid Artifact Detection**: Finds periodic patterns in the frequency domain typical of upsampling artifacts.
- **Noise Consistency**: Compares noise correlation across RGB channels.
- **Color Space Anomalies**: Analyzes LAB and HSV distributions for unnatural color statistics.
- **Micro-Texture Analysis**: Uses wavelet transforms to examine fine texture kurtosis and symmetry.
- **Edge Profile Analysis**: Evaluates edge sharpness, consistency, and uniformity.
- **EXIF Metadata Forensics**: Flags suspicious or missing metadata, including AI software tags.
- **Ensemble Decision Engine**: Weighted scoring system for robust, explainable verdicts.
- **Detection History**: Stores and summarizes past analyses in a local SQLite database.

## Usage

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have the following key packages: `streamlit`, `opencv-python`, `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-image`, `pywt`, `Pillow`.

2. **Run the App**
   ```bash
   streamlit run app_1.py
   ```

3. **Upload an Image**
   - Supported formats: JPG, PNG, JPEG, WEBP
   - The app will process the image and display a detailed forensic report.

4. **Interpret Results**
   - The dashboard provides a confidence score, core and advanced metrics, and visualizations.
   - Flags highlight suspicious features (e.g., lack of lens aberration, artificial noise, grid artifacts).
   - A radar chart and summary statistics offer a multi-dimensional authenticity profile.


## Forensic Metrics and Equations

### 1. Gradient Field Analysis
- **Anisotropy (ρ):**
   $$
   \rho = \frac{\lambda_1}{\lambda_2}
   $$
   where $\lambda_1, \lambda_2$ are eigenvalues of the gradient covariance matrix.
- **Coherence (κ):**
   $$
   \kappa = \left(\frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2}\right)^2
   $$

### 2. Frequency Domain Analysis
- **Spectral Slope (β):**
   $$
   \beta = -\text{slope of linear fit to } \log_{10}(\text{radial frequency}) \text{ vs. } \log_{10}(\text{power})
   $$
- **High-Frequency Ratio (η):**
   $$
   \eta = \frac{\sum_{f>f_c} P(f)}{\sum_{f>0} P(f)}
   $$
   where $P(f)$ is the radial power spectrum, $f_c$ is a cutoff frequency.

### 3. Benford's Law (DCT)
- **Benford Divergence:**
   $$
   D = \sum_{d=1}^9 \frac{(p_d - b_d)^2}{b_d}
   $$
   where $p_d$ is the observed proportion of first digits, $b_d = \log_{10}(1 + 1/d)$ is the Benford distribution.

### 4. CFA Artifacts
- **CFA Score:**
   $$
   	ext{CFA} = \text{corr}(G - R, G - B)
   $$

### 5. Texture Analysis (GLCM)
- **Homogeneity:**
   $$
   \sum_{i,j} \frac{P(i,j)}{1 + |i-j|}
   $$
- **Contrast:**
   $$
   \sum_{i,j} (i-j)^2 P(i,j)
   $$
   where $P(i,j)$ is the normalized GLCM.

### 6. Chromatic Aberration
- **Radial Shift Ratio:**
   $$
   	ext{CA} = \frac{\text{mean}|G-R|_{\text{edge}}}{\text{mean}|G-R|_{\text{center}}}
   $$

### 7. Noise Residual Kurtosis
- **Kurtosis:**
   $$
   	ext{Kurtosis} = \frac{E[(X - \mu)^4]}{\sigma^4}
   $$
   where $X$ is the noise residual, $\mu$ is the mean, $\sigma$ is the standard deviation.

### 8. ELA (Error Level Analysis)
- **ELA Mean:**
   $$
   	ext{ELA Mean} = \text{mean}(|I_{\text{orig}} - I_{\text{jpeg}}|)
   $$

### 9. Grid Artifact Score
- **Grid Score:**
   $$
   	ext{Grid Score} = \frac{\text{# horizontal peaks} + \text{# vertical peaks}}{2}
   $$

### 10. Noise Consistency
- **Noise Correlation:**
   $$
   	ext{Noise Corr} = \text{mean}([\text{corr}(R,G), \text{corr}(R,B), \text{corr}(G,B)])
   $$

### 11. Color Space Anomalies
- **Color Anomaly Score:**
   $$
   	ext{Color Anomaly} = \frac{|\text{skew}(a^*)| + |\text{skew}(b^*)|}{2}
   $$

### 12. Micro-Texture (Wavelet Kurtosis)
- **Wavelet Kurtosis:**
   $$
   	ext{mean}(|\text{kurtosis of detail coefficients}|)
   $$

### 13. Edge Profile Analysis
- **Edge Consistency:**
   $$
   	ext{Edge Consistency} = \frac{\text{std}(\text{edge gradients})}{\text{mean}(\text{edge gradients})}
   $$

---

## Example Testing Results

![Testing Results](test_results/example_results.png)

*Replace the above image with your actual results screenshot or visualization.*


## Customization

- Adjust detection thresholds in the sidebar for sensitivity tuning.
- View detection history and statistics in the sidebar.

## Limitations

- Some heavily post-processed or compressed real images may trigger false positives.
- The tool is designed for single-image analysis; batch or video support is not included.


## Citation
If you use SynthDetect Ultra in your research or project, please cite appropriately.

---

**Author:** Koshik Debanath

**License:** MIT
