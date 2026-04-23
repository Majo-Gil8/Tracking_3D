# Tracking_3D

# DHM Particle Tracker — 3D
 
Kalman-filter-based 3D particle tracker for off-axis Digital Holographic Microscopy (DHM). Extends the 2D tracker with a full numerical reconstruction pipeline — spatial filtering, Vortex compensation, Legendre phase correction, and Angular Spectrum propagation — to recover the axial position (Z) of each particle frame by frame.
 
---
 
## Repository structure
 
```
├── tracker_gui.py                # Graphical user interface (2D + 3D)
├── function_tracking_3D.py   # Core 2D tracking library
├── vortexLegendre.py               # Spatial filter, Vortex, Legendre compensation
├── focus_metrics.py                # Focus metrics for Z search
├── Videos                          # Examples videos with the tracking/reconstructions parameters
└── README.md
```
 
---
 
## Requirements
 
| Package | Tested version |
|---------|---------------|
| Python | ≥ 3.9 |
| OpenCV (`opencv-python`) | ≥ 4.7 |
| NumPy | ≥ 1.24 |
| SciPy | ≥ 1.10 |
| Matplotlib | ≥ 3.7 |
| Pillow | ≥ 9.0 |
| scikit-image | ≥ 0.20 |
 
Install all dependencies at once:
 
```bash
pip install opencv-python numpy scipy matplotlib pillow scikit-image
```
 
---
 
## Quick start — GUI
 
```bash
python tracker_gui_7.py
```
 
### 2D tracking only
 
1. Select video type, open video file, and fill in optical system parameters.
2. Adjust detection and Kalman parameters if needed.
3. Click **▶ RUN TRACKING**.
   
### 3D tracking
 
> **3D tracking is only available when the video type is set to `Hologram`.** The reconstruction pipeline requires a raw off-axis hologram as input. The checkbox is automatically disabled for all other modes.
 
1. Select **Hologram** as the video type.
2. Set the **Wavelength λ** and **Vortex filter factor** in the Optical System section.
3. Open your raw hologram video.
4. Scroll to the **3D TRACKING** section and enable it.
5. Choose the **Focus domain** (Amplitude or Phase) and **Focus metric**.
6. Set the **Z search range** (Z min, Z max) and **Z step** in µm.
7. Click **▶ RUN TRACKING**.
The preview window shows the reconstructed field (amplitude or phase) with tracks overlaid in real time. The live info bar displays the current frame number, number of active tracks, processing speed (fps), and current Z estimate.
 
---

## Optical system parameters
 
| Parameter | Description |
|-----------|-------------|
| Camera pixel size (µm) | Physical pixel pitch of the sensor (e.g. `3.75` µm) |
| Magnification (x) | Objective magnification (e.g. `20`, `40`). The effective pixel size is computed automatically as `cam_pixel / magnification` |
| Wavelength λ (µm) | Illumination wavelength (e.g. `0.633` for a He-Ne laser) |
| Vortex filter factor | Controls the radius of the circular spatial filter: `radius = peak_distance / factor`. Larger values give a tighter (smaller) filter. Typical range: `2.0`–`8.0` |
 
---

## 3D reconstruction pipeline
 
Each hologram frame goes through the following steps. 

```
Raw hologram
    │
    ▼
1. Crop to square (M × M)
    │
    ▼
2. Spatial filter  ──►  Isolates the +1 diffraction order
   (FFT → circular mask → iFFT)
    │
    ▼
3. Vortex compensation  ──►  Sub-pixel refinement of the carrier frequency
   (2D Hilbert / Spiral Phase Transform + over-interpolation)
    │
    ▼
4. Reference wave  ──►  obj_complex = ref_wave × holo_filtered
   (carrier tilt removed — no aberration correction yet)
    │
    ▼
5. Angular Spectrum propagation  ──►  Propagate obj_complex to every Z plane
   (no Legendre applied here — field still has low-order aberrations)
    │
    ▼
6. Focus metric evaluation  ──►  Best Z = argmax(metric)
   (crops around detected particles; Legendre not yet applied)
    │
    ▼
7. Legendre phase compensation  ──►  Applied to the field AT the focal plane
   (Legendre polynomial fit, orders 1–9, on the already-propagated field)
    │
    ▼
8. Compensated complex field at best Z  ──►  Used for display and tracking
```

### Vortex compensation (`vortexLegendre.py`)
 
Implements the algorithm of Trujillo et al. to locate the carrier frequency with sub-pixel accuracy using the 2D Hilbert (Spiral Phase) transform and over-interpolation by a factor of 55. This removes residual tilt from the reconstructed phase without requiring an external reference.
 
### Legendre compensation (`vortexLegendre.py`)
 
Fits the unwrapped phase of the propagated field **at the focal plane** with a 2D Legendre polynomial basis (orders 1–9) and subtracts the fitted wavefront. This corrects for residual low-order aberrations (defocus, tilt, astigmatism, coma, and higher-order terms) at the correct propagation distance. Applying compensation at the focal plane rather than before propagation ensures the Z estimate is not biased by Z-dependent aberration terms.
 
- **`RemovePiston=True`** (default) — zeroes the piston coefficient (order 1) before subtracting the wavefront, which removes the global DC offset.
- **`RemovePiston=False`** — searches for the optimal piston value that minimises the phase variance of the compensated field.
- **`UsePCA=True`** — extracts the dominant wavefront component via SVD before fitting, which improves robustness when the field contains multiple superimposed contributions.
### Angular Spectrum propagation (`vortexLegendre.py` / `tracker_gui.py`)
 
Propagates the compensated complex field to each plane in the Z search range using the exact transfer function:
 
```
H(fx, fy; z) = exp( i · k · z · √(1 − (λfx)² − (λfy)²) )
```
 
Evanescent components (`(λfx)² + (λfy)² > 1`) are suppressed by clipping the argument to zero. This is an exact scalar diffraction result with no paraxial approximation.
 
---
 
## Focus metrics (`focus_metrics.py`)
 
The Z position is found by maximising a focus metric over the propagated stack. The metric is evaluated only on small crops (48 px radius) centred on each detected particle, which significantly reduces computation time.
 
### Amplitude domain
 
| Metric | Description | Recommended for |
|--------|-------------|-----------------|
| **Tenengrad** | Mean squared Sobel gradient energy. Best noise robustness among spatial operators (Pertuz et al. 2013). | General amplitude DHM. Default choice. |
| **Variance** | Image variance. Fast, works well for high-contrast particles. | Dense particle fields. |
| **Laplacian** | Variance of the Laplacian. High sensitivity but more noise-sensitive than Tenengrad. | High-SNR conditions. |
 
### Phase domain
 
| Metric | Description | Recommended for |
|--------|-------------|-----------------|
| **Phase Gradient** | Mean gradient magnitude of the wrapped phase. Best axial localisation for off-axis DHM (Dubois et al. 2006; Langehanenberg et al. 2008). | Default for phase-mode tracking. |
| **Phase Variance** | Variance of the unwrapped phase. Effective for large or phase-rich objects (Ma et al. 2004). | Large cells, dense phase objects. |
| **Spectral Energy** | High-frequency energy of the phase spectrum. Focuses energy in high spatial frequencies at the focal plane (Ferraro et al. 2003). | High-resolution setups. |
 
---
 
## Z search parameters
 
| Parameter | Description |
|-----------|-------------|
| `Z min (µm)` | Lower bound of the axial search range |
| `Z max (µm)` | Upper bound of the axial search range |
| `Z step (µm)` | Axial step between propagation planes. Smaller steps give better Z resolution but increase computation time |
 
The total number of planes evaluated per frame is `(Z_max − Z_min) / Z_step`. A practical starting point is a ±50 µm range with a 2 µm step (50 planes).
 
---

 ## Output
 
### Plots (shown after tracking)
 
1. **XY Trajectories** — all tracks in the focal plane (µm).
2. **Z vs. frame** — axial position over time for each track (3D mode only).
3. **3D Trajectories** — full XYZ trajectories (3D mode only).
4. **Speed Profiles** — instantaneous XY speed (µm/frame).
### CSV export
 
**2D CSV** — same as the 2D tracker:
 
| Column | Description |
|--------|-------------|
| `track_id` | Track index |
| `frame` | Frame number |
| `x_px`, `y_px` | XY position in pixels |
| `x_um`, `y_um` | XY position in µm |
 
**3D CSV** (saved as `<name>_3D.csv`):
 
| Column | Description |
|--------|-------------|
| `track` | Track index |
| `frame` | Frame number |
| `x_um`, `y_um` | XY position in µm |
| `z_um` | Axial position in µm |
 
---
 
## Tips
 
- **Vortex filter factor** — if the spatial filter is too large and picks up noise from neighbouring orders, increase the factor (e.g. from `2.0` to `5.0`). If the filter clips the signal, decrease it.
- **Z step** — use a coarse step (5–10 µm) for a first pass to find the approximate Z range, then refine with a finer step (1–2 µm).
- **Focus domain** — use *Amplitude* for bright, high-contrast particles. Use *Phase* when particles are weakly scattering or nearly transparent.
- **Tenengrad vs. Phase Gradient** — Tenengrad is the most reliable amplitude metric. Phase Gradient gives the sharpest axial localisation for off-axis DHM phase reconstructions.
- **`UsePCA=True`** in Legendre compensation helps when the hologram contains a strong background or multiple overlapping wavefronts.
- Processing speed depends directly on the number of Z planes. Reducing the Z range or increasing the step significantly speeds up tracking for long videos.
---

## References
 
- Pertuz S. et al., *Pattern Recognition* **46** (2013) 1415–1432. Focus measure operators.
- Dubois F. et al., *Appl. Opt.* **45** (2006) 7127. Phase gradient autofocus.
- Langehanenberg P. et al., *Appl. Opt.* **47** (2008) D176. Autofocusing in DHM.
- Ma L. et al., *Opt. Lett.* **29** (2004) 1671. Phase variance focus metric.
- Ferraro P. et al., *Opt. Lett.* **28** (2003) 1257. Spectral energy focus metric.- Ortega K., Restrepo R., Padilla-Vivanco A., Castaneda R., Doblas A., Trujillo C. *Intricate Quantitative Phase Imaging via Vortex-Legendre High-Order Phase Compensation.* Opt. Lasers Eng. **195** (2025) 109318. https://doi.org/10.2139/ssrn.5282404
- Goodman J.W., *Introduction to Fourier Optics*, 4th ed. Angular Spectrum method.
