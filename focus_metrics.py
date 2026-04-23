"""
focus_metrics.py
================
Focus metrics for depth (Z-axis) search in DHM reconstruction.

AMPLITUDE metrics  :
  - Variance      : Image variance. Classic, fast, works well for bright particles.
  - Tenengrad     : Sum of squared Sobel gradients. Best noise robustness among
                    spatial-domain operators (Pertuz et al. 2013; Sun et al. 2021).
  - Laplacian     : Variance of the Laplacian (EOL). High sensitivity but less
                    noise-robust than Tenengrad (Nayar & Nakagawa 1994).

PHASE metrics  :
  - Phase Gradient  : Mean gradient magnitude of the wrapped phase. Best axial
                      localization for off-axis DHM (Dubois et al. 2006;
                      Langehanenberg et al. 2008).
  - Phase Variance  : Variance of the unwrapped phase. Increases near focus as
                      the structured phase becomes visible (Ma et al. 2004).
  - Spectral Energy : High-frequency energy of the phase spectrum. Focuses energy
                      in high spatial frequencies at the focal plane
                      (Ferraro et al. 2003).

References
----------
- Pertuz S. et al., Pattern Recognition 46 (2013) 1415–1432.
  "Analysis of focus measure operators for shape-from-focus"
- Nayar S.K. & Nakagawa Y., IEEE TPAMI 16(8) (1994) 824–831.
- Dubois F. et al., Appl. Opt. 45 (2006) 7127.
- Langehanenberg P. et al., Appl. Opt. 47 (2008) D176.
- Ma L. et al., Opt. Lett. 29 (2004) 1671.
- Ferraro P. et al., Opt. Lett. 28 (2003) 1257.
"""

import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from skimage.restoration import unwrap_phase


# ── AMPLITUDE METRICS ─────────────────────────────────────────────────────────

def metric_variance(image: np.ndarray) -> float:
    """
    Variance of the amplitude image.
    Classic focus metric: peaks sharply at the focal plane because a focused
    particle produces high-contrast intensity variations (Pertuz et al. 2013).
    Fast and reliable for bright, high-contrast particles.
    """
    return float(np.var(image.astype(np.float64)))


def metric_tenengrad(image: np.ndarray) -> float:
    """
    Sum of squared Sobel gradients (Tenengrad).
    Computes the energy of the intensity gradient using 3×3 Sobel operators.
    Among spatial-domain focus operators, Tenengrad shows the best noise
    robustness while maintaining good sensitivity (Pertuz et al. 2013;
    Sun et al. 2021). Recommended default for amplitude DHM.

    Formula: F = Σ (Gx² + Gy²)  where Gx, Gy are Sobel responses.
    """
    img8 = cv2.normalize(np.abs(image), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gx = cv2.Sobel(img8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def metric_laplacian(image: np.ndarray) -> float:
    """
    Variance of the Laplacian (Energy of Laplace, EOL).
    Applies the Laplacian operator (second-order derivative) and returns its
    variance. Detects sharp transitions but is more sensitive to noise than
    Tenengrad (Nayar & Nakagawa 1994; Pertuz et al. 2013).

    Formula: F = Var(∇²I)
    """
    img8 = cv2.normalize(np.abs(image), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lap = cv2.Laplacian(img8, cv2.CV_64F)
    return float(np.var(lap))


# ── PHASE METRICS ─────────────────────────────────────────────────────────────

def metric_phase_gradient(complex_field: np.ndarray) -> float:
    """
    Mean gradient magnitude of the wrapped phase.
    At focus, a particle produces strong, spatially localized phase gradients.
    This metric provides excellent axial localization in off-axis DHM and is
    less sensitive to phase unwrapping errors than Phase Variance
    (Dubois et al. 2006; Langehanenberg et al. 2008).

    Formula: F = mean(|∇φ|)  where φ = angle(U) is the wrapped phase.

    Note: Uses the wrapped phase intentionally — the gradient of the wrapped
    phase is well-defined and avoids unwrapping artifacts at sharp edges.
    """
    phase = np.angle(complex_field)
    gy, gx = np.gradient(phase)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def metric_phase_variance(complex_field: np.ndarray) -> float:
    """
    Variance of the unwrapped phase.
    At focus, the reconstructed phase of a particle becomes structured
    (ring pattern for a sphere), increasing the variance above the background
    noise level. Works best for large or phase-rich objects (Ma et al. 2004).

    Formula: F = Var(unwrap(angle(U)))

    Note: Requires phase unwrapping, which adds computation time and may fail
    for very noisy or wrapped phases. Phase Gradient is generally preferred
    for particle tracking.
    """
    phase = np.angle(complex_field)
    unwrapped = unwrap_phase(phase)
    return float(np.var(unwrapped))


def metric_spectral_energy(complex_field: np.ndarray) -> float:
    """
    High-frequency energy of the phase spectrum.
    At the focal plane, the phase map of a focused object concentrates energy
    in high spatial frequencies. The metric sums the power spectrum outside
    the inner 25% (by radius) of the frequency domain (Ferraro et al. 2003).

    Formula: F = Σ |FFT(φ)|²  for spatial frequencies outside the inner 25% radius.

    Note: The 25% radius threshold was chosen as a practical default. For objects
    that occupy a large fraction of the field of view, this boundary may need
    to be tuned.
    """
    phase = np.angle(complex_field)
    N, M = phase.shape
    F = np.abs(fftshift(fft2(phase)))**2
    cy, cx = N // 2, M // 2
    ry, rx = N // 4, M // 4
    Y, X = np.ogrid[:N, :M]
    inner = ((Y - cy)**2 / ry**2 + (X - cx)**2 / rx**2) <= 1.0
    return float(np.sum(F[~inner]))


# ── DISPATCHER ────────────────────────────────────────────────────────────────

AMPLITUDE_METRICS = {
    'Variance':   metric_variance,
    'Tenengrad':  metric_tenengrad,
    'Laplacian':  metric_laplacian,
}

PHASE_METRICS = {
    'Phase Gradient':   metric_phase_gradient,
    'Phase Variance':   metric_phase_variance,
    'Spectral Energy':  metric_spectral_energy,
}

ALL_METRICS = {**AMPLITUDE_METRICS, **PHASE_METRICS}


def compute_focus_metric(complex_field: np.ndarray,
                         domain: str,
                         metric_name: str) -> float:
    """
    Compute a focus metric for a reconstructed complex field at plane z_i.

    Parameters
    ----------
    complex_field : np.ndarray (complex)
        Reconstructed complex field at propagation distance z_i.
    domain : str
        'amplitude' — metric operates on |complex_field|.
        'phase'     — metric operates on angle(complex_field).
    metric_name : str
        Name of the metric. See AMPLITUDE_METRICS / PHASE_METRICS.

    Returns
    -------
    float
        Metric value. Higher = better focus (all metrics are defined this way).
    """
    if domain == 'amplitude':
        fn = AMPLITUDE_METRICS.get(metric_name)
        if fn is None:
            raise ValueError(f"Unknown amplitude metric: '{metric_name}'. "
                             f"Available: {list(AMPLITUDE_METRICS)}")
        return fn(np.abs(complex_field))
    else:  # phase
        fn = PHASE_METRICS.get(metric_name)
        if fn is None:
            raise ValueError(f"Unknown phase metric: '{metric_name}'. "
                             f"Available: {list(PHASE_METRICS)}")
        return fn(complex_field)
