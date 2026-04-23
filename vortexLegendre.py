import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import hilbert
from scipy.sparse.linalg import svds
from skimage.restoration import unwrap_phase

def reference_wave(fx_max, fy_max, m, n, _lambda, dx, k, fx_0, fy_0, M, N, dy=None):
    """
    Generates the reference wave for off-axis DHM phase compensation.

    Parameters
    ----------
    fx_max, fy_max : float
        Frequency coordinates of the +1 diffraction order peak.
    m, n : ndarray
        Spatial coordinate meshgrids (columns and rows respectively).
    _lambda : float
        Wavelength (µm).
    dx : float
        Pixel size in x (µm). Used for both axes if dy is not provided.
    k : float
        Wavenumber 2π/λ.
    fx_0, fy_0 : float
        Center of the frequency domain (M/2, N/2).
    M, N : int
        Width and height of the field.
    dy : float, optional
        Pixel size in y (µm). Defaults to dx if not provided (square pixels).
    """
    if dy is None:
        dy = dx
    arg_x = (fx_0 - fx_max) * _lambda / (M * dx)
    arg_y = (fy_0 - fy_max) * _lambda / (N * dy)
    theta_x = np.arcsin(arg_x)
    theta_y = np.arcsin(arg_y)
    ref_wave = np.exp(1j * k * (dx * np.sin(theta_x) * m + dy * np.sin(theta_y) * n))
    return ref_wave

def spatial_filter(holo, M, N, save='Yes', factor=2.0, rotate:bool=False):
    # Apply Fourier transform to the hologram
    ft_holo = fftshift(fft2(fftshift(holo)))
    ft_holo[:5, :5] = 0  # suppress low-frequency components at the origin

    # Create a mask to eliminate the central DC component
    mask1 = np.ones((N, M), dtype=np.float32)
    mask1[int(N / 2 - 20):int(N / 2 + 20), int(M / 2 - 20):int(M / 2 + 20)] = 0
    ft_holo_I = ft_holo * mask1

    # Remove specular reflection or bright central peak
    mask1 = np.ones((N, M), dtype=np.float32)
    mask1[0, 0] = 0
    ft_holo_I *= mask1

    region_interest = ft_holo_I;
    # Select region of interest: left half of the spectrum
    if rotate:
        region_interest[:, :int(M / 2)] = 0
    else:
        region_interest[:, int(M / 2):-1] = 0

    # Find the peak in the region of interest (corresponding to +1 diffraction order)
    max_value = np.max(np.abs(region_interest))
    max_pos = np.where(np.abs(region_interest) == max_value)
    fy_max = max_pos[0][0]  # vertical coordinate
    fx_max = max_pos[1][0]  # horizontal coordinate

    # Compute distance from center of ROI to peak (for circular mask)
    distance = np.sqrt((fx_max - M / 2) ** 2 + (fy_max - N / 2) ** 2)
    resc = distance / factor  # define mask radius relative to peak location

    # Create circular mask centered on peak location
    Y, X = np.meshgrid(np.arange(M), np.arange(N))
    cir_mask = np.sqrt((X - fy_max) ** 2 + (Y - fx_max) ** 2) <= resc
    cir_mask = cir_mask.astype(np.float32)

    # Apply circular mask to filter the +1 order
    ft_holo_filtered = ft_holo * cir_mask
    holo_filtered = fftshift(ifft2(ifftshift(ft_holo_filtered)))

    # Optional visualization
    if save == 'Yes':
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(np.log1p(np.abs(ft_holo) ** 2), cmap='gray')
        plt.title('FT Hologram')
        plt.axis('equal')

        plt.subplot(1, 3, 2)
        plt.imshow(cir_mask, cmap='gray')
        plt.title('Circular Filter')
        plt.axis('equal')

        plt.subplot(1, 3, 3)
        plt.imshow(np.log1p(np.abs(ft_holo_filtered) ** 2), cmap='gray')
        plt.title('FT Filtered Hologram')
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig('filter_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    return ft_holo, holo_filtered, fx_max, fy_max, cir_mask

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def hilbert_transform_2d(c, hilbert_or_energy_operator=1):
    """
    hilbert_transform_2d(c, hilbert_or_energy_operator)

    Computes the 2D Hilbert Transform (Spiral Phase Transform) or Energy Operator.

    Parameters:
        c : 2D numpy array (real or complex)
            Input image or interferogram: c = b * cos(psi)
        hilbert_or_energy_operator : int
            If 1: computes i * exp(i * beta) * sin(psi)
            If 0: computes -b * exp(i * beta) * sin(psi)

    Returns:
        quadrature : 2D numpy array (complex)
            Quadrature signal (complex-valued)
    """
    NR, NC = c.shape
    u, v = np.meshgrid(np.arange(NC), np.arange(NR))
    u0 = NC // 2
    v0 = NR // 2

    u = u - u0
    v = v - v0

    # Avoid division by zero at the origin
    H = (u + 1j * v).astype(np.complex128)
    H /= (np.abs(H) + 1e-6)
    H[v0, u0] = 0

    C = fft2(c)

    if hilbert_or_energy_operator:
        CH = C * ifftshift(H)
    else:
        CH = C * ifftshift(1j * H)

    quadrature = np.conj(ifft2(CH))
    return quadrature


def vortex_compensation(field, fxOverMax, fyOverMax):
    cropVortex = 5  # Pixels for interpolation
    factorOverInterpolation = 55

    # Crop around the max frequency
    sd = field[
        int(fyOverMax - cropVortex) : int(fyOverMax + cropVortex),
        int(fxOverMax - cropVortex) : int(fxOverMax + cropVortex)
    ]

    # Hilbert transform
    sd_crop = hilbert_transform_2d(sd, hilbert_or_energy_operator=1)  # 2D Hilbert transform

    sz = np.abs(sd_crop).shape
    xg = np.arange(0, sz[0])
    yg = np.arange(0, sz[1])

    F_real = RegularGridInterpolator((xg, yg), np.real(sd_crop), bounds_error=False, fill_value=0)
    F_imag = RegularGridInterpolator((xg, yg), np.imag(sd_crop), bounds_error=False, fill_value=0)

    xq = np.arange(0, sz[0] - 1 / factorOverInterpolation + 1e-6, 1 / factorOverInterpolation)
    yq = np.arange(0, sz[1] - 1 / factorOverInterpolation + 1e-6, 1 / factorOverInterpolation)

    xv, yv = np.meshgrid(xq, yq, indexing='ij')
    pts = np.stack([xv.ravel(), yv.ravel()], axis=-1)

    vq = F_real(pts).reshape(xv.shape)
    vq2 = F_imag(pts).reshape(xv.shape)

    psi = np.angle(vq + 1j * vq2)

    n1, m1 = psi.shape
    Ml = np.zeros_like(psi)

    M1 = np.zeros_like(psi)
    M2 = np.zeros_like(psi)
    M3 = np.zeros_like(psi)
    M4 = np.zeros_like(psi)
    M5 = np.zeros_like(psi)
    M6 = np.zeros_like(psi)
    M7 = np.zeros_like(psi)
    M8 = np.zeros_like(psi)

    Y1 = np.arange(0, n1 - 2)
    Y2 = np.arange(1, n1 - 1)
    Y3 = np.arange(2, n1)
    X1 = np.arange(0, m1 - 2)
    X2 = np.arange(1, m1 - 1)
    X3 = np.arange(2, m1)

    M1[np.ix_(Y2, X2)] = psi[np.ix_(Y1, X1)]
    M2[np.ix_(Y2, X2)] = psi[np.ix_(Y1, X2)]
    M3[np.ix_(Y2, X2)] = psi[np.ix_(Y1, X3)]
    M4[np.ix_(Y2, X2)] = psi[np.ix_(Y2, X3)]
    M5[np.ix_(Y2, X2)] = psi[np.ix_(Y3, X3)]
    M6[np.ix_(Y2, X2)] = psi[np.ix_(Y3, X2)]
    M7[np.ix_(Y2, X2)] = psi[np.ix_(Y3, X1)]
    M8[np.ix_(Y2, X2)] = psi[np.ix_(Y2, X1)]

    D1 = wrap_to_pi(M2 - M1)
    D2 = wrap_to_pi(M3 - M2)
    D3 = wrap_to_pi(M4 - M3)
    D4 = wrap_to_pi(M5 - M4)
    D5 = wrap_to_pi(M6 - M5)
    D6 = wrap_to_pi(M7 - M6)
    D7 = wrap_to_pi(M8 - M7)
    D8 = wrap_to_pi(M1 - M8)

    Ml = (D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8) / (2 * np.pi)
    Ml = fftshift(Ml)
    Ml[70:, 70:] = 0
    Ml = ifftshift(Ml)

    linearIndex = np.argmin(Ml)
    yOverInterpolVortex, xOverInterpolVortex = np.unravel_index(linearIndex, Ml.shape)

    positions = []
    x_pos = (xOverInterpolVortex / factorOverInterpolation) + (fxOverMax - cropVortex)
    y_pos = (yOverInterpolVortex / factorOverInterpolation) + (fyOverMax - cropVortex)
    positions.append([x_pos, y_pos])

    return positions
    
def legendre_compensation(field_compensate, limit, RemovePiston=True, UsePCA=False):
    """
    Compensates the phase of a complex field using a fit with Legendre polynomials.

    Parameters:
    -----------
    field_compensate : np.ndarray
        Complex field to be corrected.
    limit : int
        Radius of the region to analyze around the center.
    RemovePiston : bool
        If True (default), removes the piston term by setting coefficient[0] = 0.
        If False, searches for the optimal piston value that minimizes phase variance.
    UsePCA : bool
        If True, uses SVD decomposition to extract the dominant wavefront.

    Returns:
    --------
    compensatedHologram : np.ndarray
        Phase-compensated complex field.
    Legendre_Coefficients : np.ndarray
        Coefficients of the Legendre polynomial fit.
    """

    # Centered Fourier transform
    fftField = fftshift(fft2(ifftshift(field_compensate)))

    A, B = fftField.shape
    center_A = int(round(A / 2))
    center_B = int(round(B / 2))

    start_A = int(center_A - limit)
    end_A = int(center_A + limit)
    start_B = int(center_B - limit)
    end_B = int(center_B + limit)

    fftField = fftField[start_A:end_A, start_B:end_B]
    square = ifftshift(ifft2(fftshift(fftField)))

    # Extract dominant wavefront
    if UsePCA:
        u, s, vt = svds(square, k=1, which='LM')
        dominant = u[:, :1] @ np.diag(s[:1]) @ vt[:1, :]
        dominant = unwrap_phase(np.angle(dominant))
    else:
        dominant = unwrap_phase(np.angle(square))

    # Normalized spatial grid
    gridSize = dominant.shape[0]
    coords = np.linspace(-1, 1 - 2 / gridSize, gridSize)
    X, Y = np.meshgrid(coords, coords)

    dA = (2 / gridSize) ** 2
    order = np.arange(1, 11)

    # Get orthonormal Legendre polynomial basis
    polynomials = square_legendre_fitting(order, X, Y)
    ny, nx, n_terms = polynomials.shape
    Legendres = polynomials.reshape(ny * nx, n_terms)

    zProds = Legendres.T @ Legendres * dA
    Legendres = Legendres / np.sqrt(np.diag(zProds))

    Legendres_norm_const = np.sum(Legendres ** 2, axis=0) * dA
    phaseVector = dominant.reshape(-1, 1)

    # Projection onto Legendre basis
    Legendre_Coefficients = np.sum(Legendres * phaseVector, axis=0) * dA

    if RemovePiston:
        # Zero out the piston coefficient and reconstruct the wavefront
        coeffs_used = Legendre_Coefficients.copy()
        coeffs_used[0] = 0.0
        coeffs_norm = coeffs_used / np.sqrt(Legendres_norm_const)
        wavefront = np.sum(coeffs_norm[:, np.newaxis] * Legendres.T, axis=0)
    else:
        # Search for the optimal piston value
        values = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
        variances = []

        for val in values:
            temp_coeffs = Legendre_Coefficients.copy()
            temp_coeffs[0] = val
            coeffs_norm = temp_coeffs / np.sqrt(Legendres_norm_const)
            wavefront = np.sum((coeffs_norm[:, np.newaxis]) * Legendres.T, axis=0)
            temp_holo = np.exp(1j * np.angle(square)) / np.exp(1j * wavefront.reshape(ny, nx))
            variances.append(np.var(np.angle(temp_holo)))

        best = values[np.argmin(variances)]
        Legendre_Coefficients[0] = best
        coeffs_norm = Legendre_Coefficients / np.sqrt(Legendres_norm_const)
        wavefront = np.sum(coeffs_norm[:, np.newaxis] * Legendres.T, axis=0)

    # Final phase compensation
    wavefront = wavefront.reshape(ny, nx)
    compensatedHologram = np.exp(1j * np.angle(square)) / np.exp(1j * wavefront)

    return compensatedHologram, Legendre_Coefficients


def square_legendre_fitting(order, X, Y):
    polynomials = []
    for i in order:
        if i == 1:
            polynomials.append(np.ones_like(X))
        elif i == 2:
            polynomials.append(X)
        elif i == 3:
            polynomials.append(Y)
        elif i == 4:
            polynomials.append((3 * X**2 - 1) / 2)
        elif i == 5:
            polynomials.append(X * Y)
        elif i == 6:
            polynomials.append((3 * Y**2 - 1) / 2)
        elif i == 7:
            polynomials.append((X * (5 * X**2 - 3)) / 2)
        elif i == 8:
            polynomials.append((Y * (3 * X**2 - 1)) / 2)
        elif i == 9:
            polynomials.append((X * (3 * Y**2 - 1)) / 2)
        elif i == 10:
            polynomials.append((Y * (5 * Y**2 - 3)) / 2)
        elif i == 11:
            polynomials.append((35 * X**4 - 30 * X**2 + 3) / 8)
        elif i == 12:
            polynomials.append((X * Y * (5 * X**2 - 3)) / 2)
        elif i == 13:
            polynomials.append(((3 * Y**2 - 1) * (3 * X**2 - 1)) / 4)
        elif i == 14:
            polynomials.append((X * Y * (5 * Y**2 - 3)) / 2)
        elif i == 15:
            polynomials.append((35 * Y**4 - 30 * Y**2 + 3) / 8)
    return np.stack(polynomials, axis=-1)
    
def fringes_normalization(hologram, R):
    M, N = hologram.shape
    u0 = N // 2
    v0 = M // 2

    u, v = np.meshgrid(np.arange(N), np.arange(M))
    u = u - u0
    v = v - v0

    H = 1 - np.exp(-(u ** 2 + v ** 2) / (2 * R ** 2))
    C = np.fft.fft2(hologram)
    CH = C * np.fft.ifftshift(H)
    ch = np.fft.ifft2(CH)

    ib = C * np.fft.ifftshift(np.exp(-(u ** 2 + v ** 2) / (2 * R ** 2)))
    background = np.real(np.fft.ifft2(ib))

    s = spiralTransform(ch)
    
    s = np.abs(s)

    fringeNorm = np.cos(np.arctan2(s, ch.real))

    modulation = np.abs(ch + 1j * s)


    return background, modulation, fringeNorm


def spiralTransform(c):
    """
    Computes the spiral phase transform of a complex-valued 2D array c.
    This corresponds to the quadrature component modulated by a spiral phase factor.

    Based on:
    Kieran G. Larkin, Donald J. Bone, and Michael A. Oldfield,
    "Natural demodulation of two-dimensional fringe patterns. I.
    General background of the spiral phase quadrature transform,"
    J. Opt. Soc. Am. A 18, 1862-1870 (2001)
    """

    try:
        TH = np.max(np.abs(c))
        if np.mean(np.real(c)) > 0.01 * TH:
            print("Warning: Input must be DC filtered")

        NR, NC = c.shape

        # Create normalized frequency coordinates in [-1, 1)
        x = np.linspace(-1, 1, NC, endpoint=False)
        y = np.linspace(-1, 1, NR, endpoint=False)
        X, Y = np.meshgrid(x, y)

        # Convert to polar coordinates
        Theta = np.arctan2(Y, X)

        # Spiral filter (vortex definition)
        H = np.exp(-1j * Theta)

        # Apply spiral filter in Fourier domain
        C = np.fft.fft2(c)
        CH = C * np.fft.ifftshift(H)

        # Inverse transform and apply complex conjugate (for coordinate system consistency)
        sd = np.conj(np.fft.ifft2(CH))


        return sd

    except Exception as e:
        raise e


