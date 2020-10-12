import numpy as np
from lmfit.lineshapes import lorentzian, skewed_voigt, tiny
from scipy.special import erf


def cos(angle):
    """Cosine in degrees"""
    return np.cos(angle*np.pi/180.)


def sin(angle):
    """Sine in degrees"""
    return np.sin(angle*np.pi/180.)


def paramagnon(x, amplitude, center, sigma, res, kBT):
    """Damped harmonic oscillator convolved with resolution
    
    Parameters
    ----------
    x : array
        independent variable
    amplitude: float
        peak height
    center : float
        Plot of osciallator -- not the same as the peak
    sigma : float
        damping
        This corresponds to HWHM when sigma<<center
    res : float
        Resolution -- sigma parameter of Gaussian resolution used
        for convolution. FWHM of a gaussian is 2*np.sqrt(2*np.log(2))=2.355
    kBT : float
        Temperature for Bose factor.
        kBT should be in the same units as x
        n.b. kB = 8.617e-5 eV/K

    Form of equation from https://journals.aps.org/prb/pdf/10.1103/PhysRevB.93.214513
    (eq 4)
    
    N.B. This should be updated to use the function we contributed to lmfit at some point.
    """
    step = min(np.abs(np.mean(np.diff(x))), sigma/20, res/20)
    x_paramagnon = np.arange(np.min(x) - res*10, np.max(x) + res*10, step)

    chi_paramagnon = (2*x_paramagnon*amplitude*sigma*center /
                     (( x_paramagnon**2 - center**2)**2 +
                      (x_paramagnon*sigma)**2 ))

    kernal = make_gaussian_kernal(x_paramagnon, res)
    y_paramagnon = convolve(chi_paramagnon * bose(x_paramagnon, kBT), kernal)
    return np.interp(x, x_paramagnon, y_paramagnon)


def paramagnon_integrated_I(x, amplitude, center, sigma, res, kBT):
    """Damped harmonic oscillator convolved with resolution
    
    Parameters
    ----------
    x : array
        independent variable
    amplitude: float
        Integrated intensity prior to convolution
    center : float
        Plot of osciallator -- not the same as the peak
    sigma : float
        damping
        This corresponds to HWHM when sigma<<center
    res : float
        Resolution -- sigma parameter of Gaussian resolution used
        for convolution. FWHM of a gaussian is 2*np.sqrt(2*np.log(2))=2.355
    kBT : float
        Temperature for Bose factor.
        kBT should be in the same units as x
        n.b. kB = 8.617e-5 eV/K

    Form of equation from https://journals.aps.org/prb/pdf/10.1103/PhysRevB.93.214513
    (eq 4)
    
    N.B. This should be updated to use the function we contributed to lmfit at some point.
    """
    step = min(np.abs(np.mean(np.diff(x))), sigma/20, res/20)
    x_paramagnon = np.arange(np.min(x) - res*10, np.max(x) + res*10, step)

    y0 = (2*x_paramagnon*sigma*center /
          (( x_paramagnon**2 - center**2)**2 +
           (x_paramagnon*sigma)**2 )) * bose(x_paramagnon, kBT)
    
    y1 = amplitude*y0/np.trapz(y0, x=x_paramagnon)
    
    kernal = make_gaussian_kernal(x_paramagnon, res)
    y_paramagnon = convolve(y1, kernal)
    return np.interp(x, x_paramagnon, y_paramagnon)


def magnon(x, amplitude, center, sigma, res, kBT):
    """Return a 1-dimensional Antisymmeterized Lorenzian multiplied by Bose factor
    and convolved with resolution.
    magnon(x, amplditude, center, sigma, res, kBT) =
        convolve(antisymlorz(x, amplitude, center, sigma)*bose(x, kBT), kernal)
    """
    step = min(np.abs(np.mean(np.diff(x))), sigma/20., res/20.)
    x_magnon = np.arange(np.min(x) - res*10, x.max() + res*10, step)
    kernal = make_gaussian_kernal(x_magnon, res)
    y_magnon = convolve(antisymlorz(x_magnon, amplitude, center, sigma)*bose(x_magnon, kBT), kernal)
    return np.interp(x, x_magnon, y_magnon)


def bose(x, kBT):
    """Return a 1-dimensinoal Bose factor function
    kBT should be in the same units as x
    bose(x, kBT) = 1 / (1 - exp(-x / kBT) )

    n.b. kB = 8.617e-5 eV/K
    """
    return np.real(1./ (1 - np.exp(-x / (kBT)) +0.0001*1j ))


def make_gaussian_kernal(x, sigma):
    """Return 1-dimensional normalized Gaussian kernal suitable for performing a convolution
    This ensures even steps mirroring x.
    make_gaussian_kernal(x, sigma) =
        exp(-x**2 / (2*sigma**2))
    """
    step = np.abs(np.mean(np.diff(x)))
    x_kern = np.arange(-sigma*10, sigma*10, step)
    y = np.exp(-x_kern**2/(2 * sigma**2))
    return y / np.sum(y)


def convolve(y, kernal):
    """ Convolve signal y with kernal. """
    return np.convolve(y, kernal, mode='same')


def zero2Linear(x, center=0, sigma=1, grad=1):
    """Return 1-dimension function that goes from zero << x0 to linear >> x0
    the cross over is smooth as controlled by gaussian convolution of width x
    zero2Linear(x, x0, sigma, grad) =
    0       : x<center
    (x-center)*grad  : x>center
    convolved by gaussian(x, sigma)
    """
    step = min(np.abs(np.mean(np.diff(x))), sigma) / 20
    x_linear = np.arange(-20*sigma+np.min(x), 20*sigma+np.max(x), step)
    y_linear = (x_linear-center)*grad
    y_linear[x_linear<center] = 0
    y_linear = convolve(y_linear, make_gaussian_kernal(x_linear, sigma))
    return np.interp(x, x_linear, y_linear)


def zero2Quad(x, center=0, sigma=1, quad=1):
    """Return 1-dimension function that goes from zero << x0 to quadratic >> x0
    the cross over is smooth as controlled by gaussian convolution of width x
    zero2Linear(x, x0, sigma, grad) =
    0                       : x<center
    (x-center)**2 * quad    : x>center
    convolved by gaussian(x, sigma)
    """
    step = min(np.abs(np.mean(np.diff(x))), sigma) / 20
    x_quad = np.arange(-20*sigma+np.min(x), 20*sigma+np.max(x), step)
    y_quad = (x_quad-center)**2 * quad
    y_quad[x_quad<center] = 0
    y_quad = convolve(y_quad, make_gaussian_kernal(x_quad, sigma))
    return np.interp(x, x_quad, y_quad)


def antisymlorz(x, amplitude=0.1, center=0.15, sigma=0.05):
    """ Return Antisymmeterized Lorentzian
    antisymlorz(x, amplitude, center, sigma) =
        lorentzian(x, amplitude, center, sigma) - lorentzian(x, amplitude, -center, sigma)
    """
    chi = lorentzian(x, amplitude, center, sigma) - lorentzian(x, amplitude, -center, sigma)
    return chi


def plane2D(X, Y, C=0, slopex=0., slopey=0.):
    """ Return a 3-dimensional plane
    plane2D(X, Y, C=0, slopex=0., slopey=0., slopez=0.)
    I = C + X*slopex + Y*slopey
    """
    I = C + X*slopex + Y*slopey
    return I


def plane3D(X, Y, Z, C=0, slopex=0., slopey=0., slopez=0.):
    """ Return a 3-dimensional plane
    plane_3D(X, Y, Z, C=0, slopex=0., slopey=0., slopez=0.)
    C + X*slopex + Y*slopey + Z*slopez
    """
    I = C + X*slopex + Y*slopey + Z*slopez
    return I


def plane3Dcentered(X, Y, Z, C=0, centerx=0., centery=0., centerz=0.,
                     slopex=0., slopey=0., slopez=0.):
    """ Return a 3-dimensional plane
    plane_3D(X, Y, Z, C=0, slopex=0., slopey=0., slopez=0.)
    C + X*slopex + Y*slopey + Z*slopez
    """
    I = C + (X - centerx)*slopex + (Y - centery)*slopey + (Z - centerz) *slopez
    return I


def lorentzianSq2D(X, Y, amplitude=1.,
                     centerx=0., centery=0.,
                     sigmax=1., sigmay=1.):
    """ Return a 2-dimensional lorentzian squared function
    I = amplitude**2/(2*np.pi) * 1/(1 +
        (((X-centerx)/sigmax)**2 + ((Y-centery)/sigmay)**2))
    """
    I = amplitude**2/(2*np.pi) * 1/(1 +
        (((X-centerx)/sigmax)**2 + ((Y-centery)/sigmay)**2))
    return I


def lorentzianSq2DRot(X, Y, amplitude=1.,
                     centerx=0., centery=0.,
                     sigmax=1., sigmay=1., angle=0.):
    """ Return a 2-dimensional lorentzian squared function rotated by angle
    I = amplitude**2/(2*np.pi) * 1/(1 +
        (((X-centerx)/sigmax)**2 + ((Y-centery)/sigmay)**2))
    Note that angle is in degrees
    """
    Xrot = (X - centerx) * cos(angle) - (Y - centery) * sin(angle)
    Yrot = (X - centerx) * sin(angle) + (Y - centery) * cos(angle)

    I = amplitude**2/(2*np.pi) * 1/(1 +
        ((Xrot/sigmax)**2 + (Yrot/sigmay)**2))**2

    return I


def lorentzianSq3D(X, Y, Z, amplitude=1.,
                     centerx=0., centery=0., centerz=0.,
                     sigmax=1., sigmay=1., sigmaz=1.):
    """ Return a 3-dimensional lorentzian squared function
    I = 1/(1 + (((X-centerx)/sigmax)**2 +
                        ((Y-centery)/sigmay)**2 +
                        ((Z-centerz)/sigmaz)**2))
    I = amplitude**2/(2*np.pi) * I ** 2
    """
    I = 1/(1 + (((X-centerx)/sigmax)**2 +
                        ((Y-centery)/sigmay)**2 +
                        ((Z-centerz)/sigmaz)**2))
    I = amplitude**2/(2*np.pi) * I ** 2
    return I


def lorentzianSq3DRot(X, Y, Z, amplitude=1.,
                     centerx=0., centery=0., centerz=0.,
                     sigmax=1., sigmay=1., sigmaz=1., angle=0):
    """ Return a 3-dimensional lorentzian squared function, allow the peak rotated in x-y plane
    Xrot = (X - centerx) * cos(angle) - (Y - centery) * sin(angle)
    Yrot = (X - centerx) * sin(angle) + (Y - centery) * cos(angle)
    Z = Z-centerz

    I = 1/(1 + (((Xrot)/sigmax)**2 +
                        ((Yrot)/sigmay)**2 +
                        ((Z)/sigmaz)**2))
    I = amplitude**2/(2*np.pi) ** I **2
    """

    Xrot = (X - centerx) * cos(angle) - (Y - centery) * sin(angle)
    Yrot = (X - centerx) * sin(angle) + (Y - centery) * cos(angle)
    Z = Z-centerz

    I = 1/(1 + ((Xrot/sigmax)**2 + (Yrot/sigmay)**2 + (Z/sigmaz)**2))

    I = amplitude**2/(2*np.pi) * I ** 2
    return I


def error(x, amplitude=1., center=0., sigma=1.):
    """Return a normalized error function.
    amplitude/2*erf((x-center)/sigma) + 0.5

    For amplitude=1 sigma>0 this crosses over from 0 to 1 with increasing x
    Vice verse for sigma<0 """
    return amplitude/2*erf((x-center)/sigma) + 0.5


def power_decay(x, amplitude=1, center=0, eta=1):
    """Step and decay function
    
    Parameters
    ----------
    x : array
        independent variable
    amplitude : float
        factor proportional to height 
    center : float
        pole of intensity onset
    eta : float
        decay factor for continuum
    
    Returns
    -------
    y : array
        dependent variable
    """
    x0sq = (x**2 - center**2)
    decay_factor = np.sign(x0sq)/np.abs(x0sq)**(1-eta/2)
    if isinstance(decay_factor, (int, float)):
        decay_factor = max(tiny, decay_factor)
    else:
        decay_factor[np.where(np.isnan(decay_factor))] = tiny
        decay_factor[np.where(decay_factor <= tiny)] = tiny

    return amplitude*np.heaviside(x-center, 1)*decay_factor


def power_decay_convolved(x, amplitude=1, center=0, eta=1, res=1):
    """Step and decay function
    
    Parameters
    ----------
    x : array
        independent variable
    amplitude : float
        factor proportional to height 
    center : float
        pole of intensity onset
    eta : float
        decay factor for continuum
    res : float
        Resolution -- sigma parameter of Gaussian resolution used
        for convolution. FWHM of a gaussian is 2*np.sqrt(2*np.log(2))=2.355
    
    Returns
    -------
    y : array
        dependent variable
    """
    step = min(np.abs(np.mean(np.diff(x)))/5, np.abs(res)/20)
    x_continuum = np.arange(np.min(x) - res*10, np.max(x) + res*10, step)
    
    raw_continuum = power_decay(x_continuum, amplitude=amplitude,
                                center=center, eta=eta)

    kernal = make_gaussian_kernal(x_continuum, res)
    y_continuum = convolve(raw_continuum, kernal)
    return np.interp(x, x_continuum, y_continuum)


def Bmn(m, n, g):
    """Franck-Condon factor
    
    Parameters
    ----------
    m : int
    n : int
    g : float
        unitless electron phonon coupling factor
    
    Returns
    -------
    factor : float
        Value of Franck-Condon factor
    """
    fac = np.math.factorial
    prefac = (-1)**m*np.sqrt(np.exp(-g)*fac(m)*fac(n))
    sumterm = sum((-g)**l*np.sqrt(g)**(m - n)/(fac(n - l) * fac(l) * fac(m - n + l))
                 for l in range(n + 1))
    return prefac*sumterm


def phonon_amplitude(nprime, g, omega0, omega_det, Gamma, terms=100):
    """Amplitude of a phonon within Ament model
    [Europhysics Lett. 95, 27008 (2011)]

    Parameters
    ----------
    nprime : int
        Phonon index
    g : float
        unitless electron phonon coupling factor
    omega0 : float
        Phonon energy
    omega_det : float
        Energy detuning from resonance
    Gamma : float
        Core-hole lifetime (half width)
    terms : int
        Number of terms to use in infinite sum
        Default (100)

    Returns
    -------
    A : float
        Amplitude (i.e. square root intensity ) for 
        phonon
    """
    A = sum(Bmn(max(nprime, n), min(nprime, n), g)*Bmn(n, 0, g)
            /(omega_det + 1j*Gamma + (g - n)*omega0)
           for n in range(terms))
    return A


def phonons(x, g=1., omega0=1., sigma=1., omega_det=0., numphonons=15, Gamma=1.):
    """Spectrum of phonons within Ament model
    [Europhysics Lett. 95, 27008 (2011)]

    Parameters
    ----------
    x : array
        independent variable
    g : float
        unitless electron phonon coupling factor
    omega0 : float
        Phonon energy
    sigma : float
        Lorentizan phonon width
    omega_det : float
        Energy detuning from resonance
    numphonons : int
        Number of phonons
    Gamma : float

    Returns
    -------
    A : float
        Amplitude (i.e. square root intensity ) for 
        phonon
    """
    modes = []
    for nprime in range(1, numphonons + 1):
        A = phonon_amplitude(nprime, g, omega0, omega_det, Gamma, terms=100)
        modes.append(lorentzian(x, amplitude=A, center=omega0*nprime, sigma=sigma))
        
    return np.abs(sum(modes))**2


def skewed_voigt_convolved(x, amplitude=1.0, center=0.0, sigma=1.0,
                           gamma=None, skew=0.0, res=1.):
    """skewed_voigt convolved with resolution
    
    Parameters
    ----------
    x : array
        independent variable
    amplitude: float
        see lmfit.skewed_voigt
    center : float
        see lmfit.skewed_voigt
    sigma : float
        see lmfit.skewed_voigt
    gamma : float
        see lmfit.skewed_voigt
    skew : float
        see lmfit.skewed_voigt
    res : float
        Resolution -- sigma parameter of Gaussian resolution used
        for convolution. FWHM of a gaussian is 2*np.sqrt(2*np.log(2))=2.355
    """
    if gamma is None:
        step = min(np.abs(np.mean(np.diff(x))), sigma/20, res/20)
    else:
        step = min(np.abs(np.mean(np.diff(x))), sigma/20, gamma/20, res/20)
    x_ = np.arange(np.min(x) - res*10, np.max(x) + res*10, step)

    y_ = skewed_voigt(x_, amplitude, center, sigma, gamma, skew)
        
    kernal = make_gaussian_kernal(x_, res)
    y_convolved = convolve(y_, kernal)
    return np.interp(x, x_, y_convolved)