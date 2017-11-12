import numpy as np
from scipy import integrate


def w_k_tophat(k):
    """ return the Fourier Transform of a tophat window function. """
    return 3. / k**3 * (np.sin(k) - k * np.cos(k))


def dw_k_tophat(k):
    """
    Return the derivative w.r.t. k of the fourier transform of a tophat
    window function.

    """
    return -9. / k**4 * (np.sin(k) - k * np.cos(k)) + 3. / k**2 * np.sin(k)


class initial_ps(object):
    """
    Primordial power spectrum, characterized by
    ln P = ln(amp) + (n_s -1)*ln(k/k_pivot) + n_r/2 * ln(k/k_pivot)^2

    """

    def __init__(self, amp=2.1e-9, n_s=0.95, n_r=0.0, k_pivot=0.05):

        self.amp = amp
        self.n_s = n_s
        self.n_r = n_r
        self.k_pivot = k_pivot

    def deltasq_k(self, k):
        return (
            self.amp * np.exp((self.n_s - 1.) * np.log(k / self.k_pivot) +
            0.5 * self.n_r * np.log(k / self.k_pivot)**2))

    def pR_k(self, k):
        return 2. * np.pi**2. / k**3. * self.deltasq_k(k)


class mps():
    """
    Base class for a matter power spectrum.

    """
    def __init__(self):
        pass

    def p_kx(self, k, x):
        """
        Returns the amplitude of the matter power spectrum at wavenumber
        k (in Mpc^{-1}) and conformal distance x (in Mpc).

        """
        return self.p_kz(k, self.cosmo.z_x(x))

    def sigma_rz(self, r, z, nk=10000):
        """
        Returns the variance in a smooth density field at redshift z on
        scale r (in Mpc).

        Reference:
            Cooray and Sheth (2002, arxiv:0206508), Eq. 18
            \sigma(r) = \sqrt{ \int dk k^2 P(k) / (2*pi^2) |W(k*r)|^2 }

             W is a tophat with radius r.
             Use dlnk for integration, nk parameter controls number of points.

        """
        def integrand(lnk):
            k = np.exp(lnk)
            kr = k * r
            return (np.sin(kr) - kr * np.cos(kr))**2 / kr**3 * self.p_kz(k, z)

        dlnk = (np.log(self.kmax) - np.log(self.kmin)) / nk
        lnks = np.arange(0, nk) * dlnk + np.log(self.kmin)
        sigma2 = integrate.simps(integrand(lnks), dx=dlnk)
        sigma2 *= 9. / (r**3 * 2. * np.pi**2)

        return np.sqrt(sigma2)

    def dsigma2_rz(self, r, z, nk=10000):
        """
        Returns the derivative of sigma_rz with respect to r.

        """

        def integrand(lnk):
            k = np.exp(lnk)
            kr = k * r
            sinkr = np.sin(kr)
            kr_coskr = kr * np.cos(kr)
            return (
                -9. * (sinkr - kr_coskr) /
                kr**3 + 3. / kr * sinkr) * (sinkr - kr_coskr) * self.p_kz(k, z)

        dlnk = (np.log(self.kmax) - np.log(self.kmin)) / nk
        lnks = np.arange(0, nk) * dlnk + np.log(self.kmin)
        dsigma2 = integrate.simps(integrand(lnks), dx=dlnk)
        dsigma2 *= 3. / (r**4 * np.pi**2)

        return dsigma2

    def cl_limber_xl(self, l, k1, k2=None, xmin=0.0, xmax=13000.):
        """
        Calculate the cross-spectrum at multipole l between kernels k1 and
        k2 in the limber approximation. The distance integral is performed
        from conformal distance xmin to xmax (both in Mpc).

        """
        if k2 is None:
            k2 = k1

        def integrand(x):
            z = self.cosmo.z_x(x)
            return (
                1. / x**2 * k1.w_lxz(l, x, z) *
                k2.w_lxz(l, x, z) * self.p_kz(l / x, z))

        return integrate.quad(integrand, xmin, xmax, limit=100)[0]

    def cl_limber_x(
            self,
            k1, k2=None,
            ls=None,
            xmin=0., xmax=13000.):

        if ls is None:
            ls = np.arange(20, 2048, 20)

        powerspec = np.array([
            self.cl_limber_xl(l, k1, k2, xmin, xmax) for l in ls])

        return powerspec

    def cl_limber_zl(self, l, k1, k2=None, zmin=0.0, zmax=1100.):
        """
        Calculate the cross-spectrum at multipole l between kernels k1 and k2
        in the limber approximation. The distance integral is performed from
        redshifts zmin to zmax.

        """
        if k2 is None:
            k2 = k1

        def integrand(z):
            x = self.cosmo.x_z(z)
            return (
                1. / x**2 / self.cosmo.H_z(z) * 3.e5 * k1.w_lxz(l, x, z) *
                k2.w_lxz(l, x, z) * self.p_kz(l / x, z))

        return integrate.quad(integrand, zmin, zmax, limit=100)[0]

    def cl_limber_z(
            self,
            k1, k2=None,
            ls=None,
            zmin=0., zmax=1100.):

        if ls is None:
            ls = np.arange(20, 2048, 20)

        powerspec = np.array([
            self.cl_limber_zl(l, k1, k2, zmin, zmax) for l in ls])

        return powerspec


    def T_k(self, k):
        T_k = np.sqrt(self.p_kz(k, z=0) / self.sips.pR_k(k)) / k**2
        T_k /= T_k.max()  # normalize to 1

        return T_k


class mps_lin(mps):
    pass
