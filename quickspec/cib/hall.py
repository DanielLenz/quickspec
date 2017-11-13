"""
Implementation of the single-spectra-energy-distribution (SSED) model
from Hall et. al. 2010 (arxiv:0912.4315)

"""

import numpy as np
from scipy import constants as sc

from .. import units

alpha_mid_ir = -2.
nu_mid_ir = 4954611330474.7109


def ssed_graybody(nu, Td=34., beta=2):
    return nu**(beta) * units.planck(nu, Td)


def ssed(nu, Td=34., beta=2., alpha_mid_ir=alpha_mid_ir, nu_mid_ir=nu_mid_ir):
    """Calculation of the SSED f_{\nu} defined between pages 4 and 5 of
    Hall et al. (2010).

    """

    if np.isscalar(nu):
        if nu > nu_mid_ir:
            return (
                ssed_graybody(nu_mid_ir, Td, beta) /
                (nu_mid_ir)**(alpha_mid_ir) * nu**(alpha_mid_ir))
        else:
            return ssed_graybody(nu, Td, beta)
    else:
        ret = ssed_graybody(nu, Td, beta)
        ret[np.where(nu > nu_mid_ir)] = (
            ssed_graybody(nu_mid_ir, Td, beta) /
            (nu_mid_ir)**(alpha_mid_ir) * nu**(alpha_mid_ir))[np.where(
                nu > nu_mid_ir)]
        return ret


def jbar(
        nu, z, x, zc=2., sigmaz=2.,
        norm=7.5374829969423142e-15, ssed_kwargs={}):
    """Eq. 10 of Hall et al. (2010) nu in Hz, returns in Jansky."""


    return (
        1. / (1. + z) * x**2 * np.exp(-(z - zc)**2 / (2. * sigmaz**2)) *
        ssed(nu * (1. + z), **ssed_kwargs) * norm)


class ssed_kern():
    """
    CIB kernel $W^{\nu}$, given by Hall et. al. Eq. 5:
    $Cl = \int dz 1/H(z)/\chi(z)^2 (a\,b j(\nu, z))^2 P_{lin}(l/\chi,z)$

    For the bias $b$, we support a redshift dependence as
    $b_G = b_0 + b_1 z + b_2 z^2$.
    Furthermore, we provide a scale-dependent correction to that bias term,
    $b_{\rm eff} = b_G + b_{\rm corr}$. The correction term $b_{\rm corr}$ is
    taken from Eq. (1) of De Putter+ (2014) and scales with the amplitude
    of the primordial non-Gaussianity $f_{NL}$.
    """


    _b_G = None

    def __init__(
            self, nu,
            b0=1.0, b1=0., b2=0.,
            fnl=0., mps=None, jbar_kwargs={}, ssed_kwargs={}):

        self.nu = nu
        self.mps = mps

        # bias terms
        # the effective redshift dependent bias term is
        # b = b0 + b1*z + b2*z^2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2

        # non-gaussianity
        if fnl < 0.:
            raise ValueError('fnl must be >= 0.')
        self.fnl = fnl

        # kwargs for the jbar and ssed term
        self.jbar_kwargs = jbar_kwargs
        self.ssed_kwargs = ssed_kwargs

    def get_b_eff(self, l, x, z):

        b_G = self.get_b_G(z)

        # Scale-dependent correction for fnl > 0
        if self.fnl == 0.:
            b_k = 0.
        else:
            b_k = self.get_b_k(l, x, z, b_G)

        # The effective bias is the sum of the simple redshift-dependent bias
        # and the scale-dependent correction
        b_eff = b_G + b_k

        return b_eff


    def get_b_G(self, z):
        b_G = self.b0 + self.b1 * z + self.b2 * z * z
        return b_G

    def get_b_k(self, l, x, z, b_G):
        """Scale-dependent correction to the linear halo bias, taken from
        De Putter+ (2014).
        """

        k = l / x

        delta_c = 1.686  # critical overdensity

        b_k = (
            b_G +
            self.fnl * (b_G - 1.) * delta_c * 3. * self.mps.cosmo.omm *
            self.mps.cosmo.H0**2 / (sc.c / 1.e3)**2 / k**2 / self.mps.T_k(k) /
            self.mps.cosmo.G_z(z))

        return b_k



    def w_lxz(self, l, x, z):
        """The actual CIB kernel W

        """

        return (
            1. / (1. + z) * self.get_b_eff(l, x, z) * jbar(
                self.nu, z, x,
                ssed_kwargs=self.ssed_kwargs, **self.jbar_kwargs))
