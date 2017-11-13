from . import units


class kern():
    _xlss = None
    _cfac = None

    def __init__(self, cosmo):
        self.cosmo = cosmo


    @property
    def cfac(self):
        """Pre-factor for the CMB lensing kernel"""

        if self._cfac is None:
            self._cfac = 3. * self.cosmo.omm * \
                (self.cosmo.H0 * 1.e3 / units.c)**2
        return self._cfac


    @property
    def xlss(self):
        """Distance to the last-scattering surface in Mpc"""

        if self._xlss is None:
            self._xlss = self.cosmo.x_z(1100.)
        return self._xlss


    def w_lxz(self, l, x, z):
        return self.cfac * (1. + z) * (x / l)**2 * (1. / x - 1. / self.xlss)
