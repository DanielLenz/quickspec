from . import units


class kern():
    def __init__(self, cosmo, dndz, b=1.):
        """
        Redshift kernel for a galaxy survey.

        Input
        -----
        cosmo: quickspec.cosmo.lcdm object
            Describes the cosmology.
        dndz: Function
            Function dndz(z) which returns # galaxies per sr per z.
        b: float
            Linear bias parameter.

        """

        self.cosmo = cosmo
        self.dndz = dndz
        self.b = b

        self.cfac = 3. * cosmo.omm * (cosmo.H0 * 1.e3 / units.c)**2

    def w_lxz(self, l, x, z):
        return self.b * (self.cosmo.H_z(z) * 1.e3 / units.c) * self.dndz(z)
