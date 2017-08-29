import numpy as np
import scipy.integrate
import scipy.interpolate

from . import units


class lcdm():
    """
    Class to encapsulate a flat, lambda-cold-dark-matter (lcdm) cosmology.

    """

    def __init__(self, omr=0.0, omb=0.05, omc=0.25, oml=0.7, H0=70.):
        """
        Initialize the flat lcdm cosmology. parameters:
               * omr = omega radiation.
               * omb = omega baryon.
               * omc = omega cold dark matter.
               * oml = omega lambda.
               * H0  = hubble constant (km/s/Mpc).

       """

        # check for flatness
        if not ((omb + omc + omr + oml) == 1.0):
            raise ValueError('Sum of densities must be equal to 1.')

        self.omr = omr  # radiation
        self.omb = omb  # baryons
        self.omc = omc  # CDM
        self.oml = oml  # dark energy, Lambda
        self.H0 = H0  # Hubble constant
        self.h = H0 / 100.  # h

        self.omm = omb + omc  # total matter density

        # redshift vector
        self.zvec = np.concatenate([
            np.linspace(0., 20., 500., endpoint=False),
            np.linspace(20., 200., 200., endpoint=False),
            np.linspace(200., 1500., 100)])

        # vector of comoving distance
        self.xvec = np.array([
            scipy.integrate.quad(
                lambda z: (units.c * 1.e-3) / self.H_z(z), 0., zmax)[0]
            for zmax in self.zvec])

        self.zmin = np.min(self.zvec)
        self.zmax = np.max(self.zvec)
        self.xmin = np.min(self.xvec)
        self.xmax = np.max(self.xvec)

        # conversions between conformal distance to redshift
        self.spl_x_z = scipy.interpolate.UnivariateSpline(
            self.zvec, self.xvec, k=3, s=0)
        self.spl_z_x = scipy.interpolate.UnivariateSpline(
            self.xvec, self.zvec, k=3, s=0)

    def t_z(self, z):
        """
        Returns the age of the Universe (in Gyr) at redshift z.

        """

        # da/dt / a = H_a
        # da / H_a / a = dt
        # /int_{a=0}^{a(z)} da / H_a / a = t
        # H0 = km/s/Mpc * 1Mpc/1e6pc * 1e3m/km * 3.08e16pc / m
        # 1Mpc = 3.25e6 ly
        def get_age(a):
            age = 1. / (self.H_a(a) / 3.08e19) / a / (365 * 24. * 60. * 60.)
            return age

        lower = 1.e-10
        upper = 1. / (1. + z)
        t_z = scipy.integrate.quad(get_age, lower, upper)[0] / 1.e9

        return t_z

    def x_z(self, z):
        """
        Returns the conformal distance (in Mpc) to redshift z.

        """

        assert(np.all(z >= self.zmin))
        assert(np.all(z <= self.zmax))

        return self.spl_x_z(z)

    def z_x(self, x):
        """
        Returns the redshift z at conformal distance x (in Mpc).

        """

        assert(np.all(x >= self.xmin))
        assert(np.all(x <= self.xmax))
        return self.spl_z_x(x)

    def H_a(self, a):
        """
        Returns the Hubble factor H(a) at scale factor a=1/(1+z).

        """

        return self.H0 * np.sqrt(
            self.oml + self.omm * a**(-3) + self.omr * a**(-4))

    def H_z(self, z):
        """
        Returns the Hubble factor H(z) at redshift z.

        """

        return self.H_a(1. / (1. + z))

    def H_x(self, x):
        """
        Returns the Hubble factor H(x) at conformal distance x (in Mpc).

        """

        return self.H_z(self.z_x(x))

    def G_z(self, z):
        """
        Returns the growth factor G(z) at redshift z (Eq. 7.77 of Dodelson).

        """
        if np.isscalar(z) or (np.size(z) == 1):
            integral = scipy.integrate.quad(
                lambda a: (self.H0 / (a * self.H_a(a)))**3,
                0,
                1. / (1. + z))[0]
            Gz = 2.5 * self.omm * self.H_a(1. / (1. + z)) / self.H0 * integral
            return Gz
        else:
            return [self.G_z(tz) for tz in z]

    def G_x(self, x):
        """
        Returns the growth factor at conformal distance x (in Mpc)
        (Eq. 7.77 of Dodelson).

        """

        return self.G_z(self.z_x(x))

    def Dv_mz(self, z):
        """
        Returns the virial overdensity w.r.t. the mean matter density at
        redshift z.
        Based on:
            * Bryan & Norman (1998) ApJ, 495, 80.
            * Hu & Kravtsov (2002) astro-ph/0203169 Eq. C6.

        """

        den = self.oml + self.omm * (1. + z)**3 + self.omr * (1. + z)**4

        omm = self.omm * (1. + z)**3 / den
        omr = self.omr * (1.0 + z)**4 / den

        # sanity check that omr is negligible at this redshift.
        assert(omr < 1.e-2)

        return (18. * np.pi**2 + 82. * (omm - 1.) - 39 * (omm - 1.)**2) / omm

    def aeq_lm(self):
        """
        Returns the scale factor at lambda - matter equality.

        """

        return 1. / (self.oml / self.omm)**(1. / 3.)
