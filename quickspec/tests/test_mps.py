import numpy as np
from numpy import testing

from quickspec import mps, cosmo


class TestMps():

    mylcdm = cosmo.lcdm()
    mps_initial = mps.mps.initial_ps()
    mymps = mps.lin.mps_camb(mylcdm, mps_initial, nonlinear=True)

    def test_cosmo(self):
        assert self.mylcdm.omr == 0.
        assert self.mylcdm.omb == 0.05
        assert self.mylcdm.omc == 0.25
        assert self.mylcdm.oml == 0.7
        assert self.mylcdm.H0 == 70.

    def test_initvals(self):
        assert self.mps_initial.amp == 2.1e-9
        assert self.mps_initial.n_s == 0.95
        assert self.mps_initial.n_r == 0.
        assert self.mps_initial.k_pivot == 0.05

    def test_mps_camb_lin(self):
        kk = np.logspace(-4, 1, 5)
        zz = [0, 3, 10]
        mypkz = np.array([self.mymps.p_kz(kk, z=z) for z in zz])

        assert np.isfinite(mypkz).all()

        reference_pkz = np.array([[
            2.04892708e+03, 2.89853948e+04, 4.04059760e+04,
            1.49465958e+03, 1.31298960e+01],
            [  2.07644068e+02, 2.94844798e+03, 4.19199487e+03,
            4.73376622e+01, 5.86350087e-01],
            [  2.75856778e+01, 3.95178237e+02, 5.62569607e+02,
            4.37397611e+00, 3.26512142e-03]])

        testing.assert_allclose(mypkz, reference_pkz, rtol=1.e-3)
