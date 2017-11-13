import numpy as np
from numpy import testing

from quickspec import mps, cosmo


class TestMpsCamb():

    planck15 = cosmo.Planck15()
    mps_initial = mps.mps.initial_ps()
    mymps = mps.lin.mps_camb(planck15, mps_initial, nonlinear=True)

    def test_cosmo(self):
        assert self.planck15.omr == 0.
        assert self.planck15.omb == 0.0486
        assert self.planck15.omc == 0.2589
        assert self.planck15.oml == 0.6925
        assert self.planck15.H0 == 67.7

    def test_initvals(self):
        assert self.mps_initial.amp == 2.1e-9
        assert self.mps_initial.n_s == 0.95
        assert self.mps_initial.n_r == 0.
        assert self.mps_initial.k_pivot == 0.05

    def test_mps_camb(self):
        kk = np.logspace(-4, 1, 5)
        zz = [0, 3, 10]
        mypkz = np.array([self.mymps.p_kz(kk, z=z) for z in zz])

        assert np.isfinite(mypkz).all()

        reference_pkz = np.array([
        [2.25445737e+03,   3.17225303e+04,   4.23385525e+04,
          1.55006710e+03,   1.31187579e+01],
        [2.25947093e+02,   3.19182089e+03,   4.34380063e+03,
          4.75190250e+01,   5.83800043e-01],
        [2.99946233e+01,   4.27653414e+02,   5.82655292e+02,
          4.36329754e+00,   3.21849119e-03]])

        testing.assert_allclose(mypkz, reference_pkz, rtol=1.e-3)
