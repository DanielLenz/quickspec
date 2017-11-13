import numpy as np
from numpy import testing

from quickspec import lens, cosmo


class TestLens():

    planck15 = cosmo.Planck15()
    # mps_initial = mps.mps.initial_ps()
    # mymps = mps.lin.mps_camb(mylcdm, mps_initial, nonlinear=True)

    def test_cosmo(self):
        assert self.planck15.omr == 0.
        assert self.planck15.omb == 0.0486
        assert self.planck15.omc == 0.2589
        assert self.planck15.oml == 0.6925
        assert self.planck15.H0 == 67.7


    def test_lens(self):
        kernel_lens = lens.kern(self.planck15)
        testing.assert_almost_equal(
            kernel_lens.w_lxz(100, 1000, 2),
            1.3105653069227149e-08)
