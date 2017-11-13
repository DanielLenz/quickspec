import numpy as np
from numpy import testing

from quickspec import cosmo

class TestPlanck15():
    planck15 = cosmo.Planck15()

    def test_initvals(self):
        assert self.planck15.omr == 0.
        assert self.planck15.omb == 0.0486
        assert self.planck15.omc == 0.2589
        assert self.planck15.oml == 0.6925
        assert self.planck15.H0 == 67.7


class TestLCDM():

    lcdm = cosmo.LCDM()

    def test_initvals(self):
        assert self.lcdm.omr == 0.
        assert self.lcdm.omb == 0.05
        assert self.lcdm.omc == 0.25
        assert self.lcdm.oml == 0.7
        assert self.lcdm.H0 == 70.

    def vectors(self):
        assert(np.all(self.lcdm.zvec >= 0))
        assert(np.all(self.lcdm.xvec >= 0))

    def test_t_z(self):
        testing.assert_almost_equal(self.lcdm.t_z(0), 13.4514119711)
        testing.assert_almost_equal(self.lcdm.t_z(1), 5.74499627)

    def test_x_z(self):
        # x to z
        testing.assert_almost_equal(self.lcdm.x_z(1.), 3303.8288058874678)
        testing.assert_almost_equal(self.lcdm.x_z(1000.), 13660.5292969386)

        # z to x
        testing.assert_almost_equal(self.lcdm.z_x(100), 0.023474157001465847)
        testing.assert_almost_equal(self.lcdm.z_x(13.e3), 182.38416105113538)

        # Circular
        zvec = np.concatenate([
            np.linspace(0., 20., 500, endpoint=False),
            np.linspace(20., 200., 200, endpoint=False),
            np.linspace(200., 1500., 100)])

        testing.assert_array_almost_equal(
            zvec,
            self.lcdm.z_x(self.lcdm.x_z(zvec)))

    def test_H(self):
        # H_a
        assert self.lcdm.H_a(1.) ==  self.lcdm.H0
        testing.assert_almost_equal(self.lcdm.H_a(0.5),  123.24771803161305)

        # H_z
        assert self.lcdm.H_z(0.) == self.lcdm.H0
        testing.assert_almost_equal(self.lcdm.H_z(10),  1400)

        # H_x
        assert self.lcdm.H_x(0.) == self.lcdm.H0
        testing.assert_almost_equal(self.lcdm.H_x(1000), 79.286040172939693)

    def test_G(self):
        # G_z
        testing.assert_almost_equal(self.lcdm.G_z(0), 0.77898101676855247)
        testing.assert_almost_equal(self.lcdm.G_z(100), 0.0099009859519055138)

        # G_x
        testing.assert_almost_equal(self.lcdm.G_x(0), 0.77898101676855247)
        testing.assert_almost_equal(self.lcdm.G_x(1000), 0.68504721436132821)

    def test_Dv_mz(self):
        # Virial overdensity w.r.t. the mean matter density at redshift z.
        testing.assert_almost_equal(self.lcdm.Dv_mz(0), 337.14293073202816)
        testing.assert_almost_equal(self.lcdm.Dv_mz(100), 177.6530958454735)

    def test_aeg_lm(self):
        # scale factor at lambda - matter equality.
        testing.assert_almost_equal(self.lcdm.aeq_lm(), 0.7539474411291538)
