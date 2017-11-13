import numpy as np
from numpy import testing

from quickspec.cib import hall, halo
from quickspec.cib import ldp_2004 as ldp


class TestHall():

    def test_ssed_graybody(self):
        # 545 GHz, default Td=34., default beta=2.
        testing.assert_almost_equal(
            hall.ssed_graybody(545.e9)/1.e3,
            612112.7149841311)

        # 353 GHz, Td=20., default beta=2.
        testing.assert_almost_equal(
            hall.ssed_graybody(353.e9, Td=20.)/1.e3,
            60639.562228153124)

    def test_ssed(self):
        # 545 GHz
        testing.assert_almost_equal(
            hall.ssed(545.e9)/1.e3,
            612112.7149841311)

    def test_jbar(self):
        # 353 GHz, z=1, x=100
        testing.assert_almost_equal(
            hall.jbar(353e9, 1, 100),
            0.050333556661810691)
