# -*- coding: utf-8 -*-

import pytest
import numpy as np
from scipy.special import gamma as gamma

from lasy.laser import Laser
from lasy.profiles import CombinedLongitudinalTransverseProfile, GaussianProfile
from lasy.profiles.longitudinal import GaussianLongitudinalProfile
from lasy.profiles.transverse import (
    GaussianTransverseProfile,
    LaguerreGaussianTransverseProfile,
    SuperGaussianTransverseProfile,
    HermiteGaussianTransverseProfile,
)


@pytest.fixture(scope="function")
def gaussian():
    # Cases with Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 30.0e-15  # s
    w0 = 5.0e-6  # m
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

    return profile


def test_transverse_profiles_rt():
    npoints = 2000
    w0 = 10.0e-6

    # GaussianTransverseProfile
    print("GaussianTransverseProfile")
    std_th = w0 / np.sqrt(2)
    profile = GaussianTransverseProfile(w0)
    r = np.linspace(0, 6 * w0, npoints)
    field = profile.evaluate(r, np.zeros_like(r))
    std = np.sqrt(np.average(r**2, weights=np.abs(field)))
    print("\nstd_th = ", std_th)
    print("std = ", std)
    assert np.abs(std - std_th) / std_th < 0.01

    # LaguerreGaussianLaserProfile
    print("LaguerreGaussianLaserProfile")
    p = 2
    m = 0
    std_th = 1.2969576587040524e-05  # WRONG, just measured
    profile = LaguerreGaussianTransverseProfile(w0, p, m)
    r = np.linspace(0, 6 * w0, npoints)
    field = profile.evaluate(r, np.zeros_like(r))
    std = np.sqrt(np.average(r**2, weights=np.abs(field)))
    print("std_th = ", std_th)
    print("std = ", std)
    assert np.abs(std - std_th) / std_th < 0.01

    # SuperGaussianLaserProfile
    print("SuperGaussianLaserProfile")
    # close to flat-top, compared with flat-top theory
    n_order = 100
    std_th = w0 / np.sqrt(3)
    profile = SuperGaussianTransverseProfile(w0, n_order)
    r = np.linspace(0, 6 * w0, npoints)
    field = profile.evaluate(r, np.zeros_like(r))
    std = np.sqrt(np.average(r**2, weights=np.abs(field)))
    print("std_th = ", std_th)
    print("std = ", std)
    assert np.abs(std - std_th) / std_th < 0.01


def test_transverse_profiles_3d():
    npoints = 200
    w0 = 10.0e-6

    # HermiteGaussianTransverseProfile
    print("HermiteGaussianTransverseProfile")
    n_x = 2
    n_y = 2
    std_th = 1.2151311989441392e-05  # WRONG, just measured here
    profile = HermiteGaussianTransverseProfile(w0, n_x, n_y)
    x = np.linspace(-4 * w0, 4 * w0, npoints)
    y = np.zeros_like(x)
    field = profile.evaluate(x, y)
    std = np.sqrt(np.average(x**2, weights=np.abs(field)))
    print("std_th = ", std_th)
    print("std = ", std)
    assert np.abs(std - std_th) / std_th < 0.01


def test_profile_gaussian_3d_cartesian(gaussian):
    # - 3D Cartesian case
    dim = "xyt"
    lo = (-10e-6, -10e-6, -60e-15)
    hi = (+10e-6, +10e-6, +60e-15)
    npoints = (100, 100, 100)

    laser = Laser(dim, lo, hi, npoints, gaussian)
    laser.write_to_file("gaussianlaser3d")
    laser.propagate(1e-6)
    laser.write_to_file("gaussianlaser3d")


def test_profile_gaussian_cylindrical(gaussian):
    # - Cylindrical case
    dim = "rt"
    lo = (0e-6, -60e-15)
    hi = (10e-6, +60e-15)
    npoints = (50, 100)

    laser = Laser(dim, lo, hi, npoints, gaussian)
    laser.write_to_file("gaussianlaserRZ")
    laser.propagate(1e-6)
    laser.write_to_file("gaussianlaserRZ")
