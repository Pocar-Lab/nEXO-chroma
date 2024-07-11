#!/usr/bin/env python

from .geometry_manager import geometry_manager
from .run_manager import run_manager
import numpy as np


def test_Sebastian_FS_06082023_correctedSiPM():
    experiment_name = "Sebastian_FS_06.08.2023_correctedSiPM"
    run_id = -1
    visualize = False
    seed = 1337
    num_particles = 100_000
    plots = []

    gm = geometry_manager(
        experiment_name=experiment_name, run_id=run_id, visualize=visualize
    )
    rm = run_manager(
        geometry_manager=gm,
        experiment_name=experiment_name,
        random_seed=seed,
        num_particles=num_particles,
        run_id=run_id,
        plots=plots,
    )
    am = rm.ana_man  # analysis_manager
    hist = am.plot_angle_hist_wrapper(showPlot=True)
    expected = [
        8,
        20,
        26,
        33,
        51,
        74,
        83,
        74,
        91,
        45,
        29,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    assert np.all(np.asarray(hist) == np.asarray(expected))


def test_Sebastian_08012023liquefaction_correctedSiPM():
    experiment_name = "Sebastian_08.01.2023(liquefaction)_correctedSiPM"
    run_id = -1
    visualize = False
    seed = 1337
    num_particles = 100_000
    plots = []

    gm = geometry_manager(
        experiment_name=experiment_name, run_id=run_id, visualize=visualize
    )
    rm = run_manager(
        geometry_manager=gm,
        experiment_name=experiment_name,
        random_seed=seed,
        num_particles=num_particles,
        run_id=run_id,
        plots=plots,
    )
    am = rm.ana_man  # analysis_manager
    hist = am.plot_angle_hist_wrapper(showPlot=True)
    expected = [
        7,
        21,
        29,
        36,
        34,
        18,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        10,
        19,
        15,
        14,
        9,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    assert np.all(np.asarray(hist) == np.asarray(expected))
