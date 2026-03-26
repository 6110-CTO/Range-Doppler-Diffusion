import numpy as np
import torch

from src.data import RadarDataset
from src.data.transforms import create_rd_map_differentiable
from .cfar import ca_cfar_2d, tm_cfar_2d


def simulate_cfar_performance(cfar_func, specified_Pfa, nu_val, num_trials=100,
                              n_targets=3, random_n_targets=False, **cfar_kwargs):
    """
    For a given CFAR function, specified false-alarm parameter, and clutter nu,
    simulate num_trials frames and compute the average probability of detection (Pd)
    and measured probability of false alarm (Pfa_meas).
    """
    dataset = RadarDataset(num_samples=num_trials, n_targets=n_targets,
                           random_n_targets=random_n_targets, nu=nu_val, snr=10, cnr=15)
    total_true_detections = 0
    total_targets = 0
    total_false_alarms = 0
    total_non_target_cells = 0
    for i in range(num_trials):
        _, _, _, IQ_map, rd_label, _ = dataset[i]
        RD_map = create_rd_map_differentiable(IQ_map)
        RD_mag = torch.abs(RD_map).detach().numpy()
        detection_map = cfar_func(RD_mag, **cfar_kwargs, Pfa=specified_Pfa)
        gt = rd_label.detach().numpy()
        true_detections = np.sum((detection_map == 1) & (gt == 1))
        false_alarms = np.sum((detection_map == 1) & (gt == 0))
        total_targets += np.sum(gt)
        total_true_detections += true_detections
        total_false_alarms += false_alarms
        total_non_target_cells += (gt.size - np.sum(gt))
    pd_rate = total_true_detections / total_targets if total_targets > 0 else 0
    measured_pfa = total_false_alarms / total_non_target_cells if total_non_target_cells > 0 else 0
    return pd_rate, measured_pfa


def simulate_cfar_dif(dataset, cfar_func, specified_Pfa, nu_val, num_trials=100,
                              n_targets=3, random_n_targets=False, **cfar_kwargs):
    """
    For a given CFAR function, specified false-alarm parameter, and clutter nu,
    simulate num_trials frames and compute the average probability of detection (Pd)
    and measured probability of false alarm (Pfa_meas).
    """
    total_true_detections = 0
    total_targets = 0
    total_false_alarms = 0
    total_non_target_cells = 0
    for i in range(num_trials):
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs = dataset[i]
        RD_mag = torch.abs(RDs_norm).detach().numpy()
        detection_map = cfar_func(RD_mag, **cfar_kwargs, Pfa=specified_Pfa)
        gt = labels.detach().numpy()
        true_detections = np.sum((detection_map == 1) & (gt == 1))
        false_alarms = np.sum((detection_map == 1) & (gt == 0))
        total_targets += np.sum(gt)
        total_true_detections += true_detections
        total_false_alarms += false_alarms
        total_non_target_cells += (gt.size - np.sum(gt))
    pd_rate = total_true_detections / total_targets if total_targets > 0 else 0
    measured_pfa = total_false_alarms / total_non_target_cells if total_non_target_cells > 0 else 0
    return pd_rate, measured_pfa
