import math
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset


def generate_range_steering_matrix(N=64, dR=64, B=50e6, c=3e8):
    """
    Generates the range steering matrix R.
    """
    rng_res = c / (2 * B)
    r_vals = torch.arange(dR) * rng_res
    n_vals = torch.arange(N)
    phase = -1j * 2 * math.pi * (2 * B) / (c * N)
    R = torch.exp(phase * torch.outer(n_vals, r_vals))
    return R

def generate_doppler_steering_matrix(K=64, dV=64, fc=9.39e9, T0=1e-3, c=3e8):
    """
    Generates the Doppler steering matrix V.
    """
    vel_res = c / (2 * fc * K * T0)
    # Create a symmetric velocity vector
    v_vals = torch.linspace(-dV // 2, dV // 2, dV) * vel_res
    k_vals = torch.arange(K)
    phase = -1j * 2 * math.pi * (2 * fc * T0) / c
    V = torch.exp(phase * torch.outer(k_vals, v_vals))
    return V

def create_rd_map_differentiable(IQ_map):
    """
    Converts a complex IQ (or signal) map to a range-Doppler (RD) map.
    If the input is not already complex, it is cast to torch.complex64.
    Returns the absolute value of the radar return in the RD domain.
    """
    if not torch.is_tensor(IQ_map):
        IQ_map = torch.from_numpy(IQ_map)
    if not torch.is_complex(IQ_map):
        IQ_map = IQ_map.to(torch.complex64)
    device = IQ_map.device
    R = generate_range_steering_matrix().to(device)
    V = generate_doppler_steering_matrix().to(device)
    # The steering is applied to convert the input domain to the RD domain.
    RD_map = R.T.conj() @ IQ_map @ V.conj()
    return RD_map

def get_mean_std(radarloader, convert=False):
    IQ_total_sum = 0.0
    IQ_total_sq_sum = 0.0
    IQ_total_samples = 0
    signal_total_sum = 0.0
    signal_total_sq_sum = 0.0
    signal_total_samples = 0
    for signal, _, _, IQ, _, _ in radarloader:
        if convert:
            signal = create_rd_map_differentiable(signal)
            IQ = create_rd_map_differentiable(IQ)
        IQ_total_sum += IQ.real.sum() + IQ.imag.sum()
        IQ_total_sq_sum += (IQ.real.pow(2).sum() + IQ.imag.pow(2).sum())
        IQ_total_samples += IQ.numel() * 2 # multiply by 2 for real and imaginary
        signal_total_sum += signal.real.sum() + signal.imag.sum()
        signal_total_sq_sum += (signal.real.pow(2).sum() + signal.imag.pow(2).sum())
        signal_total_samples += signal.numel() * 2 # multiply by 2 for real and imaginary
    IQ_mean = IQ_total_sum / IQ_total_samples
    IQ_std = torch.sqrt((IQ_total_sq_sum / IQ_total_samples) - IQ_mean**2)
    signal_mean = signal_total_sum / signal_total_samples
    signal_std = torch.sqrt((signal_total_sq_sum / signal_total_samples) - signal_mean**2)
    return signal_mean, signal_std, IQ_mean, IQ_std

def normalize_and_cache_dataset(dataset, iq_signal_mean, iq_signal_std, iq_IQ_mean, iq_IQ_std, rd_signal_mean, rd_signal_std, rd_IQ_mean, rd_IQ_std):
    signals_norm = []
    rd_signals_norm = []
    IQs_norm = []
    RDs_norm = []
    labels = []
    scnr_dBs = []
    clutter_all = []
    gauss_all = []

    for idx in tqdm(range(len(dataset)), desc='Normalizing dataset'):
        signal, clutter, gaus_noise, IQ, rd_label, scnr_dB = dataset[idx]
        rd_signal = create_rd_map_differentiable(signal)
        rd_IQ = create_rd_map_differentiable(IQ)
        # Normalize signal
        signal_real_norm = (signal.real - iq_signal_mean) / iq_signal_std
        signal_imag_norm = (signal.imag - iq_signal_mean) / iq_signal_std
        signal_norm = torch.complex(signal_real_norm, signal_imag_norm)

        # Normalize IQ
        IQ_real_norm = (IQ.real - iq_IQ_mean) / iq_IQ_std
        IQ_imag_norm = (IQ.imag - iq_IQ_mean) / iq_IQ_std
        IQ_norm = torch.complex(IQ_real_norm, IQ_imag_norm)

        # Normalize rd signal
        rd_signal_real_norm = (rd_signal.real - rd_signal_mean) / rd_signal_std
        rd_signal_imag_norm = (rd_signal.imag - rd_signal_mean) / rd_signal_std
        rd_signal_norm = torch.complex(rd_signal_real_norm, rd_signal_imag_norm)

        # Normalize rd IQ
        RD_real_norm = (rd_IQ.real - rd_IQ_mean) / rd_IQ_std
        RD_imag_norm = (rd_IQ.imag - rd_IQ_mean) / rd_IQ_std
        RD_norm = torch.complex(RD_real_norm, RD_imag_norm)

        signals_norm.append(signal_norm)
        IQs_norm.append(IQ_norm)
        rd_signals_norm.append(rd_signal_norm)
        RDs_norm.append(RD_norm)
        labels.append(rd_label)
        scnr_dBs.append(scnr_dB)

        # Save clutter and gauss tensors as well
        clutter_all.append(clutter)
        gauss_all.append(gaus_noise)

    # Stack everything into tensors
    signals_norm = torch.stack(signals_norm)
    IQs_norm = torch.stack(IQs_norm)
    rd_signals_norm = torch.stack(rd_signals_norm)
    RDs_norm = torch.stack(RDs_norm)
    labels = torch.stack(labels)
    scnr_dBs = torch.tensor(scnr_dBs)
    clutter_all = torch.stack(clutter_all)
    gauss_all = torch.stack(gauss_all)

    # Return cached TensorDataset (now consistent)
    return TensorDataset(signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs)
