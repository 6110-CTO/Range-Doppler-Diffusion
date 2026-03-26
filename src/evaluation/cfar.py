import numpy as np


def ca_cfar_2d(signal, num_train, num_guard, Pfa):
    """
    Standard CA-CFAR on a 2D signal.
    """
    rows, cols = signal.shape
    detection_map = np.zeros_like(signal)

    win_size = 2 * (num_train + num_guard) + 1
    guard_size = 2 * num_guard + 1
    num_training_cells = win_size**2 - guard_size**2

    # Scaling factor for exponential noise
    alpha = num_training_cells * (Pfa**(-1/num_training_cells) - 1)

    pad = num_train + num_guard
    padded_signal = np.pad(signal, pad, mode='constant', constant_values=0)

    for i in range(pad, pad + rows):
        for j in range(pad, pad + cols):
            window = padded_signal[i - pad:i + pad + 1, j - pad:j + pad + 1]
            start = num_train
            end = num_train + 2 * num_guard + 1
            training_cells = np.concatenate((window[:start, :].ravel(),
                                             window[end:, :].ravel(),
                                             window[start:end, :start].ravel(),
                                             window[start:end, end:].ravel()))
            noise_level = np.mean(training_cells)
            threshold = alpha * noise_level
            if signal[i - pad, j - pad] > threshold:
                detection_map[i - pad, j - pad] = 1
    return detection_map

def tm_cfar_2d(signal, num_train, num_guard, trim_ratio, Pfa):
    """
    TM-CFAR on a 2D signal.
    """
    rows, cols = signal.shape
    detection_map = np.zeros_like(signal)

    win_size = 2 * (num_train + num_guard) + 1
    guard_size = 2 * num_guard + 1
    num_training_cells = win_size**2 - guard_size**2

    # Number of cells to trim from each end
    trim_cells = int(trim_ratio * num_training_cells)
    effective_cells = num_training_cells - 2 * trim_cells
    if effective_cells <= 0:
        effective_cells = num_training_cells  # fallback
    alpha = effective_cells * (Pfa**(-1/effective_cells) - 1)

    pad = num_train + num_guard
    padded_signal = np.pad(signal, pad, mode='constant', constant_values=0)

    for i in range(pad, pad + rows):
        for j in range(pad, pad + cols):
            window = padded_signal[i - pad:i + pad + 1, j - pad:j + pad + 1]
            start = num_train
            end = num_train + 2 * num_guard + 1
            training_cells = np.concatenate((window[:start, :].ravel(),
                                             window[end:, :].ravel(),
                                             window[start:end, :start].ravel(),
                                             window[start:end, end:].ravel()))
            sorted_cells = np.sort(training_cells)
            if 2 * trim_cells < num_training_cells:
                trimmed = sorted_cells[trim_cells: num_training_cells - trim_cells]
            else:
                trimmed = sorted_cells
            noise_level = np.mean(trimmed)
            threshold = alpha * noise_level
            if signal[i - pad, j - pad] > threshold:
                detection_map[i - pad, j - pad] = 1
    return detection_map
