from torch.utils.data import DataLoader, ConcatDataset

from .radar_dataset import RadarDataset
from .transforms import get_mean_std, normalize_and_cache_dataset


def prep_dataset(config):
    snr_list = config.SNR
    cnr_list = config.CNR
    nu_list  = config.NU

    # how many distinct (snr, cnr, nu) combos
    C = len(snr_list) * len(cnr_list) * len(nu_list)

    train_datasets = []
    val_datasets   = []

    # for each nu, SNR, CNR triple
    for nu in nu_list:
        for snr in snr_list:
            for cnr in cnr_list:
                # compute per-combo sizes
                n_train_with_tgt  = config.dataset_size // C
                n_train_without_tg= config.dataset_size // (10 * C)
                n_val_with_tgt    = config.dataset_size // (10 * C)
                n_val_without_tg  = config.dataset_size // (100 * C)

                # create training splits
                train_w = RadarDataset(
                    num_samples    = n_train_with_tgt,
                    n_targets      = config.n_targets,
                    random_n_targets = config.rand_n_targets,
                    snr            = snr,
                    cnr            = cnr,
                    nu             = nu
                )
                train_wo = RadarDataset(
                    num_samples    = n_train_without_tg,
                    n_targets      = 0,
                    random_n_targets = False,
                    snr            = snr,
                    cnr            = cnr,
                    nu             = nu
                )
                train_datasets.append(ConcatDataset([train_w, train_wo]))

                # create validation splits
                val_w = RadarDataset(
                    num_samples    = n_val_with_tgt,
                    n_targets      = config.n_targets,
                    random_n_targets = config.rand_n_targets,
                    snr            = snr,
                    cnr            = cnr,
                    nu             = nu
                )
                val_wo = RadarDataset(
                    num_samples    = n_val_without_tg,
                    n_targets      = 0,
                    random_n_targets = False,
                    snr            = snr,
                    cnr            = cnr,
                    nu             = nu
                )
                val_datasets.append(ConcatDataset([val_w, val_wo]))

    # concat all combos
    full_train_dataset = ConcatDataset(train_datasets)
    full_val_dataset   = ConcatDataset(val_datasets)

    # loaders & normalization (unchanged)
    train_loader = DataLoader(full_train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(full_val_dataset,   batch_size=config.batch_size, shuffle=False)

    signal_mean,   signal_std,   IQ_mean,  IQ_std   = get_mean_std(train_loader, convert=False)
    rd_s_mean,     rd_s_std,     rd_IQ_mean, rd_IQ_std = get_mean_std(train_loader, convert=True)

    norm_train_dataset = normalize_and_cache_dataset(
        full_train_dataset,
        signal_mean, signal_std, IQ_mean, IQ_std,
        rd_s_mean,   rd_s_std,   rd_IQ_mean, rd_IQ_std
    )
    norm_val_dataset   = normalize_and_cache_dataset(
        full_val_dataset,
        signal_mean, signal_std, IQ_mean, IQ_std,
        rd_s_mean,   rd_s_std,   rd_IQ_mean, rd_IQ_std
    )

    norm_train_loader = DataLoader(norm_train_dataset, batch_size=config.batch_size, shuffle=True)
    norm_val_loader   = DataLoader(norm_val_dataset,   batch_size=config.batch_size, shuffle=False)

    return norm_train_loader, norm_val_loader, norm_train_dataset, norm_val_dataset
