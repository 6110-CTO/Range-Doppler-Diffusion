import numpy as np
import matplotlib.pyplot as plt


def plot_pd_pfa(results: dict, save_path: str = 'pd_pfa.png'):
    plt.figure(figsize=(12, 6))
    plt.title('ROC Curves for Different Clutter Conditions', fontsize=16, fontweight='bold')

    for nu, (pd, pfa) in results.items():
        order      = np.argsort(pfa)
        pfa_sorted = pfa[order]
        pd_sorted  = pd[order]
        plt.semilogx(pfa_sorted, pd_sorted,
                     marker='o', linestyle='-',
                     linewidth=2, label=f'\u03bd = {nu}')

    plt.xlabel('Probability of False Alarm')
    plt.ylabel('Probability of Detection')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.xlim(1e-6, 1e-1)
    plt.xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_pd_scnr(results: dict, save_path: str = 'pd_scnr.png'):
    """
    results : dict
        maps nu -> (pd_array, pfa_array, scnr_array)
    """
    plt.figure(figsize=(12, 6))
    plt.title('PD vs SCNR for Fixed PFA', fontsize=16, fontweight='bold')

    for nu, (pd, pfa, scnr) in results.items():
        # sort by SCNR
        order     = np.argsort(scnr)
        scnr_sorted = scnr[order]
        pd_sorted   = pd[order]

        plt.plot(scnr_sorted,
                 pd_sorted,
                 marker='o',
                 linestyle='-',
                 linewidth=2,
                 label=f'\u03bd = {nu}')

    plt.xlabel('SCNR (dB)')
    plt.ylabel('Probability of Detection')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
