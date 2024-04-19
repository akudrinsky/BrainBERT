import numpy as np
from scipy import signal

class OptimizedSTFTPreprocessor():
    def __init__(self, cfg):
        super(OptimizedSTFTPreprocessor, self).__init__()
        self.cfg = cfg

    def __call__(self, wav):
        # Compute STFT using configuration directly, assume batched input if possible
        Zxx = self.compute_batched_stft(wav)
        return Zxx

    def compute_batched_stft(self, wav):
        # Configuration parameters
        fs = 500  # Sampling frequency
        show_fs = self.cfg.freq_channel_cutoff
        nperseg = self.cfg.nperseg
        noverlap = self.cfg.noverlap
        normalizing = self.cfg.normalizing

        # STFT computation
        _, _, Zxx = signal.stft(wav, fs, nperseg=nperseg, noverlap=noverlap, return_onesided=True)
        
        Zxx = np.abs(Zxx[:, :show_fs])

        # Normalization (move to GPU if possible)
        if normalizing == "zscore":
            Zxx = (Zxx - Zxx.mean(axis=-1)[:, :, None]) / Zxx.std(axis=-1)[:, :, None]
        elif normalizing == "db":
            Zxx = np.log(Zxx + 1e-6)  # Adding a small epsilon to avoid log(0)
        # assert not np.any(np.isnan(Zxx)), np.isnan(Zxx).sum()

        if np.isnan(Zxx).any():
            Zxx = np.nan_to_num(Zxx, nan=0.0)

        return Zxx

