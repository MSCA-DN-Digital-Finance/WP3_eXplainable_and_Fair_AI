# file holding generic utility functions used across codebase

import numpy as np

# function to calculate  wasserstein distance from 1D grid
def wasserstein_1d_on_grid(f, P, Q):
    f = np.asarray(f, float)
    P = np.clip(np.asarray(P, float), 0, None)
    Q = np.clip(np.asarray(Q, float), 0, None)

    P = P / (P.sum() + 1e-12)
    Q = Q / (Q.sum() + 1e-12)

    cdf_P = np.cumsum(P)
    cdf_Q = np.cumsum(Q)

    df = np.diff(f, prepend=f[0])
    return float(np.sum(np.abs(cdf_P - cdf_Q) * df))




# plot power spectrum for summer and winter
def power_spectrum_1s(x, dt=1.0, detrend=True, window=True, normalize=True, nfft=None):
    x = np.asarray(x, float)

    if detrend:
        x = x - np.mean(x)

    if window:
        w = np.hanning(len(x))
        x = x * w

    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(len(x))))

    X = np.fft.rfft(x, n=nfft)
    P = np.abs(X) ** 2
    f = np.fft.rfftfreq(nfft, d=dt)

    if normalize:
        s = P.sum()
        if s > 0:
            P = P / s

    return f, P




# function to calculate mean average error per inference and return list of maes

def calculate_mae(file_path, season='summer'):
    
    data = np.load(file_path)
    traj = np.load('data/solar_summer_winter.npz')[season]
    if 'chronos' in file_path:
        preds = data['yhat_q']
    else:
        preds = data['yhat']
    window_starts = data['window_start']
    window_ends = data['window_end']

    maes = []

    for i in range(len(window_starts)):

        if 'chronos' in file_path:
            pred = preds[i, :, :]
        else:
            pred = preds[i, :]
        target = traj[window_ends[i]:window_ends[i]+pred.shape[0]]
        try:
            mae = np.mean(np.abs(pred - target))
            maes.append(mae)
        except ValueError:
            print(f"Error calculating MAE for iteration {i}")
            continue


    return maes
        