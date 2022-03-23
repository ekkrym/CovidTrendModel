import numpy as np
from smoothing import simple_mirroring

def benchmark(ts, H=7, smoothed_dat=[], smoothing_fun=simple_mirroring):
    """
    The reference method by ISG group
    the very initial dashboard version
    ts: cumulative cases
    H: forecasting horizon
    Output:
    cumulative cases with H forecasts as the last H elements
    """
 
    if len(smoothed_dat) == 0:
        smoothed_dat = smoothing_fun(ts)
    fc_list = [] 
    ts_ = list(smoothed_dat)
    for i in range(H):
        H7 = (ts_[-1] / ts_[-8]) ** (1 / 7.) - 1
        H7 = np.where(abs(H7) > 0, H7, 10 ** (-6))
        H14 = (ts_[-8] / ts_[-15]) ** (1 / 7.) - 1
        H14 = np.where(abs(H14) > 0, H14, 10 ** (-6))
        I = (H7 / H14) ** (1 / 7.)
        est = I * (ts_[-1] - ts_[-3]) / 2.
        fc_list.append(est)
        ts_.append(est + ts_[-1])
    return ts_
