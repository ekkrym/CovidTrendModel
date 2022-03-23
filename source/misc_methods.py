import csaps
import numpy as np

from smoothing import simple_mirroring, piecewise_STL


def fit_spline_linear_extrapolation(cumul_observations, smoothing_fun=simple_mirroring, smoothed_dat=[],
                                    plotf=False, smoothep=True, smooth=0.5, ns=3, H=7):
    """
    Linear extrapolation by splines on log daily cases
 
    Input: 
    cumul_observations: cumulative observations,
    smoothed_dat:       list of trends of incremental history,
    ns:                 optional smoothing window parameter,
    H:                  forecasting horison
    smooth:             whether to compute mean from trend or from raw data
    
    Output: forecast on horison H in terms of cumulative numbers 
            starting from the last observation
    """
    if len(smoothed_dat) == 0:
        smoothed_dat = smoothing_fun(cumul_observations, Ws=ns)
    val_start = smoothed_dat[-1]
    dat = np.log(list(smoothed_dat + 1))
    spl = csaps.UnivariateCubicSmoothingSpline(range(len(dat)), dat, smooth=smooth)
    dat_diff = np.diff(spl(np.arange(len(dat))))
    x = np.arange(len(dat_diff))
    spl = csaps.UnivariateCubicSmoothingSpline(x, dat_diff, smooth=smooth)
    dat_diff_sm = spl(x)
    step = dat_diff_sm[-1] - dat_diff_sm[-2]
    if smoothep:
        dat_forecast = dat_diff_sm[-1] + step * np.arange(1, H + 1)  # + seasonality
    else:
        dat_forecast = dat_diff[-1] + step * np.arange(1, H + 1)  # + seasonality
    forecast = np.insert(np.exp(np.cumsum(dat_forecast)) * val_start, 0, val_start)

    return forecast

def mean_const(cumul_observations, smoothed_dat,
                                      ns=3, H=7, smooth=True):
    """
    constant forecasting with the mean value from the last H obsrtvations

    Input: 
    cumul_observations: cumulative observations,
    smoothed_dat:       list of trends of incremental history,
    ns:                 optional smoothing window parameter,
    H:                  forecasting horison
    smooth:             whether to compute mean from trend or from raw data
    
    Output: forecast on horison H in terms of cumulative numbers 
            starting from the last observation
    """
    smoothed_dat = np.array(smoothed_dat)
    if smooth:
        observations = np.diff(smoothed_dat)
        val_start = smoothed_dat[-1]
    else:
        observations = np.diff(cumul_observations)
        val_start = cumul_observations[-1]
    forecast = [np.mean(observations[-H:])]*H
    
    return np.insert(np.cumsum(forecast),0,0)+val_start


def linear(cumul_observations, smoothed_dat,
                                      W=3, H=7, smooth=True):
    """
    linear forecasting in original scale if the trend is increasing
    and in log-scale if trend is decreasing

    Input: 
    cumul_observations: cumulative observations,
    smoothed_dat:       list of trends of incremental history,
    W:                  number of points to compute the coefficients,
    H:                  forecasting horison
    smooth:             whether to compute mean from trend or from raw data
    
    Output: forecast on horison H in terms of cumulative numbers 
            starting from the last observation
    """
    smoothed_dat = np.array(smoothed_dat)
    if smooth: 
        observations = np.diff(smoothed_dat)
        val_start = smoothed_dat[-1]
    else: 
        observations = np.diff(cumul_observations)
        val_start = cumul_observations[-1]
    
    W=7
    solution, _, _, _ = np.linalg.lstsq(np.concatenate([np.ones((W,1)), np.arange(1,W+1)[:,np.newaxis]], axis = 1), observations[-W:])
    
    if solution[1]>0:
        forecast = np.dot(np.concatenate([np.ones((H,1)), np.arange(W+1,W+H+1)[:,np.newaxis]], axis = 1), solution)
    else:
        solution, _, _, _ = np.linalg.lstsq(np.concatenate([np.ones((W,1)), np.arange(1,W+1)[:,np.newaxis]], axis = 1), np.log(observations[-W:]+1))       
        forecast = np.dot(np.concatenate([np.ones((H,1)), np.arange(W+1,W+H+1)[:,np.newaxis]], axis = 1), solution)
        forecast = np.exp(forecast)-1
    # treating the case without taking log and possible negative values
    forecast = np.where(forecast > 0, forecast, 0) 
    
    return np.insert(np.cumsum(forecast),0,0)+val_start
    