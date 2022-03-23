import numpy as np
import scipy
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from smoothing import simple_mirroring 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import Pipeline

import glmnet_python
from glmnet import glmnet;
from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint;
from glmnetCoef import glmnetCoef;
from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet;
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot;
from cvglmnetPredict import cvglmnetPredict


def poisson(X_train, Y_train, X_test, Y_test, alpha=0.1):
    """
    Learn a linear regression model with glmnet
    """
    fit = glmnet(x=scipy.array(np.float64(X_train)), y=scipy.array(np.float64(Y_train)), family='poisson', alpha=alpha) 
    y_pred = glmnetPredict(fit, np.array(X_test).reshape(-1, 1), ptype='response', s=scipy.float64([0.01])) 
    y_pred_horizon = y_pred[-1]

    return y_pred, y_pred_horizon


def poisson_fit(allcases_ts, smoothing_fun=simple_mirroring, smoothed_dat=[],
                H=7, W=7, daily=False, log_=False, order0=True, ns=3, alpha=0.):
    """
        Polynomial prediction for 1,...,7 steps ahead with corresponding forecasting with options:
        daily: flag that the model is built whether on cumulative or daily cases
        log_ : taking log or not
        order0 : differencing or not (default True, i.e. not)
        ns : parameter for smoothing
    """
    if len(smoothed_dat) == 0:
        smoothed_dat = smoothing_fun(allcases_ts, Ws=ns)
    smoothed_dat = smoothing_fun(allcases_ts, Ws=ns)
    if daily:
        dat = np.diff(smoothed_dat)
    else:
        dat = smoothed_dat
    if log_:
        dat = np.log(dat + 1)
    if order0 == False:
        dat0 = dat.copy()[-1]
        dat = np.diff(dat)

    n = len(dat)
    X = np.arange(n - W, n).reshape(-1, 1)
    y = np.array(dat[n - W:n]).reshape(-1, 1)
    x = np.arange(n, n + H).reshape(-1, 1)

    y_out, y_last = poisson(X, y, x, [], alpha)

    if order0 == False:
        y_out = list([dat0]) + y_out
        y_out = np.cumsum(y_out)[-H:]
    if log_:
        y_out = np.exp(y_out)
    if daily:
        y_out = np.cumsum(np.insert(y_out, 0, smoothed_dat[-1]))
    else:
        y_out = np.insert(y_out, 0, smoothed_dat[-1])
    return y_out


def poisson_fit_ar(allcases_ts, smoothing_fun=simple_mirroring, smoothed_dat=[],
                   p=7, H=7, W=7, daily=False, log_=False, order0=True, ns=3, beta=0.):
    """
        Simple autoregression with forecasting one step ahead with options:
        daily: flag that the model is built whether on cumulative or daily cases
        log_ : taking log or not
        order0 : differencing or not (default True, i.e. not)
        ns : parameter for smoothing
    """
    if len(smoothed_dat) == 0:
        smoothed_dat = smoothing_fun(allcases_ts, Ws=ns)
    if daily:
        dat = np.diff(smoothed_dat)
    else:
        dat = smoothed_dat

    if log_:
        dat = np.log(dat + 1)
    if order0 == False:
        dat0 = dat.copy()[-1]
        dat = np.diff(dat)

    y, X, y_out = [], [], []
    for i in range(len(dat) - p - W, len(dat) - p):
        y.append(dat[i + p])
        X.append(dat[i:i + p])
    X, y = np.array(X).reshape(-1, 1), np.array(y).reshape(-1, 1) 
    x = np.array(dat[-p:]).reshape(-1, 1)

    y_out, y_last = poisson(X, y, x, [], beta) 

    if order0 == False:
        y_out = list([dat0]) + y_out
        y_out = np.cumsum(y_out)[-H:]
    if log_:
        y_out = np.exp(y_out)
    if daily:
        y_out = np.cumsum(np.insert(y_out, 0, smoothed_dat[-1]))
    else:
        y_out = np.insert(y_out, 0, smoothed_dat[-1])
    return y_out
