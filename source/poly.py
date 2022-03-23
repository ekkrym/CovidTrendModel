import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from smoothing import simple_mirroring
# importing libraries for polynomial transform
from sklearn.preprocessing import PolynomialFeatures
# for creating pipeline
from sklearn.pipeline import Pipeline


 


def fit_polynomials_ridge(X_train, Y_train, X_test, Y_test, HORIZON, alpha, log=False, p=2): 
    # creating pipeline and fitting it on data
    Input = [('polynomial', PolynomialFeatures(degree=p)), ('modal', Ridge(alpha=alpha, normalize=True))]
    pipe = Pipeline(Input)

    pipe.fit(X_train.reshape(-1, 1), Y_train.reshape(-1, 1))

    y_pred = pipe.predict(X_test.reshape(-1, 1))

    y_pred_horizon = y_pred[-1]

    return y_pred, y_pred_horizon


def poly_fit(allcases_ts, smoothing_fun=simple_mirroring, smoothed_dat=[],
             H=7, W=7, daily=False, log_=False, order0=True, ns=3, beta=0., p=2):
    """
        Polynomial prediction for 1,...,7 steps ahead with corresponding forecasting with options:
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

    n = len(dat)
    X = np.arange(n - W, n)
    y = np.array(dat[n - W:n])
    x = np.arange(n, n + H)

    y_out, y_last = fit_polynomials_ridge(X, y, x, [], H, beta, log_, p)
    
    if order0 == False:
        y_out = list([dat0]) + list(y_out)
        y_out = np.cumsum(y_out)[-H:]
    if log_:
        y_out = np.exp(np.array(y_out)) - 1 
        y_out = np.where(y_out>0, y_out, 0)
    if daily:
        return np.cumsum(np.insert(y_out, 0, smoothed_dat[-1]))
    else:
        return np.insert(y_out, 0, smoothed_dat[-1]) 