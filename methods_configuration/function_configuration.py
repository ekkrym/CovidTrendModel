from autoregression import fit_forecasting_loglog, fit_forecasting_ar7_daily, fit_forecast_7ahead, fit_ar_7, fit_ar_log
from misc_methods import fit_spline_linear_extrapolation
from smoothing import simple_mirroring
from poly import poly_fit
from linearfit import linear_model_fit_predict
from poisson import poisson_fit
from benchmark import benchmark
from misc_methods import mean_const, linear

METHODS = {"poly_fit": poly_fit,
           "benchmark": benchmark,
           "poisson_fit": poisson_fit,
           "fit_forecasting_loglog": fit_forecasting_loglog,
           "fit_forecasting_ar7_daily": fit_forecasting_ar7_daily,
           "fit_spline_linear_extrapolation": fit_spline_linear_extrapolation,
           "fit_forecast_7ahead": fit_forecast_7ahead,
           "fit_ar_7": fit_ar_7,
           "fit_ar_log": fit_ar_log,
           "mean_const": mean_const, 
           "linear": linear,
           "linear_model_fit_predict": linear_model_fit_predict}
