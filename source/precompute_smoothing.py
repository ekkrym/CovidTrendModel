import numpy as np
import numpy.matlib as mp
import statsmodels.api as sm  
import statsmodels
from scipy import signal
import scipy as sp 
import os
import json 
from smoothing import piecewise_STL, simple_mirroring
import collections

def update_presmoothing_one_country(cumulative, dates, country, 
                                    smoothing_fun, smoothing_params = {}, 
                                    datasource = "",
                                    newly = True):
    
    """
    save the smooth trends in the file to the file 
    or update the existent file with the new trend after the new observation is reported
    
    cumulative: cumulative observations
    dates:      dates of observations
    country:    country or region
    smoothing_fun: smoothing function
    smoothing_params: arguments of smoothing function
    datasource:    name of the datasource
    newly: whether to update the file for all incremental histories starting from the beginning or only 
           for the dates which are not yet in the file
           
    Output:
    dictionary with the key=the date of the end of time series, item is the smoothed time series
    
    """
    path  = "smoothing_res" 
    dates_str = [str(x) for x in dates]
    if not newly:
        if not os.path.exists(path):
            os.makedirs(path)

        filename = path +"/"+ country + "_" + datasource + "_" + smoothing_fun.__name__ + ".txt" 

        if os.path.isfile(filename) and datasource != 'BAG': # temporarily, as BAG probably updates several times in one day, so do not read existing smooth file for BAG 
            with open(filename) as json_file:
                try:
                    dict_smoothing = json.load(json_file)
                except:
                    print("problems with reading, recalculation")
                    dict_smoothing = collections.OrderedDict()
        else:
            dict_smoothing = collections.OrderedDict()

        dates_saved = list(dict_smoothing.keys())

        dates_to_update = list(set(dates_str)-set(dates_saved))
    else:
        dict_smoothing = collections.OrderedDict()
        dates_to_update = list(set(dates_str)) 
    start = 13 
    if len(dates_to_update)>0: 
        
        indices_to_update = [i for i, e in enumerate(dates_str ) if e in dates_to_update] 
        for i in indices_to_update:
            if i>start:
                dict_smoothing[dates_str[i]] = list(smoothing_fun(cumulative[:i+1], **smoothing_params))
            else:
                dict_smoothing[dates_str[i]] = list(simple_mirroring(cumulative[:i+1], 3)) 
        if not newly:
            with open(filename, 'w') as outfile: 
                json.dump(dict_smoothing, outfile)
     
    return [dict_smoothing[i] for i in dates_str]