"""
data reading and proprocessing

repair_increasing:      negative values preprocessing
precompute_outofsample: create time series of incremental length from initial time series
precompute_smoothing:   compute trends of incremental time series starting from a certain time point
read_countrydata:       read cumulative numbers for the country/region
read_countries:         read cumulative numbers for countries/regions
read_countrydata_jhu:   read cumulative numbers for country/region from JHU
read_countries_jhu:     read cumulative numbers for countries/regions from JHU
same for the other data sources: read_countrydata_ozh, read_countrydata_bag,
read_countrydata_bag_kt, read_countrydata_cam, read_countrydata_hub, read_countrydata_spf_dep,
read_countrydata_spf_fra

compute_method_depth:  computes the history needed for a particular extrapolation method 
minimal_method_depth: computes the minimal history needed in the case of several methods compared


precompute_forecasts: used in CI, compute forecasts based on estimated retrospective trends
"""

import numpy as np
import pandas as pd
from smoothing import piecewise_STL
from matplotlib import pyplot as plt
from test_missing_or_zero import test_poisson 

def repair_increasing(allcases_):
    """
    fix for the negative value occurring in the daily: x_{t-7} *(X_{t-1}-X_{t-8})/(X_{t-8}-X_{t-15})
    then scaling to get the last cumulative value after negative value appears
    """
    allcases = np.copy(allcases_)
    ind = np.diff(list(allcases))
    loc_negative = np.where(ind<0)[0] 
 
    for j in range(len(loc_negative)):
        i_neg = loc_negative[j] + 1 
        if i_neg < 15:
            for i in range(i_neg-1, 0, -1):
                if allcases[i+1] < allcases[i]:
                    allcases[i] = allcases[i+1]
        else: 
            cumul_toscale = allcases[i_neg].copy()
            coeff = (allcases[i_neg-1] - allcases[i_neg-8])/((allcases[i_neg-8] - allcases[i_neg-15])+1)
            daily = (allcases[i_neg-7] - allcases[i_neg-8]) *coeff
            allcases[i_neg] =  allcases[i_neg-1] + daily
            allcases[:i_neg+1] = allcases[:i_neg+1] * cumul_toscale / allcases[i_neg]
 
    return allcases 


def precompute_outofsample(allcases):
    """
    create time series of incremental length from initial time series
    """
    incremental_history = []
    for i in range(1, len(allcases) + 1):
        incremental_history.append(allcases[0:i])
    return incremental_history


def precompute_smoothing(allcases, smoothing_fun, Ws=7, start_smoothing=0, inc_history=[]):
    """
    compute trends of incremental time series starting from a certain time point
    """
    if len(inc_history) == 0:
        inc_history = precompute_outofsample(allcases)
    smoothed_history = []
    
    for i in range(start_smoothing):
        smoothed_history.append(inc_history[i])
        
    for i in range(start_smoothing, len(inc_history)):
        smoothed_history.append(smoothing_fun(inc_history[i], Ws))
    return smoothed_history


def read_countrydata(df, country, number_startcases=30, datasource = "ECDC", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day,
    when the number of cases is higher than number-startcases
    typedata ' cases or deaths'f
    
    """
   
    df_country = df[(df["countriesAndTerritories"] == country)].copy() 
    df_country["dateRep"] = pd.DatetimeIndex(pd.to_datetime(df_country["dateRep"], format="%d/%m/%Y"))
 
    
    if False: #the latest fix, the forecasts start from the last observed value
        extend = ((np.max(df_country["dateRep"]) - pd.Timestamp.now().normalize()) / np.timedelta64(1, 'D')) 
        if extend < -7.:
            print(country + " was not updating the data for more than 7 days")
            return [], [] 
    
    df_country = df_country.sort_values(by="dateRep").reset_index(drop =True) 
    df_country = df_country.set_index("dateRep")  
    
    if False: #the latest fix, the forecasts start from the last observed value
        if extend < 0:
            df_country.loc[df_country.index.max() + pd.Timedelta(days=-extend)] = None
     
    df_country = df_country.resample('D').asfreq().fillna(0)#.pad(mode = "empty")
    
    #df_country =df_country[:-1] drop today  
    allcases_ = np.cumsum(df_country[typedata])

    if np.max(allcases_) < number_startcases:
        print("The country has a number of cases  less than the threshold ", number_startcases)
        return [], []
     
    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)
    
    date_ =  pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 

    return allcases, date_


def read_countries(df, countries, number_startcases=30, datasource = "ECDC", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day,
    when the number of cases is higher than number-startcases
    """
    allcases_list, date_list = [], []
    if datasource == "ECDC":
        read_countrydatafun = read_countrydata
    elif "JHU" in datasource:
        read_countrydatafun = read_countrydata_jhu
    elif datasource == 'OZH':
        read_countrydatafun = read_countrydata_ozh
    elif datasource == 'BAG':
        read_countrydatafun = read_countrydata_bag
    elif datasource == 'BAG_KT':
        read_countrydatafun = read_countrydata_bag_kt
    elif datasource == 'CAN':
        read_countrydatafun = read_countrydata_can
    elif datasource == 'HUB':
        read_countrydatafun = read_countrydata_hub
    elif datasource == 'PHE':
        read_countrydatafun = read_countrydata_phe
    elif datasource == 'SPF_DEP':
        read_countrydatafun = read_countrydata_spf_dep
    elif datasource == 'SPF_REG':
        read_countrydatafun = read_countrydata_spf_reg
    elif datasource == 'SPF_FRA':
        read_countrydatafun = read_countrydata_spf_fra
    else:
        raise ValueError(f'Unsupported data source: {datasource}')
       
    for country in countries:
        allcases, date_ = read_countrydatafun(df, country, typedata = typedata, 
                                           number_startcases = number_startcases, datasource = datasource)
        allcases_list.append(allcases)
        date_list.append(date_)
    return allcases_list, date_list


def read_countrydata_jhu(df, country, number_startcases=30, datasource = "JHU", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day, 
    when the number of cases is higher than number-startcases
    """ 
 
    if  datasource == "JHU":
        allcases_ = df[(df["Country/Region"] == country)].drop(["Province/State"],
                                                           axis=1).groupby(["Country/Region"]).sum().to_numpy() 
        allcases_ = np.array(allcases_)[0, 2:]
        stp = 4
    else: 
        allcases_ = df[(df["Province_State"] == country)].groupby(["Province_State"]).sum().to_numpy()   
 
        allcases_ = np.array(allcases_)[0, 6:]       
        stp = 12    
 
    
    #allcases_ = repair_increasing(allcases_)
    
    if np.max(allcases_) >= number_startcases:
        starting_point = np.min(np.where(allcases_ >= number_startcases)[0])
        allcases = np.array(allcases_[starting_point:], dtype=float)
      
        
        date_ = pd.to_datetime(df.columns[stp:])
        date_ = date_[starting_point:]
    else:
        print("The country/region has a number of cases  less than the threshold ", number_startcases)
        return [], [] 
    return allcases, date_


def read_countries_jhu(df, countries, number_startcases=30):
    """
    read the one dimensional cumulative casts data and the date starting from the day,
    when the number of cases is higher than number-startcases
    """
    allcases_list, date_list = [], []
    for country in countries:
        allcases_ = df[(df["Country/Region"] == country)].drop(["Province/State"],
                                                               axis=1).groupby(["Country/Region"]).sum().to_numpy()
        allcases_ = np.array(allcases_)[0, 4:]
        if np.max(allcases_) >= number_startcases:
            starting_point = np.min(np.where(allcases_ >= number_startcases)[0])
            allcases = allcases_[starting_point:]
         #   allcases = repair_increasing(allcases)
            date_ = pd.to_datetime(df.columns[5:])
            date_ = date_[starting_point:]
        else:
            print(country + " has a number of cases  less than the threshold ", number_startcases)
            allcases = []
            date_ = []
        allcases_list.append(allcases)
        date_list.append(date_)
    return allcases_list, date_list


def read_countrydata_ozh(df, country, number_startcases=30, datasource = "OZH", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day, 
    when the number of cases is higher than number-startcases
    """ 
 
    df_country = df[(df["abbreviation_canton_and_fl"] == country)].copy() 
    df_country["date"] = pd.DatetimeIndex(pd.to_datetime(df_country["date"], format="%Y-%m-%d"))
    
    df_country = df_country.sort_values(by="date").reset_index(drop =True) 
    df_country = df_country.set_index("date") 
    
    df_country = df_country[['ncumul_conf', 'ncumul_deceased']]
    
    # drop dates if both cases/deaths are NAs (mostly in the end of timeseries)
    df_country = df_country.dropna(how='all')
    
    df_country = df_country.resample('D').asfreq()
    
    # # linear interpolate in the log scale to fill the NAs in the middle
    # df_country = np.expm1(np.log1p(df_country).interpolate(limit_direction='forward')).round()
    
    # propagate the last non-NA value until the next non-NA observation
    df_country = df_country.ffill()
    
    # fill 0 for the NAs in the beginning of the series
    df_country = df_country.fillna(0)
    
    if (typedata == 'cases'):
        allcases_ = df_country['ncumul_conf']
    elif (typedata == 'deaths'):
        allcases_ = df_country['ncumul_deceased']
    else:
        raise Exception(f"typedata='{typedata}' is not supported.")

    if np.max(allcases_) < number_startcases:
        print("The country has a number of cases  less than the threshold ", number_startcases)
        return [], []
     
    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)
    
    date_ =  pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 

    return allcases, date_

def read_countrydata_bag(df, country, number_startcases=30, datasource = "BAG", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day, 
    when the number of cases is higher than number-startcases
    """ 
 
    df_country = df[(df["country"] == country)].copy() 
    df_country["Date"] = pd.DatetimeIndex(pd.to_datetime(df_country["Date"], format="%d.%m.%y"))
    
    df_country = df_country.sort_values(by="Date").reset_index(drop =True) 
    df_country = df_country.set_index("Date") 
    
    df_country = df_country[['Nombre de cas, cumulé', 'Nombre de décés, cumulé']]
    
    df_country = df_country.resample('D').asfreq().fillna(0)
    
    if (typedata == 'cases'):
        allcases_ = df_country['Nombre de cas, cumulé']
    elif (typedata == 'deaths'):
        allcases_ = df_country['Nombre de décés, cumulé']
    else:
        raise Exception(f"typedata='{typedata}' is not supported.")

    if np.max(allcases_) < number_startcases:
        print("The country has a number of cases  less than the threshold ", number_startcases)
        return [], []
     
    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)
    
    date_ =  pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 

    return allcases, date_


def read_countrydata_bag_kt(df, country, number_startcases=30, datasource = "BAG_KT", typedata = "cases"): 
    """
    read the one dimensional cumulative casts data and the date starting from the day, 
    when the number of cases is higher than number-startcases
    """ 

    df_country = df[(df["geoRegion"] == country)].copy()
    
    date_col, data_col = 'datum', 'sumTotal'
    
    df_country = df_country[[date_col, data_col]]
    df_country = df_country[~df_country[date_col].isna()]
    df_country[date_col] = pd.DatetimeIndex(pd.to_datetime(df_country[date_col], format="%Y-%m-%d"))
    df_country = df_country.sort_values(by=date_col).reset_index(drop=True)
    df_country = df_country.set_index(date_col)
    
    allcases_ = df_country[data_col].values

    if np.max(allcases_) < number_startcases:
        print("The country has a number of cases  less than the threshold ", number_startcases)
        return [], []
     
    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)
    
    date_ = pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 

    return allcases, date_


def read_countrydata_can(df, country, number_startcases=30, datasource = "CAN", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day,
    when the number of cases is higher than number-startcases
    typedata ' cases or deaths'f
    
    """

    df_country = df[(df['province'] == country)].copy()
    date_column = 'date_report' if typedata == 'cases' else 'date_death_report'
    df_country[date_column] = pd.DatetimeIndex(pd.to_datetime(df_country[date_column], format='%d-%m-%Y'))
    df_country = df_country.sort_values(by=date_column).reset_index(drop =True) 
    df_country = df_country.set_index(date_column)
    
    df_country = df_country.resample('D').asfreq().fillna(0)
    
    #df_country =df_country[:-1] drop today  
    
    allcases_ = df_country[f'cumulative_{typedata}'].values
    
    if np.max(allcases_) < number_startcases:
        print("The country has a number of cases  less than the threshold ", number_startcases)
        return [], []

    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)

    date_ =  pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 

    return allcases, date_

def read_countrydata_hub(df, country, number_startcases=30, datasource = "HUB", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day,
    when the number of cases is higher than number-startcases
    typedata ' cases or deaths'f
    
    """

    df_country = df[(df['id'] == country)].copy()
    df_country['date'] = pd.DatetimeIndex(pd.to_datetime(df_country['date'], format='%Y-%m-%d'))
    df_country = df_country.sort_values(by='date').reset_index(drop =True) 
    df_country = df_country.set_index('date')
    
    df_country = df_country[['confirmed', 'deaths']]
    df_country = df_country.dropna(how='all').resample('D').asfreq().ffill().fillna(0)
    
    if typedata == 'cases':
        df_country = df_country['confirmed']
    elif typedata == 'deaths':
        df_country = df_country['deaths']
    else:
        raise Exception(f"typedata='{typedata}' is not supported.")
       
    #df_country =df_country[:-1] drop today  
    
    allcases_ = df_country.values
    
    if len(allcases_) == 0 or np.max(allcases_) < number_startcases:
        print(f"{country} has a number of cases  less than the threshold ", number_startcases)
        return [], []

    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)

    date_ =  pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 

    return allcases, date_ 

def read_countrydata_phe(df, country, number_startcases=30, datasource = "PHE", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day,
    when the number of cases is higher than number-startcases
    typedata ' cases or deaths'f
    
    """

    df_country = df[(df['areaName'] == country)].copy()
    df_country['date'] = pd.DatetimeIndex(pd.to_datetime(df_country['date'], format='%Y-%m-%d'))
    df_country = df_country.sort_values(by='date').reset_index(drop =True) 
    df_country = df_country.set_index('date')
    df_country = df_country[['cumDeaths28DaysByDeathDate', 'cumCasesBySpecimenDate']]
    df_country = df_country.resample('D').asfreq().ffill().fillna(0)

    if typedata == 'cases':
        df_country = df_country['cumCasesBySpecimenDate']
    elif typedata == 'deaths':
        df_country = df_country['cumDeaths28DaysByDeathDate']
    else:
        raise Exception(f"typedata='{typedata}' is not supported.")

    #df_country =df_country[:-1] drop today  

    allcases_ = df_country.values

    if len(allcases_) == 0 or np.max(allcases_) < number_startcases:
        print(f"{country} has a number of cases  less than the threshold ", number_startcases)
        return [], []

    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)

    date_ =  pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 
    
    return allcases, date_

def read_countrydata_spf_dep(df, country, number_startcases=30, datasource = "SPF_DEP", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day, 
    when the number of cases is higher than number-startcases
    """ 

    if typedata == 'cases':
        df_country = df[(df["dep"] == country)].copy()
        df_country["date"] = pd.DatetimeIndex(pd.to_datetime(df_country["jour"], format="%Y-%m-%d"))
        
        df_country = df_country.sort_values(by="date").reset_index(drop =True) 
        df_country = df_country.set_index("date") 
        
        # cl_age90 = 0 includes all the age classes
        df_country = df_country[df_country["cl_age90"] == '0'][['P']]
        df_country['P'] = df_country['P'].astype('int')
        # suppose no NAs
        df_country = df_country.resample('D').asfreq().ffill().fillna(0)
        
        allcases_ = df_country['P'].cumsum().values
    elif typedata == 'deaths':
        df_country = df[(df["dep"] == country)].copy()
        df_country["date"] = pd.DatetimeIndex(pd.to_datetime(df_country["date"], format="%Y-%m-%d"))
        
        df_country = df_country.sort_values(by="date").reset_index(drop =True) 
        df_country = df_country.set_index("date")
        
        df_country = df_country[['dchosp']]
        df_country = df_country.loc[df_country.first_valid_index():df_country.last_valid_index()]
        df_country['dchosp'] = df_country['dchosp'].astype('int')
        df_country = df_country.resample('D').asfreq().ffill().fillna(0)
        
        allcases_ = df_country['dchosp'].values
    else:
        raise Exception(f"typedata='{typedata}' is not supported.")
    
    
    if np.max(allcases_) < number_startcases:
        print("The country has a number of cases less than the threshold ", number_startcases)
        return [], []
    
    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)
    
    date_ =  pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 

    return allcases, date_

def read_countrydata_spf_reg(df, country, number_startcases=30, datasource = "SPF_REG", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day, 
    when the number of cases is higher than number-startcases
    """ 

    if typedata == 'cases':
        df_country = df[(df["reg"] == country)].copy()
        df_country["date"] = pd.DatetimeIndex(pd.to_datetime(df_country["jour"], format="%Y-%m-%d"))
        
        df_country = df_country.sort_values(by="date").reset_index(drop =True) 
        df_country = df_country.set_index("date") 
        
        # cl_age90 = 0 includes all the age classes
        df_country = df_country[df_country["cl_age90"] == '0'][['P']]
        df_country['P'] = df_country['P'].astype('int')
        # suppose no NAs
        df_country = df_country.resample('D').asfreq().ffill().fillna(0)
        
        allcases_ = df_country['P'].cumsum().values
    elif typedata == 'deaths':
        df_country = df[(df["reg"] == country)].copy()
        
        # sum up dep's deaths to get reg's deaths
        df_country = df_country[['reg', 'date', 'dchosp']].dropna()
        df_country['dchosp'] = df_country['dchosp'].astype('int')
        df_country = df_country.groupby(['date', 'reg'])[['dchosp']].sum()
        df_country = df_country.reset_index()
        
        df_country["date"] = pd.DatetimeIndex(pd.to_datetime(df_country["date"], format="%Y-%m-%d"))
        df_country = df_country.sort_values(by="date").reset_index(drop =True) 
        df_country = df_country.set_index("date")
        
        df_country = df_country[['dchosp']]
        df_country = df_country.resample('D').asfreq().ffill().fillna(0)

        allcases_ = df_country['dchosp'].values
    else:
        raise Exception(f"typedata='{typedata}' is not supported.")
    
    
    if np.max(allcases_) < number_startcases:
        print("The country has a number of cases less than the threshold ", number_startcases)
        return [], []
    
    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)
    
    date_ =  pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 

    return allcases, date_

def read_countrydata_spf_fra(df, country, number_startcases=30, datasource = "SPF_FRA", typedata = "cases"):
    """
    read the one dimensional cumulative casts data and the date starting from the day, 
    when the number of cases is higher than number-startcases
    """ 
 
    df_country = df[(df["country"] == country)].copy() 
    df_country["date"] = pd.DatetimeIndex(pd.to_datetime(df_country["date"], format="%Y-%m-%d"))
    
    df_country = df_country.sort_values(by="date").reset_index(drop =True) 
    df_country = df_country.set_index("date") 
    
    if (typedata == 'cases'):
        df_country = df_country['conf']
    elif (typedata == 'deaths'):
        df_country = df_country['dc_tot']
    else:
        raise Exception(f"typedata='{typedata}' is not supported.")
    
    df_country = df_country.dropna().astype('int').resample('D').asfreq().ffill()
    
    allcases_ = df_country

    if np.max(allcases_) < number_startcases:
        print("The country has a number of cases less than the threshold ", number_startcases)
        return [], []
     
    #allcases_ = repair_increasing(allcases_)
    starting_point = np.min(np.where(allcases_ >= number_startcases)[0]) 
    allcases = np.array(allcases_[starting_point:], dtype=float)
    
    date_ =  pd.to_datetime(df_country.index)
    date_ = date_[starting_point:] 

    return allcases, date_

def compute_method_depth(method, kwargs, H=7):
    if method.__name__ in ["benchmark"]:
        return 15
    if method.__name__ in ["mean_const", "linear"]:
        return 7

    depth_influencers = ["p", "W"]
    stp = 0
    if len(kwargs) > 0:
        infl_current = [v for k, v in kwargs.items() if k in list(depth_influencers)]
        if len(infl_current) > 0:
            stp = np.sum(infl_current)
        if method.__name__ not in ["fit_ar_log"]:
            stp = stp + H
        order_ = 0
        if "order_diff" in kwargs:
            order_ = kwargs["order_diff"]
            if "lag_ave" in kwargs:
                order_ = np.min([kwargs["order_diff"], kwargs["lag_ave"]])
        if "daily" in kwargs:
            stp = stp + 1 * (kwargs["daily"])
        if "order0" in kwargs:
            stp = stp + (1 - 1 * (kwargs["order0"]))
        stp = stp + order_
        if method.__name__ in ["fit_forecasting_ar7_daily", "fit_forecast_7ahead"]:
            stp = stp + 1
    if method.__name__ in ["linear_model_fit_predict"]:
        stp = np.max([stp, 15])
    if stp == 0:
        print("parameters of methods defining the depth are not set")
    return stp


def minimal_method_depth(method, kwargs, H=7):
    if method.__name__ in ["benchmark"]:
        return 15 
    if method.__name__ in ["mean_const", "linear"]:
        return 7
    
    stp = 3
    if method.__name__ not in ["fit_ar_log"]:
        stp = stp + H  # from p and W
    if "daily" in kwargs:
        stp = stp + 1 * (kwargs["daily"])
    if "order0" in kwargs:
        stp = stp + (1 - 1 * (kwargs["order0"]))
    if method.__name__ in ["fit_forecasting_ar7_daily", "fit_forecast_7ahead"]:
        stp = stp + 1
    if method.__name__ in ["linear_model_fit_predict"]:
        stp = np.max([stp, 15])

    return stp


def precompute_forecasts(allcases, method, kwargs, smoothing_fun, incremental_history=[],
                          smoothed_history=[], Ws=3, H=7, plot = False):
    """
    allcases: cumulative cases for the particular country
    method: method for which to compute the forecasts
    incremetral_history: if given, list of lists of the histories up to i, where i is from i till the end of obervations
    smoothed_history: smoothed incremental history
    Ws: parameter of smoothing method, which is a part of computing of smoothed_history
    H: forecast length
    
    Output:
    list of arrays with the 1,...,H forecasts for the method  
    stp: which is a minimal observations numbe needed for model tuning
    
    
    includes preprocessing step to find missing values in the end of the history
    if there were missing values found, forecasting horison increases correspondingly
    """
    stp = compute_method_depth(method, kwargs) 
    if stp > len(allcases):
        print("not enough data to fit the method")
        return [], []
    if "ns" in kwargs:
        Ws_method = kwargs["ns"]
    else:
        Ws_method = 3 

    if len(kwargs) == 0:
        kwargs = {} 
   
    if len(smoothed_history[0]) >= stp: 
        forecasts = np.zeros((len(smoothed_history), H + 1)) 
        for i in range(len(incremental_history)):
            try:
                max_nonzero = np.max(np.where(np.diff(incremental_history[i])>0))  
                td = len(incremental_history[i])-max_nonzero-2  
            except:
                td = 0
            H_ = H + td 
            if td>0: 
                try:
                    forecast_ = method(incremental_history[i][:-td],smoothed_history[i][:-td], H=H_, **kwargs)
                except:
                    forecast_ = method(incremental_history[i],smoothed_history[i], H=H_,**kwargs)
                forecast = forecast_[-(H + 1):]
            else:
                forecast_ = method(incremental_history[i],smoothed_history[i], H=H_,**kwargs) 
                forecast = forecast_[-(H + 1):] 
            forecasts[i, :] = forecast
            
            y = np.diff(incremental_history[i])

        return forecasts

    forecasts = np.zeros((len(allcases) - stp + 1, H + 1))   
    for i in range(stp - 1, len(allcases)):  
        try:
            max_nonzero = np.max(np.where(np.diff(incremental_history[i])>0))  
            td = len(incremental_history[i])-max_nonzero-2  
        except:
            td = 0        
        H_ = H + td 
        if td>0:
            print("skipping", td)
            try:
                forecast = method(incremental_history[i][:-td],smoothed_history[i][:-td], H=H_, **kwargs) 
            except:
                print(incremental_history[i][:-td])
        
        else:
            forecast = method(incremental_history[i],smoothed_history[i], H=H_, **kwargs)
        forecasts[i - stp + 1, :] = forecast[-(H + 1):]
 
    return forecasts
