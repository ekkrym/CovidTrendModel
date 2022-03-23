"""
Running deaths forecasts
the same procedure is used is as for cases
the main function with trend estimation and forecasting can be found in forecast_one_country.py 

to run in cmd on JHU or JHU_US: 
python source/run_deaths_from_cases.py --dataset="JHU"  
python source/run_deaths_from_cases.py  --dataset="JHU_US" 

"""
  
import multiprocessing
import os 
import time
import warnings
from datetime import date
import getopt, sys 
import numpy as np
import pandas as pd

from covid19dh import covid19 
sys.path.append('methods_configuration/')
from precomputing import  read_countries, repair_increasing
from smoothing import simple_mirroring, piecewise_STL
from function_configuration import METHODS 
from visualization import plot_last_forecast 
from test_missing_or_zero import test_poisson 
from forecast_one_country import compute_confidence, forecast_one_country

if not sys.warnoptions:
    warnings.simplefilter("ignore")
if not os.path.exists("results/"):
    os.makedirs("results/")

# -------------------------------------------------------------------------------------------------------
# Get full command-line arguments 
# -------------------------------------------------------------------------------------------------------

full_cmd_arguments = sys.argv 
short_options = "d:"
long_options = ["dataset="]
# Keep all but the first
argument_list = full_cmd_arguments[1:]  
path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
datasource, parse_column = "JHU", "Country/Region"
     
    
try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
    for current_argument, current_value in arguments:
        if current_argument in ("-d", "--dataset"): 
            #in case of deaths to change the paths to death data
            if  current_value == "JHU_US":
                path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
                datasource, parse_column = "JHU_US", "Province_State"
            elif current_value == "JHU":
                path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
                datasource, parse_column = "JHU", "Country/Region" 
            elif current_value == 'OZH':
                path_data = "https://raw.githubusercontent.com/openZH/covid_19/master/COVID19_Fallzahlen_CH_total_v2.csv"
                datasource, parse_column = 'OZH', 'abbreviation_canton_and_fl'
            elif current_value == 'BAG':
                path_data = "https://www.bag.admin.ch/dam/bag/fr/dokumente/mt/k-und-i/aktuelle-ausbrueche-pandemien/2019-nCoV/covid-19-datengrundlage-lagebericht.xlsx.download.xlsx/200325_base%20de%20donn%C3%A9es_graphiques_COVID-19-rapport.xlsx"
                datasource, parse_column = 'BAG', 'country'
            elif current_value == 'BAG_KT':
                path_data = "https://www.covid19.admin.ch/api/data/context"
                datasource, parse_column = 'BAG_KT', 'geoRegion'
            elif current_value == 'CAN':
                path_data = "https://api.opencovid.ca/timeseries?stat=mortality&loc=prov"
                datasource, parse_column = 'CAN', 'province'
            elif current_value == 'HUB':
                path_data = None
                datasource, parse_column = 'HUB', 'id'
            elif current_value == 'PHE':
                path_data = 'https://api.coronavirus.data.gov.uk/v2/data?areaType=nation&metric=cumDeaths28DaysByDeathDate&metric=cumCasesBySpecimenDate&format=csv'
                datasource, parse_column = 'PHE', 'areaName'
            elif current_value == 'SPF_DEP':
                path_data = 'https://www.data.gouv.fr/en/datasets/r/5c4e1452-3850-4b59-b11c-3dd51d7fb8b5'
                datasource, parse_column = 'SPF_DEP', 'dep'
            elif current_value == 'SPF_REG':
                path_data = 'https://www.data.gouv.fr/en/datasets/r/5c4e1452-3850-4b59-b11c-3dd51d7fb8b5'
                datasource, parse_column = 'SPF_REG', 'reg'
            elif current_value == 'SPF_FRA':
                path_data = 'https://www.data.gouv.fr/fr/datasets/r/f335f9ea-86e3-4ffa-9684-93c009d5e617'
                datasource, parse_column = 'SPF_FRA', 'country'
                
except getopt.error as err: 
    # Output error, and return with an error code
    print(str(err))
    print("using JHU") 
# -------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    #--------------------setup--------------------
    #default forecast is for H=7 days
    H = 7, 
    #smoothing method
    smoothing_fun = piecewise_STL 
    print(datasource)  
    #-----------------reading the data------------
    if datasource == "BAG":
        df = pd.read_excel(path_data, sheet_name=0, skiprows=6)
        df = df.sort_values('Date').iloc[:-1] # exclude the last data point as it is only data before 8am
        df[parse_column] = 'Switzerland' # add country column
    elif datasource == 'BAG_KT':
        df = pd.read_csv(pd.read_json(path_data).loc['individual', 'sources']['csv']['daily']['death'])
    elif datasource == 'CAN':
        df = pd.DataFrame.from_records(pd.read_json(path_data).values.flatten())
    elif datasource == 'HUB':
        df, _ = covid19(level=1, raw=True, verbose=False)
    elif 'SPF' in datasource:
        df = pd.read_csv(path_data, dtype='str', encoding='latin')
        if 'FRA' in datasource:
            df[parse_column] = 'France'
        else:
            df[parse_column] = parse_column + df[parse_column]
    else:
        df = pd.read_csv(path_data)
     
    # ------------------------------------------------------------------------------------------------------- 
    #loading the extrapolation method: 3 corresponds to linear extrapolation in original scale when trend 
    #is increasing, and in log-scale when decreasing
    #--------------------------------------------------------------------------------------------------------
    subset = [3]  
    methods_df_path = "methods_configuration/selected_methods_const.csv"
    methods_pd = pd.read_csv(methods_df_path) 
    # Read the methods from methods.csv with references given in METHODS 
    methods = [METHODS[x] for x in list(methods_pd["method"].values[subset])]
    kwargs_list = [eval(x) for x in list(methods_pd["parameters"].values[subset])]
    names = list(methods_pd["description"].values[subset])
    cv_flag = list(methods_pd["cv"].astype(int).values[subset])
    
    #-------------------------------countries----------------------------------------------------------------
    countries = list(set(df[parse_column])-set(["Cases_on_an_international_conveyance_Japan", "Diamond_Princess", "Repatriated"])) 
    cumulative_list, date_list = read_countries(df, countries, 0, datasource, typedata = "deaths")
    # -------------------------------forecast deaths---------------------------------------------------------  
    for i in range(len(countries)):
        return_dict, _, ci_ = forecast_one_country({}, countries[i], cumulative_list[i], date_list[i], 
                             methods, kwargs_list, names, smoothing_fun, datasource,   
                             H=7, type_data="deaths", missing_var=True, return_val = True, saveplot=False, newly=True) 
        if i==0:
            deaths_predictions = return_dict[countries[i]][2].copy()
            ci = ci_.copy() 
        else:
            deaths_predictions = deaths_predictions.append(return_dict[countries[i]][2])  
            ci =  ci.append(ci_) 
        print(countries[i], f"{i+1}/{len(countries)} done")
    #----------------------------------------save results----------------------------------------------------- 
    deaths_predictions.to_csv("results/"+ datasource +'_deaths_predictions_'+date.today().strftime("%Y_%m_%d")+'.csv')
    
    ci.to_csv("results/"+ datasource +'_deaths_CI_'+date.today().strftime("%Y_%m_%d")+'.csv')
    
    print(f'{datasource} deaths forecasting -- done')
 

 
