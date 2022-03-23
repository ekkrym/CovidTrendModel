"""
Running (parallelized in countries/regions) cases forecasts 
the main function with trend estimation and forecasting can be found in forecast_one_country.py 
 
to run the model in cmd on JHU or JHU_US:
python source/run_all_countries.py --dataset="JHU" 
python source/run_all_countries.py --dataset="JHU_US"

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
from forecast_one_country import forecast_one_country
from precomputing import read_countrydata, read_countries
from smoothing import simple_mirroring, piecewise_STL
from function_configuration import METHODS

if not sys.warnoptions:
    warnings.simplefilter("ignore")
if not os.path.exists("results/"):
    os.makedirs("results/")

# -------------------------------------------------------------------------------------------------------
# Get full command-line arguments  
# -------------------------------------------------------------------------------------------------------

full_cmd_arguments = sys.argv 
short_options = "d:s:t:"
long_options = ["dataset=","subset=","test_missing="]
# Keep all but the first
argument_list = full_cmd_arguments[1:]  
path_data = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv" 
parse_column = "countriesAndTerritories"
datasource = "ECDC"

#path for methods description 
methods_df_path = "methods_configuration/selected_methods_const.csv" 
#current default methods
subset = [3] 
#whether to perform a test with missing values, default is:
missing_test = True
try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
    for current_argument, current_value in arguments:
        if current_argument in ("-d", "--dataset"):
            
            #in case of deaths to change the paths to death data
            if  current_value == "JHU_US":
                path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv" 
                datasource, parse_column = "JHU_US", "Province_State"
            elif current_value == "JHU":
                path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv" 
                datasource, parse_column = "JHU", "Country/Region"
            elif current_value == "ECDC":
                path_data = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv" 
               # path_data = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx"
                datasource, parse_column = "ECDC", "countriesAndTerritories"
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
                path_data = "https://api.opencovid.ca/timeseries?stat=cases&loc=prov"
                datasource, parse_column = 'CAN', 'province'
            elif current_value == 'HUB':
                path_data = None
                datasource, parse_column = 'HUB', 'id'
            elif current_value == 'PHE':
                path_data = 'https://api.coronavirus.data.gov.uk/v2/data?areaType=nation&metric=cumDeaths28DaysByDeathDate&metric=cumCasesBySpecimenDate&format=csv'
                datasource, parse_column = 'PHE', 'areaName'
            elif current_value == 'SPF_DEP':
                path_data = 'https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675'
                datasource, parse_column = 'SPF_DEP', 'dep'
            elif current_value == 'SPF_REG':
                path_data = 'https://www.data.gouv.fr/fr/datasets/r/001aca18-df6a-45c8-89e6-f82d689e6c01'
                datasource, parse_column = 'SPF_REG', 'reg'
            elif current_value == 'SPF_FRA':
                path_data = 'https://www.data.gouv.fr/fr/datasets/r/f335f9ea-86e3-4ffa-9684-93c009d5e617'
                datasource, parse_column = 'SPF_FRA', 'country'
                
        if current_argument in ("-s", "--subset"):
            try:
                ls = current_value.strip('[]').replace('"', '').replace(' ', '').split(',')
                ls = [int(x) for x in ls]
                if set(ls).issubset(np.arange(9)):
                    subset = ls
            except:
                print("cannot parse the selected methods indices")
               
        if current_argument in ("-t", "--test_missing"):
            if current_argument=="False":
                missing_test = False 
            
#example of using all the arguments:
#python source/run_all_countries.py --dataset="JHU" --subset=[2,3] --test_missing=False
            
except getopt.error as err:  
    print(str(err)) 
# -------------------------------------------------------------------------------------------------------
    
if __name__ == '__main__':
    # Forecast length 
    H = 2*7 
 
    # selecting the smoothing function
    smoothing_fun = piecewise_STL 
    
    # -------------------------------------------------------------------------------------------------------
    # reading the data from datasource    
    # -------------------------------------------------------------------------------------------------------
    if datasource == "BAG":
        df = pd.read_excel(path_data, sheet_name=0, skiprows=6) 
        df = df.sort_values('Date').iloc[:-1] # exclude the last data point as it is only data before 8am
 
        df[parse_column] = 'Switzerland' # add country column
    elif datasource == 'BAG_KT':
        df = pd.read_csv(pd.read_json(path_data).loc['individual', 'sources']['csv']['daily']['cases'])
    elif datasource == 'CAN':
        df = pd.DataFrame.from_records(pd.read_json(path_data).values.flatten())
    elif datasource == 'HUB':
        df, _ = covid19(level=1, raw=True, verbose=False)
    elif 'SPF' in datasource:
        if 'DEP' in datasource:
            df = pd.read_csv(path_data, sep=';', dtype='str')
            df = df[~df.dep.isin(['975', '977', '978'])]
            df[parse_column] = parse_column + df[parse_column]
        if 'REG' in datasource:
            df = pd.read_csv(path_data, sep=';', dtype='str')
            df = df[~df.reg.isin(['05', '07', '08'])]
            df[parse_column] = parse_column + df[parse_column]
        if 'FRA' in datasource:
            df = pd.read_csv(path_data, dtype='str', encoding='latin')
            df[parse_column] = 'France'
    else:
        df = pd.read_csv(path_data) 
    countries = list(set(df[parse_column])-set(["Cases_on_an_international_conveyance_Japan", "Diamond_Princess", "Repatriated"]))   
    cumulative_list, date_list = read_countries(df, countries, 0, datasource, typedata = "cases")
    # ------------------------------------------------------------------------------------------------------- 
   
    # loading the methods 
    methods_pd = pd.read_csv(methods_df_path)
    # ------------------------------------------------------------------------------------------------------- 
    # Read the methods with references given in METHODS  
    methods = [METHODS[x] for x in list(methods_pd["method"].values[subset])]
    kwargs_list = [eval(x) for x in list(methods_pd["parameters"].values[subset])]
    names = list(methods_pd["description"].values[subset])
    cv_flag = list(methods_pd["cv"].astype(int).values[subset])
    # --------------------------------------forecasting parallellized in countries/regions------------------- 
    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    start_time = time.time()
    for i in range(len(countries)):
        p = multiprocessing.Process(target=forecast_one_country,
                                    args=(return_dict, countries[i], cumulative_list[i], date_list[i],
                                          methods, kwargs_list, names, 
                                          smoothing_fun, datasource,  H, "cases",
                                          missing_test)) 
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
    print('That took {} seconds'.format(time.time() - start_time))
    
    countries = list(return_dict.keys())
    best_methods = np.array([return_dict[country][0] for country in countries])
    forecasts_ = np.array([return_dict[country][1] for country in countries]) 
    smoothed_pd = pd.concat([return_dict[country][2] for country in countries]) 
    ci_pd = pd.concat([return_dict[country][3] for country in countries])
    smoothed_pd["country"] = smoothed_pd["country"].str.replace(r"[^a-zA-Z0-9]+", " ").str.strip()
    ci_pd["country"] = ci_pd["country"].str.replace(r"[^a-zA-Z0-9]+", " ").str.strip()

    # ------------------------------------------------------------------------------------------------------- 
    # save results
    # -------------------------------------------------------------------------------------------------------   
    
    smoothed_pd.to_csv("results/"+ datasource+ "_cases_predictions_"   + date.today().strftime("%Y_%m_%d") + ".csv")
    ci_pd.to_csv("results/"+ datasource+ "_cases_CI_"   + date.today().strftime("%Y_%m_%d") + ".csv")
 
