- to run the script with different datasets: python source/run_all_countries.py --dataset=<datasource>, 
  where <datasource> may take values in ["JHU", "JHU_US"] with default "JHU"
examples:
python source/run_all_countries.py --dataset="JHU"
python source/run_deaths_from_cases.py --dataset="JHU" 
python source/run_all_countries.py --dataset="JHU_US"
python source/run_deaths_from_cases.py --dataset="JHU_US"

  
- the forecasts and smoothing results are saved depending on the datasource in the folder "results" to the file with the name (example for cases): 
   <datasource>+ "_cases_predictions_%Y_%m_%d.csv" 
- CI are in 
   <datasource>+ "cases_CI_%Y_%m_%d.csv"

- the smooth trends are saved in the folder "smoothing_res" for all the countries in a separate file corrisponding to 
country-datasource-smoothing_method  
 
