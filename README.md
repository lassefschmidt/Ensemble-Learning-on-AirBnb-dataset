# Ensemble-Learning-on-AirBnb-dataset
Group project in Ensemble Learning Class at CentraleSupelec

# About the Project
Within this project, we will apply ensemble learning methods to predict the prices of \emph{AirBnb} listings in New York City. The data set for this project was taken from https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data and describes the listing activity and relevant metrics of over 47,000 AirBnb listings in New York City during the year of 2019.

Our task is to apply and compare all approaches taught in the course and practised in the lab sessions, such as Decision Trees as well as common Bagging and Boosting algorithms, to predict the price of each listing in the provided data set.

# Files Descriptions
- data: folder containing all additional datasets used to enrich original AirBnb data
- images: folder containing images (plots, graphs, tables...) used in the final report
- maps: contains html maps of New York City to visualize some features (e.g. quality of predictions)
- models: folder containing all joblib model files for each best tuned model (except Tree Bagging Regressor, too heavy for github)
- resources: folder containing research papers from which we used their methodology
- util: folder containing python preprocessing script (implemented as a PreprocessingPipeline class with all its functions)
- Exploratory Data Analysis and Preprocessing: python notebook (self-explanatory)
- Feature Selection: python notebook (self-explanatory)
- Model Training and Evaluation: python notebook (self-explanatory)
- AirBnb Price Prediction Project: pdf file detailing project instructions
- README + requirements files

# Additional Data
NYC State Data: <br />
2020_NYC_Boroughs.csv --> https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm  <br />
2020_NYC_Community Districts.csv --> https://data.cityofnewyork.us/City-Government/2020-Community-District-Tabulation-Areas-CDTAs-Map/xrfd-bjik  <br />
2020_NYC_Neighbourhoods.csv --> https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-Mapped/4hft-v355  <br />
2020_NYC_Census_Tracts.csv --> https://data.cityofnewyork.us/City-Government/2020-Census-Tracts-Mapped/weqx-t5xr  <br />
NYC_Subway_Entrances.csv --> https://data.cityofnewyork.us/Transportation/Subway-Entrances/drex-xx56  <br />
NYC_Museums.csv --> https://data.cityofnewyork.us/Recreation/New-York-City-Museums/ekax-ky3z  <br />

American Community Survey (most extensive nationwide survey, annually released):  <br />
2020_NYC_Censusdata_Pop_Housing.xlsx --> https://www.nyc.gov/assets/planning/download/office/planning-level/nycpopulation/census2020/nyc_decennialcensusdata_2010_2020_change.xlsx  <br />
20162020_NYC_Censusdata_Econ.xlsx --> https://www.nyc.gov/site/planning/planning-level/nyc-population/american-community-survey.page.page  <br />

New York City Housing Prices: <br />
https://www.yourlawyer.com/library/nyc-housing-prices-by-borough-and-neighborhood/

# Requirements
geopandas==0.12.2  <br />
joblib==1.2.0  <br />
matplotlib==3.6.1  <br />
numpy==1.23.5  <br />
pandas==1.5.3  <br />
Pillow==9.4.0  <br />
ray==2.3.0 <br />
scikit_learn==1.2.2 <br />
seaborn==0.12.2 <br />
shapely==2.0.1 <br />
tqdm==4.64.1 <br />
xgboost==1.6.2 <br />

# References
D. Micci-Barreca, “A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems,” SIGKDD Explor. Newsl. 3, 27–32 (2001).
