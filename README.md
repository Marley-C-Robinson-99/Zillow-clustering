## Project Goals:
>    - Create documented files to clean and prepare Zillow dataset for processing by clustering algorithm for further modeling via a regression ML algorithm.
>    - Use clustering to identify driving features of log error and to potentially create new feature combinations or simply to use the cluster as a driving feature.
>    - Evaluate the clusters and try to extrapolate usefull info from them.
>    - Pass updated dataframe with new features/chosen features/cluster features to a regression model.
>    - Present on my final Jupyter Notebook, giving a high-level overview of the process used to create the clusters and how the information garnered from them is useful to my regression model.



## Data dictionary
Target  | Description   | Data Type
--|--|--
log_error    | The logarithmic error of the existing Zestimate | float64

Categorical Features   | Description |    Data Type
--|--|--
quality_id | A numeric representation of quality | int64
heating_type |    Type of heating that a property uses    | object
bedrooms    |   Count of bedrooms per property | float64
bathrooms    |   Count of bathrooms per property | float64
city_id    | Numeric id for the city of each property | int64
county_id    | Numeric id for the county of each property | object
zip_code    | Zip code of each property | int64
roomcnt    | Spare room count of each property | int64


Date/Time Features  | Description | Data Type
--|--|--
sale_date    | Date that the trasnaction of the property was finalized | object, datetime64[ns]
year_built |    Year a home was constructed    | int64

Continuous Features | Description | Data Type
--|--|--
home_area    |   Area of the actual home in square feet | float64
lot_area    |   Area of the property lot in square feet | float64
latitude    | Latitude coordinates of the property | float64
longitude    | Longitude coordinates of the property | float64
structure_tax    | Tax value of the structure itself in dollars | float64
land_tax_value    | Tax value of the land in dollars | float64
tax_value    | Tax value of the entire property | float64
tax_paid    | Tax value paid | float64
census_tb    | Census tract and block | float64


Engineered Features  | Description   | Data Type
--|--|--
county |    Derrived from fips, denotes the actual county of a home    | object
yearly_tax |    Tax paid per year |    float64
tax_rate |    tax rate of a property, derrived from (yearly_tax / tax_value) * 100 |    float64
month_sold |    tax rate of a property, derrived from (yearly_tax / tax_value) * 100 |    float64


## Hypotheses:
>   - $H_{i}$: The key drivers of log error will be linked to tax value clusters and area clusters


## Plan:
- [x] Create repo on github to save all files related to the project.
- [x] Create README.md with goals, initial hypotheses, data dictionary, and outline plans for the project in a trello board.
- [x] Acqiure zillow data using acquire.py file drawing directly from Codeups `zillow` database with SQL queries. Create functions for use in conjunction with prepare.py.
- [x] Clean, tidy, and encode data in such a way that it is usable in a machine learning algorithm. Includes dropping unneccesary columns, creating dummies where needed and changing string values to numeric values and getting rid of outliers
- [] Utilize recursive feature elimination and clustering algorithms to search the data for meaningful driving features of log error.
- [] Create hypotheses based on preliminary statistical tests
- [] Test hypotheses with tests such as t-test, chi-squared to determine the viability of said hypotheses by comparing p-values to alpha.
- [] Establish a baseline accuracy.
- [] Train three different classification models from OLS, GLM, and Lasso + Lars, testing a variety of parameters and features, both engineered and pre-existing.
- [] Evaluate models using RMSE, R^2 score, and other metrics on in-sample and out-of-sample datasets.
- [] Once a single, best preforming model has been chosen, evaluate the preformance of the model on the test dataset.
- [] Present my jupyter notebook to Codeup instructors