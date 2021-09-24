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
bedrooms    |   Count of bedrooms per property | float64
bathrooms    |   Count of bathrooms per property | float64
year_built |    Year a home was constructed    | object
taxamount |    Amount paid in taxes so far   | float64
fips |        Numeric county code    | object


Continuous Features | Description | Data Type
--|--|--
area | Internal square footage of a home | float64

Engineered Features  | Description   | Data Type
--|--|--
county |    Derrived from fips, denotes the actual county of a home    | object
tax_rate |    tax rate of a property, derrived from (taxamount / tax_value) * 100 |    float64



## Hypotheses:
>   - $H_{i}$: The key drivers of log error will be linked to tax value clusters and area clusters


## Plan:
- [x] Create repo on github to save all files related to the project.
- [x] Create README.md with goals, initial hypotheses, data dictionary, and outline plans for the project in a trello board.
- [x] Acqiure zillow data using acquire.py file drawing directly from Codeups `zillow` database with SQL queries. Create functions for use in conjunction with prepare.py.
- [] Clean, tidy, and encode data in such a way that it is usable in a machine learning algorithm. Includes dropping unneccesary columns, creating dummies where needed and changing string values to numeric values and getting rid of outliers
- [] Utilize recursive feature elimination and clustering algorithms to search the data for meaningful driving features of log error.
- [] Create hypotheses based on preliminary statistical tests
- [] Test hypotheses with tests such as t-test, chi-squared to determine the viability of said hypotheses by comparing p-values to alpha.
- [] Establish a baseline accuracy.
- [] Train three different classification models from OLS, GLM, and Lasso + Lars, testing a variety of parameters and features, both engineered and pre-existing.
- [] Evaluate models using RMSE, R^2 score, and other metrics on in-sample and out-of-sample datasets.
- [] Once a single, best preforming model has been chosen, evaluate the preformance of the model on the test dataset.
- [] Present my jupyter notebook to Codeup instructors