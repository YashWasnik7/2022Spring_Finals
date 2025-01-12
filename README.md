# 2022Spring_Finals
Final Project (Type 1) : COVID-19 Vaccine Hesitancy
Team members: 
1. Asmita Dabholkar
2. Himani Mehta
3. Yash Wasnik

Link to original data analyis: https://www.kaggle.com/code/jaykumar1607/covid-vaccine-hesitancy-usa-plotly-mapbox

We have tried to take the original analysis further to analyze vaccine hesitancy in multi-party governed countries like India.

Hypothesis 1: In multi-party governed countries, states with the same ruling party as the central will have higher vaccinated population.
Hypothesis 2: People in rural regions are more hesitant towards taking the vaccine.

Our project notebooks are divided into three - one for original analysis improvement, and the rest for hypotheses 1 and 2.
Instructions on running the code: 
1. Download the entire folder and place it in a single directory.
2. Download the usa_data.csv from the link provided in readme.txt in Data folder and save the downloaded file in the Data folder itself.
3. Run the 'vaccine_hesitancy_us.ipynb' first. 
4. Run the 'hypothesis_1.ipynb' notebook
5. Run the 'hypothesis_2.ipynb' notebook

Results:
1. Hypothesis 1: We did not find a significant correlation between the state vs center ruling parties and their vaccinations. In fact, some Indian states where the ruling party is not the same as center like Maharashtra, had high vaccination rates. Hence, we reject hypothesis 1.
2. Hypothesis 2: From the literacy data, we found that some states with low literacy index like Uttar Pradesh had higher vaccination where as Kerela, which has the highest literacy, had lower vaccination. Hence, we rejected hypothesis 2.
