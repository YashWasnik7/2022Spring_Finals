import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cdc_data_prep(data: pd.DataFrame) -> pd.DataFrame:
    """
    The function takes the CDC data and processes it by dropping, renaming, grouping,
    and scaling columns on a state level.
    :param data: CDC hesitancy dataset
    """
    # dropping irrelevant columns
    data = data.drop(['Geographical Point', 'County Boundary', 'State Boundary'], 1)

    # renaming the columns
    data = data.rename(columns={'Estimated hesitant': 'low hesitancy', 'Estimated strongly hesitant': 'high hesitancy',
                                'Social Vulnerability Index (SVI)': 'svi',
                                'CVAC level of concern for vaccination rollout': 'cvac',
                                'Percent adults fully vaccinated against COVID-19': 'fully vaccinated',
                                'Percent Hispanic': 'hispanic',
                                'Percent non-Hispanic American Indian/Alaska Native': 'native',
                                'Percent non-Hispanic Asian': 'asian',
                                'Percent non-Hispanic Black': 'black',
                                'Percent non-Hispanic Native Hawaiian/Pacific Islander': 'pacific',
                                'Percent non-Hispanic White': 'white'})

    # grouping the county level data to state level
    statewise_df = data.groupby("State")["State Code", "State", "low hesitancy", "high hesitancy", "svi",
                                         "cvac", "fully vaccinated", "hispanic", "native",
                                         "asian", "black", "pacific", "white"].mean()

    # scaling the dataframe
    statewise_df['low hesitancy'] = (statewise_df['low hesitancy'] - statewise_df['low hesitancy'].min()) / (
                statewise_df['low hesitancy'].max() - statewise_df['low hesitancy'].min())
    statewise_df['high hesitancy'] = (statewise_df['high hesitancy'] - statewise_df['high hesitancy'].min()) / (
                statewise_df['high hesitancy'].max() - statewise_df['high hesitancy'].min())
    statewise_df['fully vaccinated'] = (statewise_df['fully vaccinated'] - statewise_df['fully vaccinated'].min()) / (
                statewise_df['fully vaccinated'].max() - statewise_df['fully vaccinated'].min())
    statewise_df['svi'] = (statewise_df['svi'] - statewise_df['svi'].min()) / (
                statewise_df['svi'].max() - statewise_df['svi'].min())
    statewise_df['cvac'] = (statewise_df['cvac'] - statewise_df['cvac'].min()) / (
                statewise_df['cvac'].max() - statewise_df['cvac'].min())

    return statewise_df


def cdc_data_plot(df: pd.DataFrame, col: str, title: str) -> None:
    """
    The function plots the percentage of fully vaccinated people in a state
    against the column given as input.
    :param df: statewise processed CDC dataset
    :param col: column which is to be plotted against percentage population fully vaccinated
    :param title: title of the plot
    """
    plot_title = title
    df[[col,'fully vaccinated']].sort_values(col).plot(title = plot_title, kind = 'bar', figsize=(20,5))
    return None


def covid_data_prep(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    The function processes the data for a given country for further analysis
    :param df: world daily covid cases data
    :param country: country for which data needs to be processed
    """
    cases = df[df['country']==country][['date','country','cumulative_total_cases','daily_new_cases']]
    cases['month'] = pd.DatetimeIndex(cases['date']).month
    cases['year'] = pd.DatetimeIndex(cases['date']).year
    cases = cases.reset_index()
    cases.drop('index', axis=1, inplace=True)
    cases.drop('cumulative_total_cases', axis=1, inplace=True)

    # grouping the dataframe by month-year
    cases_monthwise = cases.groupby(['month','year']).sum().reset_index()
    cases_monthwise['month_year'] =  cases_monthwise['year'].apply(str) + '-' + cases_monthwise['month'].apply(str)
    cases_monthwise.sort_values(by=['year', 'month'], ascending = True, inplace = True)

    return cases_monthwise


def covid_data_plot(df1: pd.DataFrame, df2: pd.DataFrame, country1: str, country2: str) -> None:
    """
    The function takes dataframes for two countries and creates a time series plot
    of the COVID-19 cases in those countries
    :param country2: name of second country
    :param country1: name of the first country
    :param df1: processed data for country1
    :param df2: processed data for country2
    """
    # merging both the dataframes
    covid_cases = pd.merge(left=df1, right=df2,
                           how='inner', left_on='month_year', right_on='month_year')

    # renaming the columns
    covid_cases = covid_cases.rename(
        columns={'daily_new_cases_x': 'cases_' + country1, 'daily_new_cases_y': 'cases_' + country2})
    covid_cases.drop(['month_y', 'year_y'], axis=1, inplace=True)

    # scaling the data
    covid_cases['cases_' + country1] = (covid_cases['cases_' + country1] - covid_cases['cases_' + country1].min()) / (
                covid_cases['cases_' + country1].max() - covid_cases['cases_' + country1].min())
    covid_cases['cases_' + country2] = (covid_cases['cases_' + country2] - covid_cases['cases_' + country2].min()) / (
                covid_cases['cases_' + country2].max() - covid_cases['cases_' + country2].min())

    # plotting the dataframe
    plot_title = "Monthly COVID-19 Cases"

    x = 'month_year'
    y = ['cases_' + country1, 'cases_' + country2]

    covid_cases.plot(x, y, figsize=(20, 5))
    plt.title(plot_title)
    plt.xlabel('month-year')
    plt.ylabel('no. of cases')

    return None


def vaccine_data_prep(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    The function takes vaccination data and processes it for further analysis.
    :param df: vaccination dataset
    :param country: country name
    """
    if country == 'India':
        df['month'] = pd.DatetimeIndex(df['Updated On']).month
        df['year'] = pd.DatetimeIndex(df['Updated On']).year
    else:
        df['month'] = pd.DatetimeIndex(df['Date']).month
        df['year'] = pd.DatetimeIndex(df['Date']).year

    df_monthwise = df.groupby(['month','year']).sum().reset_index()
    df_monthwise['month_year'] =  df_monthwise['year'].apply(str) + '-' + df_monthwise['month'].apply(str)

    return df_monthwise


def vaccine_data_plot(df1: pd.DataFrame, df2: pd.DataFrame, country1: str, country2: str) -> None:
    """
    The function creates a plot for doses administered for given two countries.
    :param country2: name of the second country
    :param country1: name of the first country
    :param df1: dataframe of country1
    :param df2: dataframe of country2
    """
    # merge the dataframes
    monthly_vax = pd.merge(left=df1, right=df2,
                           how='inner', left_on='month_year', right_on='month_year')

    # drop and rename columns
    monthly_vax = monthly_vax.rename(columns={'Total Doses Administered': 'doses_' + country1,
                                              'Total Doses Administered Daily': 'doses_' + country2})
    monthly_vax.drop(['month_y', 'year_y'], axis=1, inplace=True)

    # scaling the data
    monthly_vax['doses_' + country1] = (monthly_vax['doses_' + country1] - monthly_vax['doses_' + country1].min()) / (
                monthly_vax['doses_' + country1].max() - monthly_vax['doses_' + country1].min())
    monthly_vax['doses_' + country2] = (monthly_vax['doses_' + country2] - monthly_vax['doses_' + country2].min()) / (
                monthly_vax['doses_' + country2].max() - monthly_vax['doses_' + country2].min())

    # plot the dataframe
    plot_title = "Monthly Doses Administered"

    x = 'month_year'
    y = ['doses_' + country1, 'doses_' + country2]

    monthly_vax.plot(x, y, figsize=(20, 5))
    plt.title(plot_title)
    plt.xlabel('month-year')
    plt.ylabel('no. of doses')

    return None


def vaccine_gender_plot(df: pd.DataFrame) -> None:
    """
    The function creates a plot for gender wise COVID-19 doses administered in India
    :param df: vaccination data of India
    """
    # scaling the data
    df['Male (Individuals Vaccinated)'] = (
                (df['Male (Individuals Vaccinated)'] - df['Male (Individuals Vaccinated)'].min())
                / (df['Male (Individuals Vaccinated)'].max() - df['Male (Individuals Vaccinated)'].min())).round(2)

    df['Female (Individuals Vaccinated)'] = (
                (df['Female (Individuals Vaccinated)'] - df['Female (Individuals Vaccinated)'].min())
                / (df['Female (Individuals Vaccinated)'].max() - df['Female (Individuals Vaccinated)'].min())).round(2)

    df['Transgender (Individuals Vaccinated)'] = (
                (df['Transgender (Individuals Vaccinated)'] - df['Transgender (Individuals Vaccinated)'].min())
                / (df['Transgender (Individuals Vaccinated)'].max() - df[
            'Transgender (Individuals Vaccinated)'].min())).round(2)

    # plot male/female/trans people vaccine uptake trends in india
    plot_title = "Vaccination trends of males, females, and transgender people in India"
    x = "month_year"
    y = ["Male (Individuals Vaccinated)", "Female (Individuals Vaccinated)", "Transgender (Individuals Vaccinated)"]

    df.plot(x, y, figsize=(20, 5))
    plt.title(plot_title)
    plt.xlabel('month-year')
    plt.ylabel('no. of individuals')
    
    
def state_wise_vaccination_plot(state):
    """
    Plots a bar graph displaying the monthly vaccination of a state
    :param state: name of the state
    """
    state_doses = df[df['State'] == state][['State','Total Doses Administered', 'month']]
    sns.barplot('month', 'Total Doses Administered', data = state_doses)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower left', ncol=1)
