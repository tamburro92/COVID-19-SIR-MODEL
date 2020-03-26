import pandas as pd

def load_from_PC():
    df = pd.read_csv('data/dpc-covid19-ita-andamento-nazionale.csv')
    return df['totale_casi'], df['deceduti'], df['dimessi_guariti']

def load_from_NCVS(country, start_date):
    return _load_confirmed(country, start_date), _load_deaths(country, start_date), _load_recovered(country, start_date)

def _load_confirmed(country, start_date):
    df = pd.read_csv('data/time_series_19-covid-Confirmed.csv', delimiter=";")
    country_df = df[df['Country/Region'] == country]
    return country_df.iloc[0].loc[start_date:]


def _load_recovered(country, start_date):
    df = pd.read_csv('data/time_series_19-covid-Recovered.csv', delimiter=";")
    country_df = df[df['Country/Region'] == country]
    return country_df.iloc[0].loc[start_date:]


def _load_deaths(country, start_date):
    df = pd.read_csv('data/time_series_19-covid-Deaths.csv', delimiter=";")
    country_df = df[df['Country/Region'] == country]
    return country_df.iloc[0].loc[start_date:]