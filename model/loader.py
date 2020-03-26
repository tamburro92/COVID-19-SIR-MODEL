import pandas as pd
import datetime as dt
def load_from_PC():
    df = pd.read_csv('data/dpc-covid19-ita-andamento-nazionale.csv')
    return df['totale_casi'], df['deceduti'], df['dimessi_guariti']

def load_Region_from_PC(region, date=None):

    df = pd.read_csv('data/dpc-covid19-ita-regioni.csv')
    regions_df = df[df['denominazione_regione'] == region]
    if date:
        regions_df = regions_df.iloc[date:]


    return regions_df['totale_casi'], regions_df['deceduti'], regions_df['dimessi_guariti']

def load_Province_from_PC(province):
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