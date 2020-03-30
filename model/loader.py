import pandas as pd
import datetime as dt
from dateutil.parser import parse

PATH_andamento_nazionale_remote = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
PATH_andamento_nazionale_local = 'data/dpc-covid19-ita-andamento-nazionale.csv'
PATH_andamento_regionale_remote = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'
PATH_andamento_regionale_local = 'data/ddpc-covid19-ita-regioni.csv'

PATH_confirmed_CSSE_remote = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
PATH_confirmed_CSSE_local = 'data/time_series_covid19_confirmed_global.csv'
PATH_deaths_CSSE_remote = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
PATH_deaths_CSSE_local = 'data/time_series_covid19_deaths_global.csv'
PATH_recovered_CSSE_remote = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
PATH_recovered_CSSE_local = 'data/time_series_covid19_recovered_global.csv'


def load_from_PC(remote=False):
    path = PATH_andamento_nazionale_remote if remote else PATH_andamento_nazionale_local
    df = pd.read_csv(path)
    #dates = pd.DatetimeIndex([parse(d).strftime('%Y-%m-%dT%H:%M:%S') for d in df['data']])
    return df['totale_casi'], df['deceduti'], df['dimessi_guariti']


def load_Region_from_PC(region, date= None, remote=False):
    path = PATH_andamento_regionale_remote if remote else PATH_andamento_regionale_local
    df = pd.read_csv(path)
    regions_df = df[df['denominazione_regione'] == region]
    return regions_df['totale_casi'], regions_df['deceduti'], regions_df['dimessi_guariti']


def load_Province_from_PC(province):
    df = pd.read_csv('data/dpc-covid19-ita-andamento-nazionale.csv')
    return df['totale_casi'], df['deceduti'], df['dimessi_guariti']


def load_from_NCVS(country, start_date, remote=False):
    return _load_confirmed(country, start_date, remote), _load_deaths(country, start_date, remote), _load_recovered(country, start_date, remote)


def _load_confirmed(country, start_date, remote):
    path = PATH_confirmed_CSSE_remote if remote else PATH_confirmed_CSSE_local
    df = pd.read_csv(path)
    country_df = df[df['Country/Region'] == country]
    return country_df.iloc[0].loc[start_date:]


def _load_recovered(country, start_date, remote):
    path = PATH_recovered_CSSE_remote if remote else PATH_recovered_CSSE_local
    df = pd.read_csv(path)
    country_df = df[df['Country/Region'] == country]
    return country_df.iloc[0].loc[start_date:]


def _load_deaths(country, start_date, remote):
    path = PATH_deaths_CSSE_remote if remote else PATH_deaths_CSSE_local
    df = pd.read_csv(path)
    country_df = df[df['Country/Region'] == country]
    return country_df.iloc[0].loc[start_date:]
