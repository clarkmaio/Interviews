from dataclasses import dataclass
from urllib.request import urlopen
import os
import pandas as pd
import numpy as np
from io import StringIO

import seaborn
import tqdm
import matplotlib.pyplot as plt
import re
from datetime import datetime
from typing import Union

import seaborn as sns

from StatkraftAssessment.utils.calendar_helper import CalendarGenerator


MAP_USACE_PROJECT = {
    'dworshak dam': 'dworshak',
    'albeni falls dam': 'albeni falls',
    'albeni falls dam on': 'albeni falls',
    'big cliff dam': 'big cliff',
    'bonneville dam & lak': 'bonneville',
    'cougar dam': 'cougar',
    'detroit dam': 'detroit',
    'dexter dam & lake on': 'dexter',
    'green peter dam': 'green peter',
    'hills creek dam & la': 'hills creek',
    'john day dam & lake': 'john day',
    'lookout point dam &': 'lookout point',
    'mcnary dam & lake wa': 'mcnary',
    'the dalles dam & lak': 'the dalles',
    'chief joseph dam & e': 'chief joseph',
    'ice harbor dam': 'ice harbor',
    'little goose dam & l': 'little goose',
    'lower monumental dam': 'lower monumental',
    'libby dam & lake lib': 'libby',
    'libby dam & lake koo': 'libby',
    'lost creek dam': 'lost creek',
    'lower granite dam &': 'lower granite',
    'foster dam': 'foster',
    'grand coulee dam': 'grand coulee',
    'hungry horse dam': 'hungry horse'
}

COLUMBIA_USACE_PROJECT = ['dworshak', 'albeni fallas', 'bonneville', 'john day', 'mcnary',
                          'the dalles', 'chief joseph', 'ice harbor', 'little goose', 'lower monumental',
                          'libby', 'lower granite', 'grand coulee', 'hungry horse']

USGS_SITE_MAP = {
    14105700: 'dellas',
    14103000: 'deuschutes',
    14048000: 'john_day',
    14033500: 'umatilla'
}

@dataclass
class DataLoader:
    path: str

    def __post_init__(self):
        return

    def load(self):
        '''
        Load all datasets and concat
        '''


        # ----- Load Calendar
        calendar = CalendarGenerator().generate_calendar(start_year = 2007, end_year = 2023, freq='D')

        # ----- Load BPA
        bpa = self._load_BPA()

        # ----- Load USGS
        usgs = self._load_USGS()
        usgs.dropna(inplace=True)
        usgs_hourly = usgs.resample('H').mean()

        # ----- Load USACE
        usace = self._load_USACE()

        # Scale
        data = self.scale_total_to_columbia(bpa=bpa, usace=usace)
        data = pd.merge(data, usgs_hourly, left_index=True, right_index=True, how='left') # concat usgs data

        # Resample to daily
        data_daily = data.resample('D').sum()
        data_daily = pd.merge(data_daily, calendar, left_index=True, right_index=True, how = 'left')

        return data_daily

    def scale_total_to_columbia(self, bpa: pd.DataFrame, usace: pd.DataFrame) -> pd.DataFrame:
        '''
        Use monthly report USACE to scale BPA Total system hydro generation and obtain total hydro generation in Columbia river

        :return:
        '''

        # Monthly aggregation BPA
        bpa_month_agg = bpa.resample('MS').sum()

        # Monthly aggregation usace all projects
        usace_agg_total = usace.groupby('valuedate').sum()
        usace_agg_total.sort_index(inplace=True)

        # Monthly aggregation usace columbia projects
        usace_columbia = usace.loc[usace['project'].isin(COLUMBIA_USACE_PROJECT), :].copy() # Filter Columbia
        usace_agg_columbia = usace_columbia.groupby('valuedate').sum()
        usace_agg_columbia.sort_index(inplace=True)
        usace_agg_columbia = usace_agg_columbia[['total_generation']]
        usace_agg_columbia.columns = ['total_generation_columbia']

        # Scale bpa total_hydro to obtain total hydro columbia river
        data = bpa.copy()
        data['month_date'] = data.index.to_period('M').to_timestamp()
        data = pd.merge(data, bpa_month_agg[['total_hydro']], left_on='month_date', right_index=True, suffixes=('', '_monthly'))
        data = pd.merge(data, usace_agg_columbia.rename(columns={'total_generation_columbia': 'total_hydro_columbia_monthly'}), left_on='month_date', right_index=True, suffixes=('', '_monthly'))
        data['total_hydro_columbia'] = data['total_hydro'] * data['total_hydro_columbia_monthly'] / data['total_hydro_monthly']

        return data

    def _load_BPA(self) -> pd.DataFrame:
        filepath = os.path.join(self.path, 'BPA', 'BPA_system_generation.hdf')
        df = pd.read_hdf(filepath, key='table')

        # Merge fossil and thermal
        df['total_thermal'] = df['total_thermal'].mask(df['total_thermal'].isna(), df['total_fossil'])
        df.drop(columns=['total_fossil'], inplace=True)

        return df

    def _load_USGS(self):

        df_list = []
        folder_path = os.path.join(self.path, 'USGS')
        for f in os.listdir(folder_path):
            id = int(f.split('_')[0])
            station_name = USGS_SITE_MAP[id]

            df_tmp = pd.read_hdf(os.path.join(folder_path, f), key='table')
            df_tmp.columns = df_tmp.columns + '_' + station_name
            df_list.append(df_tmp)
        df = pd.concat(df_list, axis=1)
        return df

    def _load_USACE(self) -> pd.DataFrame:
        filepath = os.path.join(self.path, 'USACE', 'USACE_monthly_report.hdf')
        df = pd.read_hdf(filepath, key='table')
        return df

    def post_process(self):
        return


def split_string_multiple_whitespace(string: str, minimum_number_whitespace: int = 1):
    """Splits a string on multiple whitespace.

    Args:
        string (str): The string to split.
        minimum_number_whitespace (int): The minimum number of whitespace to split on.

    Returns:
        str: The split string.
    """
    return re.sub(r'\s{' + str(minimum_number_whitespace) + ',}', '\t', string)

def read_txt_from_url(url):
    """Reads a text file from a url and returns the content as a string.

    Args:
        url (str): The url to the text file.

    Returns:
        str: The content of the text file.
    """
    with urlopen(url) as response:
        return response.read().decode("utf-8")

def parse_txt_to_df(txt, sep='\t'):
    """Parses a text file to a pandas DataFrame.

    Args:
        txt (str): The text file.
        sep (str, optional): The separator. Defaults to '\t'.
        newline (str, optional): The newline character. Defaults to '\n'.

    Returns:
        pd.DataFrame: The parsed DataFrame.
    """
    return pd.read_csv(StringIO(txt), sep=sep)

@dataclass
class DataScraper:

    def __post_init__(self):
        return

    def scrape_USACE_monthly_report(self, start_year: int = 2019, end_year: int = 2020):
        """Scrapes the USACE monthly report for the given years.

        Args:
            start_year (int): The start year.
            end_year (int): The end year.

        Returns:
            str: The scraped data.
        """

        LINK_BASE = 'https://www.nwd-wc.usace.army.mil/ftppub/power/'

        report_list = []
        for y in range(start_year, end_year+ 1):
            print('Scraping USACE monthly report year', y)
            for m in range(1, 13):
                #print('Month', m)
                link_tmp = os.path.join(LINK_BASE, f'pwr_{y}{m:02}.txt')

                try:
                    report_tmp = read_txt_from_url(link_tmp)
                except:
                    print('Could not scrape', link_tmp)
                    continue

                # Workaround for some weird formatting
                if '\n \n' in report_tmp:
                    report_tmp = report_tmp.split('\n \n')[1]

                # Split on newlines
                report_rows = report_tmp.split('\n')

                # Find where data starts
                there_is_dash_line = np.any([x.startswith('--') for x in report_rows])
                if there_is_dash_line:
                    idx_first_data_row = np.where([x.startswith('--') for x in report_rows])[0][0] + 1
                else:
                    idx_first_data_row = np.where(['PROJECT' in x for x in report_rows])[0][0] + 1
                report_rows = report_rows[idx_first_data_row:]  # Remove useless rows


                report_rows = [split_string_multiple_whitespace(string = row, minimum_number_whitespace = 2).split('\t') for row in report_rows]

                columns = ['project', 'total_generation', 'station_use', 'net_generation_to_BPA', 'power_aux_use', 'net_generation_to_FERC']
                month_report = pd.DataFrame(report_rows, columns = columns).dropna()
                month_report = month_report.loc[month_report['project'] != 'TOTAL', :] # Remove total row
                month_report['valuedate'] =  datetime(y, m, 1)

                # Format columns type
                month_report['total_generation'] = month_report['total_generation'].astype(float)
                month_report['station_use'] = month_report['station_use'].astype(float)
                month_report['net_generation_to_BPA'] = month_report['net_generation_to_BPA'].astype(float)
                month_report['power_aux_use'] = month_report['power_aux_use'].astype(float)
                month_report['net_generation_to_FERC'] = month_report['net_generation_to_FERC'].astype(float)

                # Lowercase to uniform names
                month_report['project'] = month_report['project'].str.lower()

                report_list.append(month_report)
        output_report = pd.concat(report_list, axis = 0)
        output_report['project'] = output_report['project'].replace(MAP_USACE_PROJECT)
        return output_report

    def scrape_USGS_Discharge(self, site_id: Union[str, int] = 14105700, start_date: datetime = datetime(2007, 1, 1), end_date: datetime = datetime.now()):
        """Scrapes the USGS discharge data for the given site and years.

        Args:
            site_id (str, optional): The site id. Defaults to '14105700'.
            start_year (int, optional): The start year. Defaults to 2019.
            end_year (int, optional): The end year. Defaults to 2023.

        Returns:
            pd.DataFrame: The scraped data.
        """

        site_id = str(site_id)

        # Build url
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        LINK_BASE = f'https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&cb_00065=on&format=rdb&site_no={site_id}&legacy=1&period=&begin_date={start_date_str}&end_date={end_date_str}'

        # Retrieve txt
        data = read_txt_from_url(LINK_BASE)
        data_split = data.split('\n#')

        # Parse txt to DataFrame
        data_rows = data_split[-1].split('\n')
        data_rows = [row.split('\t') for row in data_rows]
        columns = ['agency_cd', 'site_no', 'datetime', 'tz_cd', 'gauge_height', 'gauge_height_cd', 'discharge', 'discharge_cd']
        raw_df = pd.DataFrame(data_rows[3:-1], columns = columns)

        # Format columns
        raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
        raw_df['gauge_height'] = raw_df['gauge_height'].replace('', np.nan)
        raw_df['gauge_height'] = raw_df['gauge_height'].astype(float)
        raw_df['discharge'] = raw_df['discharge'].replace({'': np.nan, 'Eqp': np.nan, 'Ice': np.nan})
        raw_df['discharge'] = raw_df['discharge'].astype(float)
        raw_df.set_index('datetime', inplace=True)

        # Resample to daily and keep only interesting columns
        resample_data = raw_df[['discharge','gauge_height']]
        return resample_data


def load_BPA_file(folder_path: str):

    df_list = []
    for file in os.listdir(folder_path):

        #file = 'BPA_2009.xlsx'
        print('Processing file', file)
        file_path = os.path.join(folder_path, file)

        df_tmp = pd.read_excel(file_path)
        df_tmp['valuedate'] = pd.to_datetime(df_tmp['valuedate'])
        df_tmp.set_index('valuedate', inplace=True)

        # Format data
        df_tmp = df_tmp.replace({'Suspect': np.nan})
        df_tmp = df_tmp.astype(float)

        # Resample hourly
        df_tmp = df_tmp.resample('H').mean()

        df_list.append(df_tmp)

    df_output = pd.concat(df_list, axis = 0)
    return df_output


if __name__ == '__main__':
    scraper = DataScraper()
    ## USACE monthy report Columbia river
    #df = scraper.scrape_USACE_monthly_report(start_year=2007, end_year=2023)
    #df.to_hdf('/home/clarkmaio/workspace/StatkraftAssessment/data/USACE/USACE_monthly_report.hdf', key='table')

    ## USGS discharge
    #for id in [14105700, 14103000, 14048000, 14033500]:
    #    print('Processing site', id)
    #    df = scraper.scrape_USGS_Discharge(site_id=id)
    #    df.to_hdf(f'/home/clarkmaio/workspace/StatkraftAssessment/data/USGS/{id}_Discharge.hdf', key='table')

    ## BPA system generation
    #df = load_BPA_file(folder_path = '/home/clarkmaio/Scaricati/BPA/BPA_system_generation')
    #df.to_hdf('/home/clarkmaio/workspace/StatkraftAssessment/data/BPA_system_generation.hdf', key='table')


    #data_loader = DataLoader(path='./data/')
    #df = data_loader.load()
    #df.to_hdf('./data/processed_data.hdf', key='table')
