
import polars as pl
from typing import Union, List


def extract_country_data(df: pl.DataFrame, country: str, weekly_resample: bool = False, agg_fun: str = 'sum') -> pl.DataFrame:
    """
    Extracts data for a specific country from a DataFrame and apply columns formatting
    Optionally resamples it weekly.
    """
    df_country = df.filter(
        pl.col('region') == country
    ).select(
        pl.exclude('geo_type', 'region', 'alternative_name', 'sub-region', 'country')
    ).unpivot(
        index='transportation_type',
        variable_name='date'
    ).with_columns(
        pl.col('date').str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col('value').cast(pl.Float64)
    ).pivot(
        index='date',
        on='transportation_type',
        values='value'
    ).sort(
        'date'
    )

    if weekly_resample:
        df_country = df_country.group_by_dynamic("date", every="1w")
        if agg_fun == 'sum':
            df_country = df_country.agg(pl.all().sum())
        elif agg_fun == 'mean':
            df_country = df_country.agg(pl.all().mean())
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_fun}")
        df_country = df_country.group_by_dynamic("date", every="1w").agg(pl.all().sum())

    df_country = df_country.with_columns(
        total=pl.sum_horizontal(pl.exclude('date'))
    )   
    return df_country



def myjoin(df1: pl.DataFrame, df2: pl.DataFrame, join_keys: Union[str, List[str]], suffixes: tuple = None, **join_kwargs):
    if suffixes is None:
        suffixes = ('_x', '_y')

    if isinstance(join_keys, str):
        join_keys = [join_keys]
    
    df_join = df1.select(*join_keys, pl.all().exclude(*join_keys).name.suffix(suffixes[0])).join(
        df2.select(*join_keys, pl.all().exclude(*join_keys).name.suffix(suffixes[1])),
        on=join_keys,
        **join_kwargs
    )

    return df_join