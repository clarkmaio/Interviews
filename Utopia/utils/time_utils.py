from datetime import datetime
import pandas as pd
import numpy as np
import sys

def load_calendar(st_date: datetime, en_date: datetime, freq: str = 'D') -> pd.DataFrame:
    '''
    Return main calendar features of days in date range
    '''
    valuedate = pd.DatetimeIndex(
        pd.date_range(start=st_date, end=en_date, freq=freq), name='valuedate'
    )

    calendar = pd.DataFrame(np.nan, index=valuedate, columns=['year', 'month', 'day', 'doy', 'weekday'])

    # Main calendar features
    calendar['year'] = calendar.index.year
    calendar['month'] = calendar.index.month
    calendar['day'] = calendar.index.day
    calendar['doy'] = calendar.index.dayofyear
    calendar['weekday'] = calendar.index.weekday
    calendar['week'] = calendar.index.isocalendar().week


    # Flag Holidays
    calendar['isSummer'] = [isSummer(d) for d in calendar.index]
    calendar['isChristmas'] = [isChristmas(d) for d in calendar.index]


    # Compute sin/cos
    calendar['sin_month'], calendar['cos_month'] = np.sin(2 * np.pi * calendar['month'] / 12.), np.cos(2 * np.pi * calendar['month'] / 12.)
    calendar['sin_doy'], calendar['cos_doy'] = np.sin(2 * np.pi * calendar['doy'] / 365.5), np.cos(2 * np.pi * calendar['doy'] / 365.5)
    calendar['sin_week'], calendar['cos_week'] = np.sin(2 * np.pi * calendar['week'] / 53.), np.cos(2 * np.pi * calendar['week'] / 53.)


    calendar.reset_index(inplace=True)
    calendar['day_idx'] = calendar.index

    return calendar


def isSummer(date: datetime) -> bool:
    '''Set summer as July and August'''
    return date.month in (7,8)

def isChristmas(date: datetime) -> bool:
    return (date.month == 12) and (date.day > 15) and (date.day < 31)



if __name__ == '__main__':


    import matplotlib.pyplot as plt
    calendar = load_calendar(st_date=datetime(2019,1,1), en_date=datetime(2020,12,31), freq='D')



    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    fig.suptitle('Calendar', fontweight='bold')
    ax0_twin = ax[0].twinx()
    ax0_twin.plot(calendar['valuedate'], calendar['month'], label='month', color='blue')
    ax0_twin.plot(calendar['valuedate'], calendar['week'], label='week', color='red')
    ax0_twin.plot(calendar['valuedate'], calendar['weekday'], label='weekday', color='green')
    ax[0].plot(calendar['valuedate'], calendar['doy'], label='doy')
    ax[0].legend(loc='upper left')
    ax0_twin.legend(loc='upper right')
    ax[0].grid(linestyle=':')

    ax[1].plot(calendar['valuedate'], calendar['sin_month'], label='sin month', color='blue')
    ax[1].plot(calendar['valuedate'], calendar['sin_week'], label='sin week', color='red')
    ax[1].plot(calendar['valuedate'], calendar['sin_doy'], label='sin doy', color='green')
    ax[1].legend()
    ax[1].grid(linestyle=':')

    ax[2].axvspan(xmin=datetime(2019, 7, 1), xmax=datetime(2019, 8, 31), color='blue', alpha=.2, label='Summer')
    ax[2].axvspan(xmin=datetime(2020, 7, 1), xmax=datetime(2020, 8, 31), color='blue', alpha=.2)
    ax[2].axvspan(xmin=datetime(2019, 12, 15), xmax=datetime(2019, 12, 31), color='red', alpha=.2, label='Christmas')
    ax[2].axvspan(xmin=datetime(2020, 12, 15), xmax=datetime(2020, 12, 31), color='red', alpha=.2)
    ax[2].legend()
    ax[2].grid(linestyle=':')
