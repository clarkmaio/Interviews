from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

@dataclass
class CalendarGenerator:

    def __post_init__(self):
        return

    def generate_calendar(self, start_year: int = 2019, end_year: int = 2020, freq: str = 'D') -> pd.DataFrame:
        """
        Generates a calendar info for a given interval of time

        """

        index = pd.Index(pd.date_range(start = datetime(start_year, 1, 1), end = datetime(end_year, 12, 31, 23), freq = freq))
        calendar = pd.DataFrame(index = index, columns=['year', 'month', 'day', 'hour', 'weekday', 'week', 'quarter', 'dayofyear', 'dayofweek'])

        # Build calendar features
        calendar['year'] = calendar.index.year
        calendar['month'] = calendar.index.month
        calendar['day'] = calendar.index.day
        calendar['hour'] = calendar.index.hour
        calendar['weekday'] = calendar.index.weekday
        calendar['week'] = calendar.index.week
        calendar['quarter'] = calendar.index.quarter
        calendar['dayofyear'] = calendar.index.dayofyear
        calendar['dayofweek'] = calendar.index.dayofweek
        calendar['year_category'] = calendar['year'].astype(str)
        calendar['month_category'] = calendar['month'].astype(str)

        return calendar


if __name__ == '__main__':
    calendar = CalendarGenerator().generate_calendar()
    calendar.plot(subplots = True)
    plt.show(block = True)