import pyarrow.parquet as pq
import pandas as pd


def get_max_ots_for_billboard(list_hours):
    ots_times = {
        0: 72,
        1: 72,
        2: 72,
        3: 72,
        4: 72,
        5: 72,
        6: 72,
        7: 48,
        8: 48,
        9: 48,
        10: 48,
        11: 72,
        12: 54,
        13: 54,
        14: 54,
        15: 54,
        16: 72,
        17: 54,
        18: 72,
        19: 48,
        20: 48,
        21: 48,
        22: 48,
        23: 72,
    }

    return [ots_times[hour] * 10 for hour in list_hours]


def check_holidays(list_days, check_before_holidays=False):
    if check_before_holidays:
        return [1 if dayofweek == 4 else 0 for dayofweek in list_days]
    else:
        return [0 if dayofweek < 5 else 1 for dayofweek in list_days]


def check_prazd(list_dates, check_before_prazd=False):
    prazd_days_2021 = {
        "1": [1, 2, 3, 4, 5, 6, 7, 8],
        "2": [22, 23],
        "3": [8],
        "4": [],
        "5": [3, 4, 5, 6, 7, 10],
        "6": [14],
        "7": [],
        "8": [],
        "9": [],
        "10": [],
        "11": [4, 5],
        "12": [31]
    }
    date_ = pd.DatetimeIndex(list_dates)
    days = date_.day
    months = date_.month
    result = []

    if check_before_prazd:
        for monthInd in range(len(months)):
            month = str(months[monthInd])
            result.append(1 if month in prazd_days_2021 and (days[monthInd] + 1) in prazd_days_2021[month] else 0)
    else:
        for monthInd in range(len(months)):
            month = str(months[monthInd])
            result.append(1 if month in prazd_days_2021 and days[monthInd] in prazd_days_2021[month] else 0)

    return result


def get_pandas_from_parquet():
    # table = pq.read_table('../rowData/crowd/player=1548/month=2021-6.parquet')
    # table = pq.read_table('../rowData/crowd/player=257/month=2021-6.parquet')
    table = pq.read_table('../rowData/crowd/player=257')

    return table.to_pandas()


try:
    table_pandas = get_pandas_from_parquet()
    table_pandas_ticks = pd.to_datetime(table_pandas['AddedOnTick'], unit='ms')

    # получаем данные по дате записи о MAC для группировки
    table_pandas['year'] = table_pandas_ticks.dt.year
    table_pandas['month'] = table_pandas_ticks.dt.month
    table_pandas['day'] = table_pandas_ticks.dt.day
    table_pandas['hour'] = table_pandas_ticks.dt.hour
    table_pandas['dayofweek'] = table_pandas_ticks.dt.dayofweek

    grouped_data = table_pandas[['Mac', 'AddedOnDate', 'year', 'month', 'day', 'hour', 'dayofweek']].groupby(
        ['AddedOnDate', 'year', 'month', 'day', 'hour',
         'dayofweek']).count()  # подсчет кол-ва мак адресов, сгруппированных по датам

    grouped_data['max_ots'] = get_max_ots_for_billboard(grouped_data.index.get_level_values('hour'))
    grouped_data['is_holiday'] = check_holidays(grouped_data.index.get_level_values('dayofweek'))
    grouped_data['is_before_holiday'] = check_holidays(grouped_data.index.get_level_values('dayofweek'), True)
    grouped_data['is_prazd'] = check_prazd(grouped_data.index.get_level_values('AddedOnDate'))
    grouped_data['is_before_prazd'] = check_prazd(grouped_data.index.get_level_values('AddedOnDate'), True)

    grouped_data['1_h_ago'] = grouped_data['Mac'].shift(periods=1)
    grouped_data['2_h_ago'] = grouped_data['Mac'].shift(periods=2)
    grouped_data['3_h_ago'] = grouped_data['Mac'].shift(periods=3)
    grouped_data['4_h_ago'] = grouped_data['Mac'].shift(periods=4)
    grouped_data['5_h_ago'] = grouped_data['Mac'].shift(periods=5)
    grouped_data['6_h_ago'] = grouped_data['Mac'].shift(periods=6)
    grouped_data['7_h_ago'] = grouped_data['Mac'].shift(periods=7)
    grouped_data['8_h_ago'] = grouped_data['Mac'].shift(periods=8)
    grouped_data['9_h_ago'] = grouped_data['Mac'].shift(periods=9)
    grouped_data['10_h_ago'] = grouped_data['Mac'].shift(periods=10)
    grouped_data['11_h_ago'] = grouped_data['Mac'].shift(periods=11)
    grouped_data['12_h_ago'] = grouped_data['Mac'].shift(periods=12)
    grouped_data['13_h_ago'] = grouped_data['Mac'].shift(periods=13)
    grouped_data['14_h_ago'] = grouped_data['Mac'].shift(periods=14)
    grouped_data['15_h_ago'] = grouped_data['Mac'].shift(periods=15)
    grouped_data['16_h_ago'] = grouped_data['Mac'].shift(periods=16)
    grouped_data['17_h_ago'] = grouped_data['Mac'].shift(periods=17)
    grouped_data['18_h_ago'] = grouped_data['Mac'].shift(periods=18)
    grouped_data['19_h_ago'] = grouped_data['Mac'].shift(periods=19)
    grouped_data['20_h_ago'] = grouped_data['Mac'].shift(periods=20)
    grouped_data['21_h_ago'] = grouped_data['Mac'].shift(periods=21)
    grouped_data['22_h_ago'] = grouped_data['Mac'].shift(periods=22)
    grouped_data['23_h_ago'] = grouped_data['Mac'].shift(periods=23)
    grouped_data['24_h_ago'] = grouped_data['Mac'].shift(periods=24)

    # выгрузка для нейронной сети
    grouped_data.to_csv('csv.csv')

except Exception as e:
    print(e)
