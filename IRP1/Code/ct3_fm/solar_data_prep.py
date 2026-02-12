import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def main():

    # load the data
    print('Loading data...')
    df = pd.read_csv('data/Solar_Energy_Generation.csv')
    print('Data loaded.')

    print('Preprocessing data...')
    # filter site
    df1 = df[(df['SiteKey'] == 1) & (df['CampusKey'] == 2)]

    # fill na with 0
    df1 = df1.fillna(0)

    # downsample df1 by factor of 10
    df1_ds = df1.iloc[::10, :]

    # split df1_ds into summer and winter seasons for each year
    df1_ds['Timestamp'] = pd.to_datetime(df1_ds['Timestamp'])
    df1_ds['Month'] = df1_ds['Timestamp'].dt.month

    # data was collected in the southern hemisphere, so summer is December, January, February and winter is June, July, August
    df_winter = df1_ds[df1_ds['Month'].isin([6, 7, 8])]
    df_summer = df1_ds[df1_ds['Month'].isin([12, 1, 2])]

    print('Data preprocessed.')

    print('Saving data...')
    # save winter and summer trajectories to npz
    np.savez('data/solar_summer_winter.npz', summer=df_summer['SolarGeneration'].values, winter=df_winter['SolarGeneration'].values)
    print('Data saved.')


if __name__ == '__main__':
    main()
