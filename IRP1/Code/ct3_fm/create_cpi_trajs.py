import pandas as pd
import numpy as np


def main():

    # load the data
    print('Loading data...')
    df = pd.read_csv('data/monthly_cpi_netherlands.csv')
    print('Data loaded.')

    print('Preprocessing data...')
    
    traj = df['OBS_VALUE'].pct_change().values[1:]

    print('Data preprocessed.')

    print('Saving data...')
    # save trajectory to npz
    np.savez('data/cpi_trajectory.npz', trajectory=traj)
    print('Data saved.')


if __name__ == '__main__':
    main()