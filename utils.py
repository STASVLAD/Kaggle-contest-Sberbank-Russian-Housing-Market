import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from math import radians
from shapely.geometry import Point
from sklearn.metrics.pairwise import haversine_distances


def encode(df):
    # Timestamp encoding
    df['timestamp_year'] = df['timestamp'].dt.year
    df['timestamp_month'] = df['timestamp'].dt.month
    df['timestamp_day'] = df['timestamp'].dt.day
    df.drop(labels='timestamp', axis=1, inplace=True)

    # Categorical columns encoding
    df['product_type'] = df['product_type'].map({'Investment': 0, 'OwnerOccupier': 1})

    # Ecology
    eco_map = {'no data': np.NaN,
               'poor': 1,
               'satisfactory': 2,
               'good': 3,
               'excellent': 4, }
    df['ecology'] = df['ecology'].map(eco_map)

    # yes/no
    cat_columns = df.select_dtypes(include='object').drop(['sub_area'], axis=1).columns
    df[cat_columns] = df[cat_columns].applymap(lambda x: 0 if x == 'no' else 1)

    return df


def create_submission(model, X_test):
    submission = pd.read_csv('data/sample_submission.csv')
    pred = model.predict(X_test)
    if len(pred[pred < 0]):
        print('WARNING: NEGATIVE PREDICTIONS')
        pred = np.abs(pred)
    submission['price_doc'] = pred
    submission.to_csv('submission.csv', index=False)


def get_place(my_score):
    df = pd.read_csv('publicleaderboard.csv')
    scores = df['Score'].values
    scores = np.append(scores, my_score)
    scores = np.sort(scores)
    print(f'{np.where(scores == my_score)[0][0]} / {len(scores)}')


def distance2kremlin(pos):
    kremlin = [55.752121, 37.617664]

    kremlin_in_radians = [radians(_) for _ in kremlin]
    pos_in_radians = [radians(_) for _ in pos]

    result = haversine_distances([kremlin_in_radians, pos_in_radians])[0][1]
    return result * 6371000/1000


def make_grid(geometry, tol=100):

    xmin, xmax, ymin, ymax = 36.665063, 37.955376, 55.113476, 56.046934
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, tol), np.linspace(ymin, ymax, tol))
    xc = xx.flatten()
    yc = yy.flatten()

    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xc, yc)])
    in_map = np.array([pts.within(geom) for geom in geometry]).sum(axis=0)
    pts = gpd.GeoSeries([val for pos, val in enumerate(pts) if in_map[pos]])

    plt.close()

    return pts


def remove_outliers(all_df):
    # Numerical outliers
    all_df.loc[all_df['full_sq'] < 10, 'full_sq'] = np.nan
    all_df.loc[all_df['full_sq'] > 250, 'full_sq'] /= 10
    all_df.loc[all_df['full_sq'] > 1000, 'full_sq'] /= 100  # >1500 : np.nan

    all_df.loc[all_df['life_sq'] < 7, 'life_sq'] = np.nan  # 5
    all_df.loc[all_df['life_sq'] > 500, 'life_sq'] = np.nan

    all_df.loc[all_df['floor'] == 0, 'floor'] = np.nan
    all_df.loc[all_df['floor'] == 77, 'floor'] = np.nan

    all_df.loc[all_df['max_floor'] == 0, 'max_floor'] = np.nan
    all_df.loc[all_df['max_floor'] > 57, 'max_floor'] = np.nan

    all_df.loc[all_df['build_year'] == 2, 'build_year'] = 2014
    all_df.loc[all_df['build_year'] == 20, 'build_year'] = 2014
    all_df.loc[all_df['build_year'] == 215, 'build_year'] = 2015
    all_df.loc[all_df['build_year'] == 1691, 'build_year'] = 1961
    all_df.loc[all_df['build_year'] == 4965, 'build_year'] = 1965
    all_df.loc[all_df['build_year'] == 20052009, 'build_year'] = 2009
    all_df.loc[all_df['build_year'] < 1850, 'build_year'] = np.nan
    all_df.loc[all_df['build_year'] > 2020, 'build_year'] = np.nan

    all_df.loc[all_df['num_room'] == 0, 'num_room'] = np.nan
    all_df.loc[all_df['num_room'] > 15, 'num_room'] = np.nan

    all_df.loc[all_df['kitch_sq'] > 500, 'kitch_sq'] = np.nan
    all_df.loc[all_df['kitch_sq'] < 5, 'kitch_sq'] = np.nan  # 2

    all_df.loc[all_df['state'] > 30, 'state'] = np.nan

    # Logical outliers
    all_df.loc[all_df['life_sq'] > all_df['full_sq'], 'life_sq'] = np.nan
    all_df.loc[all_df['floor'] > all_df['max_floor'], 'max_floor'] = np.nan
    all_df.loc[all_df['kitch_sq'] >= all_df['life_sq'], 'kitch_sq'] = np.nan

    # Price outliers
    idx_outliers_high = all_df[all_df['price_doc'] / all_df['full_sq'] > 350000].index
    idx_outliers_low = all_df[all_df['price_doc'] / all_df['full_sq'] < 10000].index  # !!!
    idx_outliers = idx_outliers_low.append(idx_outliers_high)

    all_df.drop(idx_outliers, axis=0, inplace=True)

    return all_df
