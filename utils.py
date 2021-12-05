import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from math import radians
from shapely.geometry import Point
from sklearn.metrics.pairwise import haversine_distances


def encode(df):
    # Timestamp encoding
    df['timestamp_day'] = df['timestamp'].dt.day
    df['timestamp_month'] = df['timestamp'].dt.month
    df['timestamp_year'] = df['timestamp'].dt.year

    df['timestamp_weekday'] = df['timestamp'].dt.weekday
    df['timestamp_quarter'] = df['timestamp'].dt.quarter

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
    submission = pd.read_csv('data/submits/sample_submission.csv')
    pred = model.predict(X_test)
    if len(pred[pred < 0]):
        print('WARNING: NEGATIVE PREDICTIONS')
        pred = np.abs(pred)
    submission['price_doc'] = pred
    submission.to_csv('data/submits/submission.csv', index=False)


def get_place(my_score):
    df = pd.read_csv('data/submits/publicleaderboard.csv')
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
    all_df.loc[all_df['full_sq'] > 1000, 'full_sq'] /= 100  # np.nan or /= 100 (1)

    all_df.loc[(all_df["full_sq"] > 250) &
               (all_df["life_sq"] / all_df["full_sq"] < 0.3), 'full_sq'] /= 10  # np.nan or /= 10

    all_df.loc[(all_df["full_sq"] > 210) &
               (all_df["price_doc"] < 10_000_000) &
               (all_df["life_sq"] > 100), 'life_sq'] /= 10  # np.nan or /= 10
    all_df.loc[(all_df["full_sq"] > 210) & (all_df["price_doc"] < 10_000_000), 'full_sq'] /= 10  # np.nan or /= 10

    all_df.loc[all_df['life_sq'] < 5, 'life_sq'] = np.nan
    all_df.loc[all_df['life_sq'] > 1000, 'life_sq'] /= 100  # 74/78 or /= 100
    all_df.loc[(all_df['life_sq'] > 300) &
               (all_df['life_sq'] == all_df['full_sq']*10), 'life_sq'] /= 10
    all_df.loc[all_df['life_sq'] > 300, 'life_sq'] /= 10  # np.nan or /= 10

    all_df.loc[13120, "build_year"] = all_df.loc[13120, "kitch_sq"]
    all_df.loc[all_df['kitch_sq'] > 200, 'kitch_sq'] = np.nan
    all_df.loc[all_df['kitch_sq'] < 2, 'kitch_sq'] = np.nan

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

    all_df.loc[all_df['state'] > 30, 'state'] = np.nan

    all_df.loc[all_df['preschool_quota'] == 0, 'preschool_quota'] = np.nan

    # Logical outliers
    all_df.loc[all_df['life_sq'] > all_df['full_sq'], 'life_sq'] = np.nan
    all_df.loc[all_df['floor'] > all_df['max_floor'], 'max_floor'] = np.nan
    all_df.loc[all_df['kitch_sq'] >= all_df['life_sq'], 'kitch_sq'] = np.nan

    return all_df


def remove_fake_prices(df, price_sqm_l=10000, price_sqm_h=600000):
    # Price outliers
    idx_outliers_high = df[df['price_doc'] / df['full_sq'] > price_sqm_h].index
    idx_outliers_low = df[df['price_doc'] / df['full_sq'] < price_sqm_l].index
    idx_outliers = idx_outliers_low.append(idx_outliers_high)

    df = df.drop(idx_outliers, axis=0)
    print('REMOVED:', len(idx_outliers))

    # idx_1M = df.loc[(df['product_type'] == 0) & (df['price_doc'] == 1_000_000)].index.values
    # idx_2M = df.loc[(df['product_type'] == 0) & (df['price_doc'] == 2_000_000)].index.values
    # idx_3M = df.loc[(df['product_type'] == 0) & (df['price_doc'] == 3_000_000)].index.values
    # idx_1M_usampled = np.random.choice(idx_1M, size=int(M1_ratio * len(idx_1M)), replace=False)
    # idx_2M_usampled = np.random.choice(idx_2M, size=int(M2_ratio * len(idx_2M)), replace=False)
    # idx_3M_usampled = np.random.choice(idx_3M, size=int(M3_ratio * len(idx_3M)), replace=False)

    # df = df.drop(idx_1M_usampled, axis=0)
    # df = df.drop(idx_2M_usampled, axis=0)
    # df = df.drop(idx_3M_usampled, axis=0)

    return df


def tverskoe_issue_fix(df):
    fix_df = pd.read_excel('data/BAD_ADDRESS_FIX.xlsx').drop_duplicates('id').set_index('id')
    df.update(fix_df, overwrite=True)
    print('Fix: ', df.index.intersection(fix_df.index).shape[0])


def create_new_features(all_df):
    all_df['floor_by_max_floor'] = all_df['floor'] / all_df['max_floor']
    all_df["extra_sq"] = all_df["full_sq"] - all_df["life_sq"]

    # Room
    all_df['avg_room_size'] = (all_df['life_sq'] - all_df['kitch_sq']) / all_df['num_room']
    all_df['life_sq_prop'] = all_df['life_sq'] / all_df['full_sq']
    all_df['kitch_sq_prop'] = all_df['kitch_sq'] / all_df['full_sq']

    # Calculate age of building
    all_df['build_age'] = all_df['timestamp_year'] - all_df['build_year']
    all_df = all_df.drop('build_year', axis=1)

    # Population
    all_df['population_den'] = all_df['raion_popul'] / all_df['area_m']
    all_df['gender_rate'] = all_df['male_f'] / all_df['female_f']
    all_df['working_rate'] = all_df['work_all'] / all_df['full_all']

    # Education
    all_df['preschool_ratio'] = all_df['children_preschool'] / all_df['preschool_quota']
    all_df['school_ratio'] = all_df['children_school'] / all_df['school_quota']

    # NaNs count
    all_df['nan_count'] = all_df[['full_sq', 'build_age', 'life_sq', 'floor', 'max_floor', 'num_room']].isnull().sum(axis=1)

    # all_df = all_df.drop('timestamp_year', axis=1)
    return all_df
