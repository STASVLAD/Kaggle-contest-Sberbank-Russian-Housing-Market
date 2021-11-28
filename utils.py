import folium
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

    # Sub_area
    # one_hot = pd.get_dummies(df['sub_area'])
    # df.drop('sub_area', axis=1, inplace=True)
    # df = df.join(one_hot)

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
