# Kaggle contest "Sberbank Russian Housing Market"

:link: [https://www.kaggle.com/c/sberbank-russian-housing-market/overview](https://www.kaggle.com/c/sberbank-russian-housing-market/overview)

## Notebooks
* `report.ipynb` - draft of final report
* `solution.ipynb` - current best solution, **result**: 1472 / 3266 (public: 1472 / 3266)
* `utils.py` - custom functions holder to reduce `report.ipynb` size
* `research/` - folder with notebooks which contains some analysis (not very informative)
* Almost absolutely uninformative files and folders: `old_notebooks/`, `data/`, `submissions/`

## Data description

In this competition, Sberbank is challenging Kagglers to develop algorithms which use a broad spectrum of features to predict realty prices. Competitors will rely on a rich dataset that includes housing data and macroeconomic patterns. 

The aim of this competition is to predict the sale price of each property. The target variable is called price_doc in train.csv.

The training data is from August 2011 to June 2015, and the test set is from July 2015 to May 2016. The dataset also includes information about overall conditions in Russia's economy and finance sector. 

## Data preprocessing
**progress**: 90% done

### :question: Data has a lot of missing data
:bulb: We used gradient boosting model XGBRegressor which could handle missing data by itself

### :question: Data has some outliers
:bulb: We can divide outliers into 3 categories.

1. Numerical outliers: some extreme values which are very unlikely to be in reality (like an enormous number of rooms and etc.). Solution: assigning np.NaN;
2. Logical outliers: for instance, when the maximum floor is less than the current floor. Solution: assigning np.NaN;
3. Price outliers: fake prices and some extreme data. Solution: dropped them;

### :question: Data has some typos
:bulb: We corrected them manually. For bad data we assigned np.NaN, for some typos (like `20152019` for `build_year` feature) we assigned values depending on other features

### :question: There are fake prices due to taxes avoiding
:warning: Such data mostly belong to `product_type == Investment`. There is no real approach to handle such data besides dropping it but the main problem is that the test set also contains apartments with fake prices. So, we just consider such data as "noise". What is better to train with it, or drop it, or undersample it we will choose later. Now we just choose the best parameters for models by applying them only on the part of data with `product_type == OwnerOccupied`.

### :question: There is no exact location of apartments
:bulb: We approximate location by using:
1. `kremlin_km` and `sub_area` polygons
2. centroids of `sub_area` polygons

**Comments:** By name of the district (`sub_area` feature) we downloaded geo-polygons, created a grid of points within it, and found the most appropriate location by comparing the calculated distance to kremlin with the `kremlin_km` feature. As a result, we got a quite good approximation but of course there are some mistakes

:thought_balloon: Later we will try another approach by using distances from the main circle roads of Moscow city which are also presented in the dataset

### :question: There are some categorical features
:bulb: Binary categorical features we encoded as 0,1. Feature `ecology` was encoded 'poor': 1, 'satisfactory': 2, 'good': 3 and etc. Feature `sub_area` was encoded as location coordinates as described before

## Models
**progress**: 80% done

We use XGBRegressor as the main model because it is one of the SOTA models for this kind of regression tasks.

We split the dataset based on the `product_type` feature. We trained XGBRegressor on both of them. One of them shows quite good results, and the second (which is for `product_type == Investment`) of course performs much worse due to fake prices as we described before

:thought_balloon: We will experiment with stacking models. Maybe we will train LightGBM and/or RandomForestRegressor and/or LinearRegression on the whole dataset (or on some split) and stack them with XGBRegressor by using some kind of LinearRegression as a final estimator (StackingRegressor)

## Macroeconomic data
**progress**: 10% done
This is the main problem of this competition. We somehow have to detrend our prices. Dataset `macro.csv` may be helpful in this situation, but we didn't find any good approaches to solve it. Also the large part of participants didn't solve it too, they just found some good coefficients by brute forcing test submits.

:thought_balloon: Further we will try to train our data with some manually estimated coefficients. Something like: impact (weights) for objects from 2012 is less than 2015 since 2015 objects is more relevant for 2016 year.

:thought_balloon: Also, we will try to detect some linear dependence between price and time depending on clusters of similar apartments

:thought_balloon: If we have enough time we will try to analyze prices and macroeconomic data using time series analysis

To sum up, dealing with this problem is our main goal for now 

## Validation
**progress**: 10% done

It's an open question how to train models better, how to perform train_test_split, cross-validation and etc. (there is not so much data). We will deal with these problems on the final part

