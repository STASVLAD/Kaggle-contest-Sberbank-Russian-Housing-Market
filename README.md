# Kaggle contest "Sberbank Russian Housing Market"

:link: [https://www.kaggle.com/c/sberbank-russian-housing-market/overview](https://www.kaggle.com/c/sberbank-russian-housing-market/overview)

## Data description

In this competition, Sberbank is challenging Kagglers to develop algorithms which use a broad spectrum of features to predict realty prices. Competitors will rely on a rich dataset that includes housing data and macroeconomic patterns. 

The aim of this competition is to predict the sale price of each property. The target variable is called price_doc in train.csv.

The training data is from August 2011 to June 2015, and the test set is from July 2015 to May 2016. The dataset also includes information about overall conditions in Russia's economy and finance sector. 

## Data preprocessing
**progress**: 80% done

### :question: Data has a lot missing data
:bulb: We used gradient boosting model XGBRegressor which could handle missing data by itself

### :question: Data has some outliers
:bulb: We just dropped such extreme data by analyzing ranges and logic of main features
### :question: Data has some typos
:bulb: We corrected them manually. For bad data we assigned np.NaN, for some typos (like `20152019` for `build_year` feature) we assigned logically

### :question: There are fake prices due to taxes avoiding
:warning: Such data mostly belong to `product_type == Investment`. There is no real approach to handle such data besides dropping it but the main problem is that the test set also contains apartments with fake prices. So, we just consider such data as "noise". What is better to train with it, or drop it, or undersample it we will choose later. Now we just choose the best parameters for models by applying them only on the part of data with `product_type == OwnerOccupied`.

### :question: There is no exact location of apartments
:bulb: We approximate location by using `kremlin_km` and `sub_area` features

**Comments:** By name of the district (`sub_area` feature) we downloaded geo-polygons, created a grid of points within it, and found the most appropriate location by comparing the calculated distance to kremlin with `kremlin_km` feature. As a result, we got a quite good approximation but of course there are some mistakes

:thought_balloon: Later we will try another approach by using distances from the main circle roads of Moscow city which are also presented in the dataset

### :question: There are some categorical features
:bulb: Binary categorical features we encoded as 0,1. Feature `ecology` was encoded bad: 1, good:1, perfect: 2 and etc. Feature `sub_area` was encoded as location coordinates as described before

## Models
progress: 70% done

We use XGBRegressor as the main one because it is one of the SOTA models for this kind of regression tasks.

We split dataset based on `product_type` feature as described before. We trained XGBRegressor on both of them. So we trained two models, one of them shows quite good results, and the second (which is for `product_type == Investment`) of course performs much worse due to fake prices.

:thought_balloon: We will experiment with stacking models. Maybe we will train LightGBM and/or RandomForestRegressor and/or LinearRegression on the whole dataset (or on some split) and stack them with XGBRegressor by using some kind of LinearRegression as a final estimator

## Macroeconomic data
This is the main problem of this competition. We somehow have to detrend our prices. Dataset `macro.csv` should be helpful in this situation, but we didn't find any good approaches to solve it. Also the large part of participants didn't solve it too, they just found some good coefficients by test submits.

:thought_balloon: Further we will try to train our data with some manually estimated coefficients. Something like impact (weights) for objects from 2012 is less than 2015 since 2015 objects is more relevant for 2016 year. 

:thought_balloon: To sum up, dealing with problem will be oue main goal 

## Validation

It's an open question how to train models better, how to perform train_test_split, cross-validation and etc. (there is not so much data). We will deal with these problems on the final part

