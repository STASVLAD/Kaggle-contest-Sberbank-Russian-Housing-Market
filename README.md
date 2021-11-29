# Kaggle contest "Sberbank Russian Housing Market"

:link: [https://www.kaggle.com/c/sberbank-russian-housing-market/overview](https://www.kaggle.com/c/sberbank-russian-housing-market/overview)

## Data description

In this competition, Sberbank is challenging Kagglers to develop algorithms which use a broad spectrum of features to predict realty prices. Competitors will rely on a rich dataset that includes housing data and macroeconomic patterns. 

The aim of this competition is to predict the sale price of each property. The target variable is called price_doc in train.csv.

The training data is from August 2011 to June 2015, and the test set is from July 2015 to May 2016. The dataset also includes information about overall conditions in Russia's economy and finance sector. 

## Data preprocessing

## Problems and solutions:
:question: Data has a lot missing data

:bulb: We used gradient boosting model XGBRegressor which could handle missing data by itself

:question: Data has some outliers
:bulb: We just dropped such extreme data

:question: Data has some typos
:bulb: We corrected them manually 

:question: There are fake prices due to taxes avoiding
:warning: Such data mostly belong to `product_type == Investment` so we trained two models, one of it shows quite good results and second (which is for `product_type == Investment`) of course performs much worser due to fake prices. There is no real approach to handle such data besides dropping it but the main problem is that test set also contains apartments with fake price. So, we just consider such data as "noise". What is better to train with it, or drop it, or undersample it we will choose later. Now we just choose best parameters for models by applying them on `product_type == OwnerOccupied` split.

* There is no exact location of apartments but we can calculate them using other features
:bulb: We approximate location by using `kremlin_km` feature and district name
:thought_balloon: Later we will try 
* There are some categorical features

##


