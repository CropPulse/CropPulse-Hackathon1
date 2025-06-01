import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')


plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df['target'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title('Distribution of Target', fontsize=14)
plt.show()

print("All Features in Train data with NaN Values =", str(train_df.columns[train_df.isnull().sum() != 0].size) )
# print("All Features in Test data with NaN Values =", str(test_df.columns[train_df.isnull().sum() != 0].size) )

const_columns_to_remove = []
for col in train_df.columns:
    if col != 'ID' and col != 'target':
        if train_df[col].std() == 0:
            const_columns_to_remove.append(col)

# Now remove that array of const columns from the data
train_df.drop(const_columns_to_remove, axis=1, inplace=True)
test_df.drop(const_columns_to_remove, axis=1, inplace=True)

# Print to see the reduction of columns
print('train_df rows and columns after removing constant columns: ', train_df.shape)

print('Following `{}` Constant Column\n are removed'.format(len(const_columns_to_remove)))
print(const_columns_to_remove)



def print_memory_usage_of_df(df):
    bytes_per_mb = 0.000001
    memory_usage = round(df.memory_usage().sum() * bytes_per_mb, 3)
    print('Memory usage is ', str(memory_usage) + " MB")

print_memory_usage_of_df(train_df)
print(train_df.shape)



dummy_encoded_train_df = pd.get_dummies(train_df)
dummy_encoded_train_df.shape

def convert_df_to_sparse_array(df, exclude_columns=[]):
    df = df.copy()
    exclude_columns = set(exclude_columns)

    for (column_name, column_data) in df.iteritems():
        if column_name in exclude_columns:
            continue
        df[column_name] = pd.SparseArray(column_data.values, dtype='uint8')

    return df


def drop_sparse_from_train_test(train, test):
    column_list_to_drop_data_from = [x for x in train.columns if not x in ['ID','target']]
    for f in column_list_to_drop_data_from:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test

train_df, test_df = drop_sparse_from_train_test(train_df, test_df)

# Split data into Train and Test for Model Training

X_train = train_df.drop(['ID', 'target'], axis=1)

y_train = np.log1p(train_df['target'].values)

X_test_original = test_df.drop('ID', axis=1)

X_train_split, X_validation, y_train_split, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# LightGBM Model Training
# Fundamentals of LightGBM Model

# It is a gradient boosting model that makes use of tree based learning algorithms. It is considered to be a fast processing algorithm.

# While other algorithms trees grow horizontally, LightGBM algorithm grows vertically, meaning it grows leaf-wise and other algorithms grow level-wise. LightGBM chooses the leaf with large loss to grow. It can lower down more loss than a level wise algorithm when growing the same leaf.

def light_gbm_model_run(train_x, train_y, validation_x, validation_y, test_x):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 100,
        "learning_rate" : 0.001,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    # Given its a regression case, I am using the RMSE as the metric.

    lg_train = lgb.Dataset(train_x, label=train_y)
    lg_validation = lgb.Dataset(validation_x, label=validation_y)
    evals_result_lgbm = {}

    model_light_gbm = lgb.train(params, lg_train, 5000,
                      valid_sets=[lg_train, lg_validation],
                      early_stopping_rounds=100,
                      verbose_eval=150,
                      evals_result=evals_result_lgbm )

    pred_test_light_gbm = np.expm1(model_light_gbm.predict(test_x, num_iteration=model_light_gbm.best_iteration ))

    return pred_test_light_gbm, model_light_gbm, evals_result_lgbm

# Training and output of LightGBM Model
predictions_test_y_light_gbm, model_lgbm, evals_result = light_gbm_model_run(X_train_split, y_train_split, X_validation, y_validation, X_test_original)
print('Output of LightGBM Model training..')
