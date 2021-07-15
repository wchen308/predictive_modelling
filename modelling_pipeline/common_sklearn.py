# data manipulation
import pandas as pd
import numpy as np

# preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# modelling
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def create_preprocessor(X):
	# numerical columns
	num_cols = X.columns[X.dtypes != 'object'].tolist()
	num_pipe = Pipeline([
		('scaler', StandardScaler()), 
		('imputer', SimpleImputer(strategy = 'median'))])

	# categorical columns
	cat_cols = X.columns[X.dtypes == 'object'].tolist()
	cat_pipe = Pipeline([
		('imputer', SimpleImputer(
			missing_values = None,    # may set to default value if does not work
			strategy = 'constant', 
			fill_value = 'missing')), 
		('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

	preprocessor = ColumnTransformer([
		('num', num_pipe, num_cols), 
		('cat', cat_pipe, cat_cols)], remainder = 'passthrough')

	return preprocessor

#<----------------------------------------->
"""
model comparison
"""
def model_comparison(name, clfs, preprocessor, X_train, y_train, X_holdout, y_holdout, cv_folds):
	cv_auc = []
	holdout_auc = []

	for i in range(len(name)):
		pipe = Pipeline([
			('preprocessor', preprocessor), 
			('classifier', clfs[i])])

		scores = cross_validate(pipe, X_train, y_train, scoring = 'roc_auc', cv = cv_folds)
		cv_auc.append(round(scores['test_score'].mean(), 3))

		# unbiased estimate
		pipe.fit(X_train, y_train)
		y_pred = pipe.predict(X_holdout)
		holdout_auc.append(round(roc_auc_score(y_holdout, y_pred), 3))

	result_table = zip(name, cv_auc, holdout_auc)

	result = pd.DataFrame(list(result_table), columns = ['Classifier', 'CV AUC', 'Holdout AUC'])

	return result

def main_model_comparison():
	name = ['Logistic Regression', 'XGBoost']
	clfs = [LogisticRegression(solver = 'liblinear', max_iter = 1000), 
		xgb.XGBClassifier()]
	preprocessor = create_preprocessor(X)

	model_comparison(name, clfs, preprocessor, X_train, y_train, X_holdout, y_holdout, cv_folds = 5)

#<----------------------------------------->
"""
model tuning
"""
def tune_model(pipe, param_grid, X_train, y_train, iterations, cv_folds, name = None):
	search = RandomizedSearchCV(
		pipe, 
		param_distributions = param_grid, 
		n_iter = iterations,
		scoring = 'roc_auc',
		cv = cv_folds)

	search.fit(X_train, y_train)
	result = pd.DataFrame(search.cv_results_)

	print(name + ' Best Set of Hyperparameter is: ')
	print(search.best_params_)
	print(name + ' Best CV AUC Score is: ')
	print(round(search.best_score_, 3))

	return result

def main_tune_LR():
	preprocessor = create_preprocessor(X)

	lr_pipe = Pipeline([
		('preprocessor', preprocessor), 
		('feature_selection', SelectKBest()), 
		('lr', LogisticRegression(solver = 'liblinear', max_iter = 1000, penalty = 'l1'))])

	param_grid = {
	'feature_selection__k': [10, 20, 30],
	'lr__C': [0.33, 0.66, 1]
	}

	tune_model(lr_pipe, param_grid, X_train, y_train, iterations = 9, cv_folds = 5, 
		name = 'LR with Feature Selection and L1 Regularization')

def main_tune_xgb():
	preprocessor = create_preprocessor(X)

	xgb_pipe = Pipeline([
		('preprocessor', preprocessor), 
		('feature_selection', SelectKBest()), 
		('xgbclassifier', xgb.XGBClassifier())])

    param_grid = {
    'feature_selection__k': [10, 20, 30], 
	'xgbclassifier__max_depth': [4, 6, 8], 
	'xgbclassifier__min_child_weight': [1, 10, 25], 
	'xgbclassifier__subsample': [0.5, 0.8, 1], 
	'xgbclassifier__colsample_bytree': [0.5, 0.8, 1], 
	'xgbclassifier__eta': [0.1, 0.3, 0.5]
	}

	tune_model(xgb_pipe, param_grid, X_train, y_train, iterations = 100, cv_folds = 5, 
		name = 'XGBoost with Feature Selection')

#<----------------------------------------->
"""
holdout estimate
"""

def holdout_estimate(pipe, X_train, y_train, X_holdout, y_holdout, name = None):
	pipe.fit(X_train, y_train)
	y_pred = pipe.predict(X_holdout)

	print(name + ' holdout AUC: ' + str(round(roc_auc_score(y_holdout, y_pred), 3)))
