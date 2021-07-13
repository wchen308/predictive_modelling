# data manipulation
import pandas as pd
import numpy as np

# preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

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
			missing_values = None, 
			strategy = 'constant', 
			fill_value = 'missing')), 
		('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

	preprocessor = ColumnTransformer([
		('num', num_pipe, num_cols), 
		('cat', cat_pipe, cat_cols)], remainder = 'passthrough')

	return preprocessor

def undersampler_model_comparison(name, clfs, preprocessor, X_train, y_train, X_holdout, y_holdout, cv_folds):
	cv_accuracy = []
	cv_recall = []
	cv_f1 = []
	holdout_accuracy = []
	holdout_recall = []
	holdout_f1 = []

	for i in range(len(name)):
		pipe = Pipeline([ 
			('preprocessor', preprocessor),    # you may want to put sampler at the top if too big to fit final full data
			('sampler', RandomUnderSampler()),
			('classifier', clfs[i])])

		scores = cross_validate(pipe, X_train, y_train, scoring = ('accuracy', 'recall', 'f1'), cv = cv_folds)

		cv_accuracy.append(round(scores['test_accuracy'].mean(), 5))
		cv_recall.append(round(scores['test_recall'].mean(), 5))
		cv_f1.append(round(scores['test_f1'].mean(), 5))

		# unbiased estimate
		pipe.fit(X_train, y_train)
		y_pred = pipe.predict(X_holdout)
		holdout_accuracy.append(round(accuracy_score(y_holdout, y_pred), 5))
		holdout_recall.append(round(recall_score(y_holdout, y_pred), 5))
		holdout_f1.append(round(f1_score(y_holdout, y_pred), 5))

	result_table = zip(name, cv_accuracy, cv_recall, cv_f1, 
		holdout_accuracy, holdout_recall, holdout_f1)

	result = pd.DataFrame(list(result_table), columns = ['Classifier', 'CV Accuracy', 'CV Recall', 'CV F1', 
		'Holdout Accuracy', 'Holdout Recall', 'Holdout F1'])

	return result

def main_model_comparison():
	name = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']
	clfs = [LogisticRegression(solver = 'liblinear', max_iter = 1000), 
		RandomForestClassifier(), 
		xgb.XGBClassifier(), 
		LGBMClassifier(silent = False), 
		CatBoostClassifier(verbose = False)]
	preprocessor = create_preprocessor(X)

	undersampler_model_comparison(name, clfs, preprocessor, X_train, y_train, X_holdout, y_holdout, cv_folds = 10)