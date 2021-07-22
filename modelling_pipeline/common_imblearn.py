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
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV
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
			('sampler', RandomUnderSampler(random_state = 1)),
			('classifier', clfs[i])])

		scores = cross_validate(pipe, X_train, y_train, scoring = ('accuracy', 'recall', 'f1'), cv = StratifiedKFold(cv_folds))

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

"""
the above method will need to do sampling every time you run the undersampler pipeline 
with a classifier, and can be less efficient if your dataset is super huge...

the following method will do stratified sampling once and loop each pipeline, 
which is more time efficient and flexible(you can use whatever pipeline you want)
"""

def compare_models_cv(names, pipes, X, y, nfolds):
	"""
	less time consuming and more flexible than the above
	"""

	# run stratified K fold CV for all pipelines
	skf = StratifiedKFold(n_splits = nfolds)
	test_accuracy = {}
	test_recall = {}
	test_f1 = {}

	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]

		for i in range(len(pipes)):
			# model fitting and prediction
			pipe = pipes[i]
			pipe.fit(X_train, y_train)
			y_pred = pipe.predict(X_test)

			# save scores to dictionary
			if names[i] in test_accuracy: 
				test_accuracy[names[i]].append(accuracy_score(y_test, y_pred))
				test_recall[names[i]].append(recall_score(y_test, y_pred))
				test_f1[names[i]].append(f1_score(y_test, y_pred))

			else:
				test_accuracy[names[i]] = [accuracy_score(y_test, y_pred)]
				test_recall[names[i]] = [recall_score(y_test, y_pred)]
				test_f1[names[i]] = [f1_score(y_test, y_pred)]

	# retrieve result from dictionary
	result_accuracy = []
	result_recall = []
	result_f1 = []

	for key in test_accuracy:
		result_accuracy.append(round(np.mean(test_accuracy[key]), 5))

	for key in test_recall:
		result_recall.append(round(np.mean(test_recall[key]), 5))

	for key in test_f1:
		result_f1.append(round(np.mean(test_f1[key]), 5))

	# organize into a pandas df
	result_table = zip(names, result_accuracy, result_recall, result_f1)
	result = pd.DataFrame(list(result_table), columns = ['Classifiers', 'CV Accuracy', 'CV Recall', 'CV F1'])

	return result

def main_model_cv():
	preprocessor = create_preprocessor(X_train)

	lr_pipe = Pipeline([
		('preprocessor', preprocessor), 
		('sampler', RandomUnderSampler()),
		('lr', LogisticRegression(solver = 'liblinear', max_iter = 1000))])

	xgb_pipe = Pipeline([
		('preprocessor', preprocessor), 
		('sampler', RandomUnderSampler()),
		('xgbclassifier', xgb.XGBClassifier())])

	names = ['Logistic Regression', 'XGBoost']
	pipes = [lr_pipe, xgb_pipe]

	compare_models_cv(names, pipes, X_train, y_train, nfolds = 5)

def holdout_estimate(pipe, X_train, y_train, X_holdout, y_holdout, name = None):
	pipe.fit(X_train, y_train)
	y_pred = pipe.predict(X_holdout)

	holdout_accuracy = round(accuracy_score(y_holdout, y_pred), 5)
	holdout_recall = round(recall_score(y_holdout, y_pred), 5)
	holdout_f1 = round(f1_score(y_holdout, y_pred), 5)

	print(name + ' Holdout Accuracy: ' + str(holdout_accuracy))
	print(name + ' Holdout Recall: ' + str(holdout_recall))
	print(name + ' Holdout F1: ' + str(holdout_f1))

	return holdout_accuracy, holdout_recall, holdout_f1