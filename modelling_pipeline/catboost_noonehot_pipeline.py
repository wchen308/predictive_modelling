# reference
https://medium.com/analytics-vidhya/combining-scikit-learn-pipelines-with-catboost-and-dask-part-2-9240242966a7

from catboost import CatBoostClassifier

class CustomCatBoostClassifier(CatBoostClassifier):
	def fit(self, X, y = None, **fit_params):
		print('categorical variables: ')
		print(X.filter(regex = '_cat').columns.tolist())

		return super().fit(X, y = y, cat_features = X.filter(regex = '_cat').columns.tolist(), **fit_params)

class CustomFeatureSelection(SelectFromModel):
	def transform(self, X):
		important_features_indices = list(self.get_support(indices = True))
		selected_X = X.iloc[:, important_features_indices].copy()

		return selected_X

def tune_model(pipe, param_grid, X, y, iterations, cv_folds, name):
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

def main():
	feature_selection_pipe = Pipeline([
		('feature_selection', CustomFeatureSelection(CustomCatBoostClassifier(verbose = False), 
			threshold = -np.inf)), 
		('classifier', CustomCatBoostClassifier(verboes = False))])

	param_grid = {'feature_selection__max_features': [20, 30, 40]}
	cv_folds = StratifiedKFold(5)
	name = 'CatBoost Feature Selection'
	iterations = 3

	tune_model(feature_selection_pipe, param_grid, X, y, iterations, cv_folds, name)