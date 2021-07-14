import pandas as pd
import numpy as np 

# visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

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

def get_catboost_feature_importance(catboost_pipe, X, y): 
	catboost_pipe.fit(X, y)

	# get feature names and feature importances
	num_cols = X.columns[X.dtypes != 'object'].tolist()
	cat_cols = X.columns[X.dtypes == 'object'].tolist()
	ohe = (catboost_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'])
	ohe_names = ohe.get_feature_names(input_features = cat_cols)
	feature_names = np.r_[num_cols, ohe_names]
	feature_importances = catboost_pipe.named_steps['classifier'].feature_importances_

	# organize into a table
	table = zip(feature_names, feature_importances)
	result = pd.DataFrame(list(table), columns = ['cols', 'normalized_predictval_change']).sort_values(by = 'normalized_predictval_change', ascending = False)
	result['cumulative_feature_importance'] = result['normalized_predictval_change'].cumsum()

	result = result.reset_index()

	return result

def visualize_feature_importance(importance_table, top, bottom):
	# subset data
	subset_data = importance_table.loc[top - 1:bottom - 1, ['cols', 'normalized_predictval_change']]

	# visualization
	sns.barplot(x = 'normalized_predictval_change', y = 'cols', data = subset_data)
	plt.title('CatBoost Feature Importance From Rank ' + str(top) + ' to Rank ' + str(bottom))

def main():
	catboost_pipe = Pipeline([
		('preprocessor', preprocessor), 
		('classifier', CatBoostClassifier(verbose = False, scale_pos_weight = 3689/8312))])

	importance_table = get_catboost_feature_importance(pipe, X, y)
	visualize_feature_importance(importance_table, 1, 20)