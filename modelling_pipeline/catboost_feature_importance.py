def get_catboost_feature_importance(pipe, X, y): 
	# get feature names and feature importances
	feature_names = catboost_pipe.named_steps['classifier'].feature_names_
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
	sns.barplot(x = 'Normalized Prediction Values Change', y = 'Features', data = subset_data)
	plt.title('CatBoost Feature Importance From Rank ' + str(top) + ' to Rank ' + str(bottom))
