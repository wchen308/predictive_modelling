import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
%matplotlib inline

def calculate_gain_table(info_table):
	"""
	info table must have 'proba' and 'target' column and in pandas df

	1. sort predicted probability in decreasing order, separate into 10 deciles
	2. group by each decile, calculate gain score = sum(target = 1) / total(target = 1)
	3. gain = cumsum(gain score)
	4. plot x = decile, y = gain, text attonation
	"""

	# sort and separate
	info_table['decile'] = 10 - pd.qcut(info_table['proba'], 10, labels = False)

	# group by each decile, calculate gain score
	total_positive = sum(info_table['target'])
	gain_table = info_table.groupby('decile', as_index = False)['target'].sum()
	gain_table = gain_table.rename(columns = {'target': 'positive'})
	gain_table['gain_score'] = gain_table['positive'] / total_positive

	# gain = cumsum(gain score)
	gain_table['gain'] = gain_table['gain_score'].cumsum()

	return gain_table

def plot_gain_chart(decile, gain1, gain2):
	# plot
	plt.plot(decile, gain1, color = 'red', label = 'Model1 Gain')
	plt.plot(decile, gain2, color = 'green', label = 'Model2 Gain')
	plt.legend()

	# label
	plt.xlabel("Percentage of Population Ranked By Predicted Probability")
	plt.ylabel("Cumulative Detected Positive")
	plt.title("Gain Chart")