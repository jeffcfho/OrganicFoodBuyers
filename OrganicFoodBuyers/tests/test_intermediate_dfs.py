"""
	Testing pre-saved dataframes necessary for model fitting.
"""

import pandas as pd
import pickle

def test_users_order_by_prev_org(filepath_rel='./'):
	'''
		Test that the dataframe users_order_by_prev_org.csv has the right columns.
	'''
	df_filename = filepath_rel+'modeling_dfs/users_order_by_prev_org.csv'
	df_modeling = pd.read_csv(df_filename)

	# list of required columns for final model
	required_cols = ['any_organic_produce','any_organic_produce_prev']
	assert all([col in df_modeling.columns for col in required_cols]), 'Table users_order_by_prev_org does not have required columns.'

if __name__=='__main__':
	test_users_order_by_prev_org(filepath_rel='../../') #for testing in the same folder