"""
	Utility functions  
"""

import pandas as pd
import numpy as np

def create_modeling_df():
	"""
		Loads previously saved dataframes and joins them together to set up df for modeling
		(Current best model contains past organic produce purchase + Percent of top 20 recommendations that are organic)
		Returns a Pandas dataframe
	"""
	# Read in base df about users
	df_modeling = pd.read_csv('../modeling_dfs/users_order_by_prev_dept.csv')

	# Join a dataframe with info on past organic_produce purchase
	df_org_produce = pd.read_csv('../modeling_dfs/users_order_by_prev_org.csv')
	df_org_produce.drop(['Unnamed: 0'],axis=1,inplace=True)
	df_modeling = df_modeling.merge(df_org_produce[['order_id','organic_produce','any_organic_produce',
	                              					'organic_produce_prev','any_organic_produce_prev']],on='order_id')

	# Join a feature with percentage of top recommended products that are organic
	# Load top 20 recommended products
	df_recs = pd.read_csv('top20_products_recom.csv')
	df_recs.drop(['Unnamed: 0'],axis=1,inplace=True)

	# Group by organic column to get the sum of organic items in top 20 recs
	user_pct_organic = df_recs[['user_id','organic']].groupby('user_id').sum()\
	                        .rename({'organic':'pct_organic'},axis=1)/20
	user_pct_organic.reset_index(inplace=True)
	df_modeling = df_modeling.merge(user_pct_organic,on='user_id')
	
	return df_modeling