"""
	Fits and saves a collaborative filtering model based on Instacart data.
"""

import pandas as pd 
import numpy as np 
from scipy.sparse import coo_matrix # for constructing sparse matrix
from scipy.sparse import save_npz
# lightfm 
from lightfm import LightFM # model
from lightfm.evaluation import auc_score,precision_at_k,recall_at_k
import pickle

def load_data(filepath):
	"""
		Loads Instacart data from csv files.
		Returns 4 dataframes.
	"""
	os = pd.read_csv(filepath+'orders.csv')
	o_ps__prior = pd.read_csv(filepath+'order_products__prior.csv')
	o_ps__train = pd.read_csv(filepath+'order_products__train.csv')
	ps = pd.read_csv(filepath+'products.csv')
	return os, ps, o_ps__prior, o_ps__train

def create_user_product_freq_table(os,o_ps__prior):
	"""
		Manipulates orders and orders_products table to create user-product frequency table.
		Returns a pandas dataframe where each row contains a user-product pair and the frequency of that pair.
	"""
	# Get prior orders by user
	users_order_prior = os.loc[os['eval_set'].map(lambda x: x in ['prior'])]

	# Merge products in prior orders to the user for each order
	user_order_products_prior_train = o_ps__prior.merge(users_order_prior[['order_id','user_id']],
                                                        on='order_id')

	# Count frequency of each product bought by each user. These will be the "ratings" for lightfm
	user_products_freq_train_long = user_order_products_prior_train[['user_id','product_id']]\
	                                .groupby(['user_id','product_id']).size()

	# Reset index to remove MultiIndex
	user_products_freq_train_long = user_products_freq_train_long.reset_index()
	user_products_freq_train_long.columns = ['user_id','product_id','freq']
	return user_products_freq_train_long

def create_user_product_freq_matrix(user_products_freq_table_long):
	"""
		Turns long user/product frequency table into a matrix
		Returns a scipy sparse matrix.
	"""

	# This fails so try to do it using coo_matrix() below
	# user_products_freq_train_wide = user_products_freq_table_long.unstack(fill_value=0)

	# Row index will be user_id -1 (because no row_id=0)
	row_ind = user_products_freq_table_long['user_id'].values-1
	num_rows = max(user_products_freq_table_long['user_id'].unique())

	# Column index will be prod_id -1 (because no prod_id=0)
	col_ind = user_products_freq_table_long['product_id'].values-1
	num_cols = max(user_products_freq_table_long['product_id'].unique())

	# Frequency data are cell values
	freq = user_products_freq_table_long['freq'].values

	# Create sparse matrix
	user_prod_interaction_train = coo_matrix((freq, (row_ind, col_ind)), shape = (num_rows, num_cols))

	return user_prod_interaction_train


def fit_colab_filter_model(user_prod_matrix):
	"""
		Fits simple colab filter model with LightFM using default parameters
		Returns model
	"""
	m = LightFM(loss = "warp")
	m.fit(user_prod_matrix,epochs=1)
	return m

def save_colab_filter_model(m,model_filename):
	"""
		Save model file
	"""
	with open(model_filename, 'wb') as fle:
	    pickle.dump(m, fle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
	path = '/content/drive/My Drive/Insight/'
	orders, products, orders_products__prior, order_products__train = load_data(path)

	# Data pre-processing
	user_products_freq_long = create_user_product_freq(orders,orders_products__prior)
	user_prod_interaction_mat = create_user_product_freq_matrix(user_products_freq_long)
	save_npz('user_prod_interaction',user_prod_interaction_mat)

	# Fit model and save
	model = fit_colab_filter_model(user_prod_interaction_mat)
	save_colab_filter_model(model,'model_lightfm_v2.pkl')


if __name__ == '__main__':
	main()