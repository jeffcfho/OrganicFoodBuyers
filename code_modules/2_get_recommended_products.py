"""
	Given a collaborative filtering model based on Instacart data, generates top 20 recommended products by user.
"""

import pandas as pd 
import scipy
from lightfm import LightFM
import pickle

def load_cb_model_and_user_product_mat(filepath):
	"""
		Loads previously saved collaborative filter model and user-product matrix
		Returns a LightFM model and a scipy sparse matrix
	"""
	cb_model = pickle.load(open(filepath+'model_lightfm_v2.pkl', 'rb'))
	user_prod_mat = scipy.sparse.load_npz(filepath+'user_prod_interaction.npz')
	return cb_model, user_prod_mat

def load_user_df(filepath):
	"""
		Load a table where each row is a user in the cb model.
		Returns a Pandas DF.
	"""
	df  = pd.read_csv(filepath+'users_order_by_prev_dept.csv')
	df.drop(['Unnamed: 0'],axis=1,inplace=True)
	return df

def load_product_list(filepath):
	"""
		Load a table where each row is a product that the cb model was built on.
		Returns a Pandas DF.
	"""
	# prods = pd.read_csv(path+'products.csv')

	# Currently loading only the top 200 products rather than all ~50000
	prods = pd.read_csv(filepath+'top200_products.csv')
	prods.drop(['Unnamed: 0'],axis=1,inplace=True)
	prods['freq_rank'] = np.arange(1,201,1)

	return prods

def generate_recommendations(users,prods,recs_to_save=20,num_top_prods_to_rec=100):
	"""
		Use LightFM model.predict() to perform cosine similarity of users with top NUM_TOP_PRODS_TO_REC products.
		Returns a Pandas DF with top RECS_TO_SAVE recommended products for each user
	"""
	recom_products_master = pd.DataFrame(None,columns=\
	  ['product_id','product_name','organic','freq_rank','user_id','rec_rank'])

	#Loop through all users and get recommendations:
	# TO-DO: Re-factor by passing in the right-sized np arrays into lightfm.predict() rather than in a loop
	for user_id_to_get_rec in users['user_id']:
		user_id_single = [user_id_to_get_rec]
		row_ixs = [user_id-1 for user_id in user_id_single] #subtract one to get indices

		# Scores from LightFM model prediction for specific products:
		products_for_rec = prods.head(num_top_prods_to_rec)
		prod_ixs = products_for_rec['product_id'].values - 1 #remember prod_ixs are one off the actual prod_ids
		scores = model.predict(user_ids = row_ixs, item_ids = prod_ixs) 

		# Get the re-ordered product list
		recom_products = products_for_rec.loc[np.argsort(-scores)]

		# Add in extra columns before appending to master table
		recom_products['user_id'] = user_id_to_get_rec 
		recom_products['rec_rank'] = np.arange(1,num_top_prods_to_rec+1,1) 
		recom_products.drop('frequency',axis=1,inplace=True)

		recom_products_master = recom_products_master.append(recom_products.head(recs_to_save))

	return recom_products_master

def main():
	path = '/content/drive/My Drive/Insight/'
	model,user_prod_interaction = load_cb_model_and_user_product_mat(path)
	df_users = load_user_df(path)
	products = load_product_list(path)

	# perform cosine similarity for each user with top 100 products to get 20 recs per user
	recom_products = generate_recommendations(df_users,products,recs_to_save=20,num_top_prods_to_rec=100)
	recom_products.to_csv('top20_products_recom.csv')

if __name__ == '__main__':
	main()