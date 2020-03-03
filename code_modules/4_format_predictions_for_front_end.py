"""
	Given a model, generates and formats predictions to be displayed in a front end
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from utils import create_modeling_df

def load_log_reg_model(model_filename):
	"""
		Load a scikit-learn logistic regression model
	"""
    return pickle.load(open(model_filename, 'rb'))

def add_front_end_cols(df_users,model,feat_cols):
	"""
		Add a column to df_users predicting likelihoods based on feat_cols and add a formatted column for users.
		- Assumes model has a .predict_proba() function, e.g., sklearn LogisticRegression
		Returns df_users with new columns
	"""
	df_users['predicted_prob'] = model.predict_proba(df_users[feature_cols])[:,1] # take second column because that is "True"

	assert 'user_id' in test.columns, "df_users needs to have column 'user_id' "
	df_users['user_dropdown'] = df_users.apply(lambda x: f"User {x['user_id']} (p={x['predicted_prob']:.2f})",axis=1)
	df_users['user_emails'] = df_users.apply(lambda x: f"User {x['user_id']} <user{x['user_id']}@email.com)",axis=1)
	return df_users.sort_values(by='predicted_prob',ascending=False)

def main():
	# Load best model for front end
	logisticRegr = load_log_reg_model('..saved_models/model_final.pkl')
	feature_cols = ['any_organic_produce_prev','pct_organic']
	df = create_modeling_df()

	# Add columns for front end and export
	df_frontend = add_front_end_cols(df,logisticRegr,feature_cols)
	df_frontend.to_csv('../modeling_dfs/final_users_50k.csv')

if __name__ == '__main__':
	main()