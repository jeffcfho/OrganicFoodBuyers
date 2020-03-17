"""
	Testing whether the final model in 'saved_models/final_model.pkl' has the right characteristics to make predictions
"""

import pandas as pd
import pickle

def test_predict_proba(filepath_rel='./'):
	'''
		Test that the model has predict_proba(), e.g., sklearn LogisticRegression
	'''
	model_filename = filepath_rel+'saved_models/model_final.pkl'
	model = pickle.load(open(model_filename, 'rb'))
	assert hasattr(model,'predict_proba'), 'Model in saved_models/final_model.pkl does not have predict_proba().'

if __name__=='__main__':
	test_predict_proba(filepath_rel='../../') #for testing in the same folder