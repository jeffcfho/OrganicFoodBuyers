"""
	Fits and saves a model that predicts which users are likely to buy organic food next.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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

def print_scores(X,y,model):
	"""
		Print fit metrics of model using built-in methods
	"""
    print('Precision:')
    print(metrics.precision_score(y.astype(int),model.predict(X).astype(int)))
    print('Recall:')
    print(metrics.recall_score(y.astype(int),model.predict(X).astype(int)))
    print('F1 score (0.3):')
    print(metrics.f1_score(y.astype(int),(model.predict_proba(X)[:,1]>0.3).astype(int)))
    print('F1 score (0.4):')
    print(metrics.f1_score(y.astype(int),(model.predict_proba(X)[:,1]>0.4).astype(int)))
    print('F1 score (0.5):')
    print(metrics.f1_score(y.astype(int),(model.predict_proba(X)[:,1]>0.5).astype(int)))
    print('F1 score (0.6):')
    print(metrics.f1_score(y.astype(int),(model.predict_proba(X)[:,1]>0.6).astype(int)))
    print('Mean accuracy:')
    print(model.score(X,y))
    print('ROC AUC score:')
    print(metrics.roc_auc_score(y.astype(int),model.predict(X).astype(int)))
    print('Confusion matrix:')
    print(metrics.confusion_matrix(y, model.predict(X)))
    
    print('ROC curve')
    fpr, tpr, threshes = metrics.roc_curve(y.astype(int),model.predict_proba(X)[:,1])
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()

def fit_log_reg(df_modeling,y_col,X_cols,print_scores=False):
	"""
		Fits a logistic regression model to a particular dataframe given the Y column name (string) and X column names (list of strings).
		Returns a Scikit-learn logistic regression model object
	"""

	# Use an 80/20 train test split
	X_train, X_test, y_train, y_test = train_test_split(\
    	df_modeling[X_cols], df_modeling[y_col], test_size=0.20, random_state=22)

	# Fit model on training data
	log_reg_model = LogisticRegression(penalty='l1',solver='liblinear')
	log_reg_model.fit(X_train,y_train)

	# Show how well the model did on train and test data
	if print_scores:
		print('Model performance on test data:')
		print_scores(X_test,y_test,logisticRegr)
		print('Model performance on training data:')
		print_scores(X_train,y_train,logisticRegr)

	return log_reg_model

def save_log_reg_model(m,model_filename):
	"""
		Save logistic regression model file
	"""
    pickle.dump(m, open(model_filename, 'wb'))

def main():
	df = create_modeling_df()
	pred_col = 'any_organic_produce'
	feature_cols = ['any_organic_produce_prev','pct_organic',]
	logisticRegr = fit_log_reg(df_modeling,pred_col,feature_cols,print_scores=True)
	save_log_reg_model(logisticRegr,'../saved_models/model_final.pkl')

if __name__ == '__main__':
	main()