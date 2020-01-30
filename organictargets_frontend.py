import streamlit as st
import base64
import pickle
import pandas as pd
import numpy as np

st.sidebar.title('Organic Targets')
st.sidebar.subheader('A tool for targeting organic produce buyers')

# Select by user
user_list = pd.read_csv('modeling_dfs/final_users_50k.csv')

view_names = ['Target individual users','Get likely buyers','About']
view = st.sidebar.radio('',view_names)
# Individual user view
if view==view_names[0]:
	st.subheader(view_names[0])
	#Sidebar elements
	max_predicted_prob = st.sidebar.slider("Show users with probabilities less than",min_value=0.1,max_value=1.0,step=0.01,value=0.9)
	option = st.sidebar.selectbox(
	    'Select a user to see what products to recommend',
	     user_list.loc[(user_list['predicted_prob']<max_predicted_prob),'user_dropdown'].head(20).values)

	selected_user_id = int(option[5:-9])
	user_one = user_list.loc[user_list['user_id']==selected_user_id]
	# if st.sidebar.checkbox('Show dataframe (internal)'):
	# 	st.write(user_one)

	#Main page elements
	user_name = option[:-9] # Show about user
	user_prob = user_one['predicted_prob'].values[0]
	if (user_one['any_organic_produce_prev'].values[0]):
		bought_produce = 'bought'  
	else:
		bought_produce = 'did not buy'  
	pct_organic = 100*user_one['pct_organic'].values[0]
	st.write(f"{user_name} will next buy organic produce with a probability of {user_prob:.2f}.")
	st.text(f"- This user {bought_produce} organic produce in their most previous order.")
	st.text(f"- {pct_organic:.0f}% of their top 20 recommendations are organic, compared to the 52% user average.")

	# Show top recommendations for user
	rec_list = pd.read_csv('top20_products_recom.csv')
	rec_user = rec_list.loc[(rec_list['user_id']==selected_user_id)&(rec_list['organic']==1),['product_name','freq_rank']]
	rec_user.set_index(np.arange(1,len(rec_user)+1,1),inplace=True)
	rec_user.columns = ['Recommended items','Item popularity rank']
	st.dataframe(rec_user)
# List of users view
elif view==view_names[1]:
	st.subheader(view_names[1])
	min_predicted_prob = st.slider("Show users with probabilities greater than",min_value=0.5,max_value=0.99,step=0.01,value=0.9)
	user_emails = user_list.loc[(user_list['predicted_prob']>min_predicted_prob),'user_emails']
	st.write(f"There are {len(user_emails)} users to target. A random sample of 20 is shown below:")
	st.code(',\n'.join(user_emails.sample(20).values)+'...')
# About page
elif view==view_names[2]:
	st.subheader(view_names[2])
	st.write('Organic food is the fastest growing category in retail grocery today, but still represents only 6% of the total market share in the U.S. To retain and continue to grow the market, organic trade associations provide coupons and other advertising to customers to incentivize purchase.')
	st.write('Organic Targets is an API that identifies customers who are likely to buy organic produce in their next grocery purchase based on their past shopping history.')
	st.write('Compared to traditional methods such as marketing and demographics surveys, this allows for more focused targeting based on actual past purchases.')
	st.write("Organic Targets blends results from collaborative filtering and logistic regression models to provide both a likelihood of purchase plus specific items to recommend.")




