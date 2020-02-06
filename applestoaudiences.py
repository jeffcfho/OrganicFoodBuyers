import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

st.sidebar.markdown("""
	# <span style="color:green">Apples to Audiences:green_apple::family:</span>
	""",unsafe_allow_html=True)
st.sidebar.subheader('A tool for targeting organic produce buyers')

# import data (for this demo, the data lives locally, but in a more 
#   secure application it would be retrieved from an external source)
user_list = pd.read_csv('modeling_dfs/final_users_50k.csv')
rec_list = pd.read_csv('top20_products_recom.csv')
prod_list = pd.read_csv('top200_products.csv')
prod_list = prod_list.loc[prod_list['organic']==1]

view_names = ['Target individual users','Get likely buyers','Get likely buyers by item','About']
view = st.sidebar.radio('',view_names)
# Individual user view
if view==view_names[0]:
	# st.subheader()
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
	st.markdown(f"""
		## {view_names[0]}
		### <span style="color:green">{user_name}</span> will next buy organic produce with a probability of <span style="color:red">{user_prob:.2f}</span>.
		What would increase the probability?
		- This user <span style="color:green">{bought_produce}</span> organic produce in their most previous order.
		- <span style="color:green">{pct_organic:.0f}%</span> of their top 20 recommendations are organic, compared to the 52% user average.
		""",unsafe_allow_html=True)

	# Show top recommendations for user
	rec_user = rec_list.loc[(rec_list['user_id']==selected_user_id)&(rec_list['organic']==1),['product_name','freq_rank']]
	rec_user.set_index(np.arange(1,len(rec_user)+1,1),inplace=True)
	rec_user.columns = ['Recommended items','Item popularity rank']
	st.dataframe(rec_user)
# List of users view
elif view==view_names[1]:
	st.markdown(f"## {view_names[1]}")
	min_predicted_prob = st.slider("Show users with probabilities greater than",min_value=0.5,max_value=0.99,step=0.01,value=0.9)
	user_emails = user_list.loc[(user_list['predicted_prob']>min_predicted_prob),'user_emails']

	st.markdown(f"""
		There are <span style="color:red">{len(user_emails)}</span> users to target. A random sample of 20 is shown below:
		""",unsafe_allow_html=True)
	st.code(',\n'.join(user_emails.sample(20).values)+'...')
elif view==view_names[2]:
	st.markdown(f"## {view_names[2]}")
	# Get list of users who were recommended a product
	prod_option = st.selectbox(
	    'Select an item to get a list of users with likelihood of buying > 0.5 who are recommended that product',
	     prod_list['product_name'].head(19).values)

	users_with_prod_rec = rec_list.loc[rec_list['product_name']==prod_option,'user_id']
	user_list_prod = user_list.loc[user_list['user_id'].map(lambda x: x in users_with_prod_rec)]
	user_emails = user_list_prod.loc[(user_list['predicted_prob']>0.5),'user_emails']

	st.markdown(f"""
		There are <span style="color:red">{len(user_emails)}</span> users to target who are likely to buy the item <span style="color:green">[{prod_option}]</span>.
		
		A random sample of at most 20 is shown below:
		""",unsafe_allow_html=True)
	if (len(user_emails)>=20):
		st.code(',\n'.join(user_emails.sample(20).values)+'...')
	else:
		st.code(',\n'.join(user_emails.values)+'...')
# About page
elif view==view_names[3]:
	st.markdown(f"""
		## {view_names[3]}
		Organic food is the <span style="font-weight:bold">fastest growing category in retail grocery</span> today, but still represents <span style="color:green;font-weight:bold">just 6% of the total market share</span> in the U.S. To retain and continue to grow the market, <a href="https://maccabee.com/case_study/building-awareness-sales-for-organic-food-products/" target="_blank">organic trade associations provide coupons and other advertising</a> to customers to incentivize purchases.

		Apples to Audiences is a web app built upon an API that <span style="font-weight:bold">identifies customers who are likely to buy organic produce next and recommends items for them</span>, based on their past shopping history. The model blend results from collaborative filtering and logistic regression models to provide both a likelihood of purchase plus specific items to recommend. 
		""",unsafe_allow_html=True)
	#Show histogram of probabilities
	f = px.histogram(user_list, x="predicted_prob", nbins=20, title="User distribution",
					 color_discrete_sequence=['green'])
	f.update_xaxes(title="Probability of next organic purchase")
	f.update_yaxes(title="Number of users")
	st.plotly_chart(f)

	st.markdown("""
		Compared to traditional marketing and demographics surveys, this allows for more focused targeting based on actual past purchases. More focused targeting will <span style="color:green;font-weight:bold">increase lift in the percentage of purchases with organic items</span>.

		The data used to train the model come from 3.4 million orders made by 200k users made public by <a href ="https://www.instacart.com/datasets/grocery-shopping-2017" target="_blank">Instacart</a>.
		""",unsafe_allow_html=True)




