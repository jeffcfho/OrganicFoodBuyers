import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title('Or-gui-nic: Data-driven insights for the organic food industry')

top50_organic = pd.read_csv('top50_organic_items.csv')
top50_organic.columns = ['product_name','num_items']

option = st.sidebar.selectbox(
    'Which item are you interested in?',
     top50_organic['product_name'])

st.sidebar.markdown('You selected: {}'.format(option))

filename = 'saved_models/model_3predictors.pkl'
model = pickle.load(open(filename, 'rb'))

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)