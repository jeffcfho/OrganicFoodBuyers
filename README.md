# Apples to Audiences: A tool for targeting organic produce buyers

## Motivation 
Organic food is the fastest-growing category in retail grocery today, but still represents just 6% of the total market share in the U.S. To grow the market, organic trade associations provide coupons and other advertising to incentivize purchases, but currently such promotions are not targeted to individual consumers.

Can likely organic buyers be identified in a way that grows organic sales? Apples to Audiences is a tool that identifies users who are likely to buy organic produce based on their past shopping history, combining recommendations from a collaborative filtering model with predictions from a logistic regression model. 

The model behind Apples to Audiences identifies 10% new likely buyers of organic food and reduces spam to unlikely buyers by 40% compared to not targeting. More focused targeting will increase lift in the percentage of purchases with organic items.

## Data sources

1. [Instacart order data (3.4 million orders made by 200k users)](https://www.instacart.com/datasets/grocery-shopping-2017)

## Tech stack

1. Pandas
2. LightFM
3. Scikit-learn
4. Streamlit
5. EC2
