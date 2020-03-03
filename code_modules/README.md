# Code overview

This folder modularizes the final model development pipeline shown in the `notebooks/` folder.

## Modules

1. Collaborative Filter - Procedural code to load Instacart data, generate a user-product-frequency matrix, and fit/save a collaborative filtering model using LightFM.

2. Get Recommended Products - Procedural code to generate and save the top 20 recommended products for each user.

3. Buyer Likelihood Model - Procedural code to fit and save a model predicting which users are likely to buy organic food next.

4. Format Predictions for Front End - Procedural code to generate and format predicted probabilities for display in a front end