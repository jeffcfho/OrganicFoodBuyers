# Code overview

This folder contains the development pipeline for the project. Dataframes are passed between notebooks through the exporting of dataframes as csv files and the saving of models as pickle files. For modular code that stem from these notebooks, see the `code_modules/` folder.

## Notebook themes

1. Manipulating different data tables (joining/merging) using pandas to get data ready for model development
	- `0b_Create_Modeling_Dataframes`


2. Collaborative filter building and testing 
	- `1a_Building_collaborative_filter_colab` - Fitting the collaborative filter using LightFM
	- `2a_Collaborative_filter_recommendations_colab` - Generating recommendations from the model
	- `2c_Validating_recommendations` - Comparing the top recommended items with items actually purchased by users
	- `2d_What_pct_recommendations_were_new` - Assessing how many recommendations were items that the user had not purchased before.


3. Logistic regression building and testing with scikit-learn
	- `1b_LogisticRegression_model_exploration` - First explorations to see whether I could predict individual items in carts (e.g., organic bananas)
	- `2b_LogisticRegression_simple_model_benchmark` - Setting the benchmark for predicting whether a user will buy organic next - did they buy organic produce before?
	- `3_Final_Model_Tests_ClassificationTree_vs_LogisticRegression` - Evaluating different interpretable models for predicting next organic purchase. Contains feature engineering and model evaluation.