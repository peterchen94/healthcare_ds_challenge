
# Healthcare startup DS challenge

# High Level Process
1. Parse text files to generate training/testing dataframes where each row corresponds to a single patient mapped to a specific outcome
2. Visualize features/target
3. Do an initial model search with the Dataiku tool - determine which models might work the best
4. Analyze best model and simplify to avoid overfitting
5. Predict on test set


# Low Level Process

Preprocessing
1. Initialize paths and column headers
2. Extract and clean target variables for training and testing
3. Drop records with missing target variable and visualize distribution between training/test sets
4. Extract non-time, 1day, 2day features from text files using pivots and aggregations (mean)
5. Concatenante patient data frames and merge columns to generate the training/testing dataframes with feature/target pairs
6. Drop excessive Nan columns

Initial Model search
7. Use Dataiku to explore models for nontime features
8. Use Dataiku to explore models for 1day features
9. Use Dataiku to explore models for 2day features
10. Look into classification analysis for the best regression model - 2 day model

Analysis
11. Replicate best classification model with python
12. Model Diagnostics via learning curve
13. Build Simpler Random Forest Model
14. Simplify model additionally with Sequential Feature Selector
15. Predict on Testing set
s

# Files

1. data_preprocessing.ipynb - data preprocessing
2. initial_model_search.ipynb - initial model search
3. model_analysis - detailed model analysis

# Directory Structure


/ - root folder. This is where the coding files go 

/data - datafiles. This is where the preprocessed data and the files from the challenge go
	1. The datafiles used from the challenge are the set-a and set-b folders along with the outcome txt files

/img - images from the dataiku tool