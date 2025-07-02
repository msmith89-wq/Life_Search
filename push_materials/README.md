# Life_Search

Exoplanet Detection and Classification for Habitability - Final Capstone Project


# Project Overview

This Notebook is a final capstone project focused on analyzing and classifying the potential of exoplanets inhabiting life using astronomical data. The dataset includes planetary and stellar properties of confirmed exoplanets. The project performs data cleaning, exploratory analysis, visualization, and machine learning to make predictions whether or not an exoplanet is capable of supporting life. 

# Dataset

Source: NASA Exoplanet Archive (PSCompPars dataset)

Key features include:

pl_rade: Planet radius (Earth radii)

pl_bmasse: Planet mass (Earth masses)

pl_eqt: Equilibrium temperature

discoverymethod

st_spectype 

st_metratio


# Technologies Used

Python

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn library

imblearn (SMOTE)

Jupyter Notebook


# Workflow Summary

Data Import & Cleaning:
First, read in the exoplanets dataframe, drop first 88 rows or metadata.
Get dataframe info
Display all columns and rows
Find out NaN percentage to determine features to drop from high NaN percentages
Drops columns with >50% missing values.
Make a correlation table of correlated features.
Make sure there are no duplicated rows.
Investigate all the different ways of the feature discovery method.
See how many spectral types there are.
Find out how many metallicity ratio types there are.
Replace [m/H] with [M/H].


Exploratory Data Analysis (EDA):
Investigate distribution of exoplanet count by Central Star Spectral Type.
Investigate distribution of exoplanet count by Discovery Method.
Investigate distribution of exoplanet count by Metallicity Ratio Type.
Investigate distribution of exoplanet count by Mass Provenance.


Feature Engineering & Preprocessing:
Drop irrelevant columns.
Analyze Habitable Worlds Catalog and determine ranges of values of features in common with the exoplanets dataframe.
Gather features into scaled features, ohe features, and bool features.
Extract rows with NaN values into a separate df for predictions later.
Form target variable using the mask function and Habitable World Catalog ranges
Impute scale features with mean strategy and ohe and bool features with most frequent strategy.
Divide predictor variables from target variable and perform train/test split.
Create pipeline with Standard Scaler for numeric features and One Hot Encoder for categorical features.
Fit pipeline with X_train and y_train.
Apply pipeline to X_train and X_test.
Apply SMOTE to X_train_transformed and y_train.

Modeling:
Using Randomized Search CV, fit a Gradient Boosting Classifier to X_train_balanced and y_train_balanced with the best parameters.
Make predictions on X_test_transformed and print classification report and confusion matrix.
Using Randomized Search CV, fit a Logistic Regression Model to X_train_balanced and y_train_balanced with the best parameters.
Make predictions on X_test_transformed and print classification report and confusion matrix.
Using Randomized Search CV, fit a MLP Classifier to X_train_balanced and y_train_balanced with the best parameters.
Make predictions on X_test_transformed and print classification report and confusion matrix.
Compare the f1 scores of the models, pick the best model and assign to best_model.
Use the best model's true positive predictions to extract the corresponding rows or exoplanet names that are predicted to be habitable to life.

Feature Importance:
Create a dataframe with feature names in one column and their corresponding importances in another column, and sort in descending importance.
Create a bar plot of the features with the top ten importances.

Applying Best Model to the Unlabeled NaN Dataframe:
Apply the same imputation and preprocessing steps to the unlabeled dataframe
Make predictions using best_model on the transformed unlabeled dataframe
Use the best model's true positive predictions to extract the corresponding rows or exoplanet names that are predicted to be habitable to life.

Applied a Partial Dependence Plot



#How to Run

Ensure required libraries are installed:

pip install pandas, numpy, matplotlib, seaborn, and scikit-learn

Place the dataset CSV in the correct path (adjust if needed).

Open the notebook in Jupyter and run all cells sequentially.


Interpretation

The notebook identifies key predictors of planet classification.

Shows which discovery methods are most common.

Uses robust modeling techniques to handle missing data and imbalanced classes.


Notes

Notebook assumes data is current and properly formatted.

Future work may include model tuning with RandomizedSearchCV and incorporating more advanced astrophysical models.

