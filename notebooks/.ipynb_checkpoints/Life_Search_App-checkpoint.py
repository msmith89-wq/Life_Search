#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    r2_score,
    mean_squared_error, 
    root_mean_squared_error,
    mean_absolute_error, 
    mean_absolute_percentage_error,
    accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# In[ ]:


import streamlit as st

st.title("Search for Life")


tab_intro_content = [
    """
    **Project Goals**  
    
    ● **Executive Summary** - The purpose of this project is to build a model that predicts whether or not an 
      exoplanet is able to support life based on the features provided in the dataset by NASA’s Exoplanet 
      Archive.  
   
    ● **MVP** - A supervised or unsupervised model that can correctly predict whether an exoplanet can 
                potentially support life or not.  
    """
      
    ,
    """
    **Sources**  
  
    ● Source of Main Dataframe: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars  
    ● Source of Habitable Worlds Catalog: https://phl.upr.edu/hwc  
    """     
]






exoplanets_df = pd.read_csv('../data/PSCompPars_2025.06.10_14.20.14.csv', skiprows=88)
exoplanets_df['st_metratio'] = exoplanets_df['st_metratio'].replace('[m/H]', '[M/H]')
dropped_columns = ['hostname', 'disc_year', 'disc_facility', 'pl_controv_flag', 'pl_orbpererr1', 'pl_orbpererr2', 'pl_orbperlim', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_orbsmaxlim', 'pl_radeerr1', 'pl_radeerr2', 'pl_radelim', 'pl_radj', 'pl_radjerr1', 'pl_radjerr2', 'pl_radjlim', 'pl_bmasseerr1', 'pl_bmasseerr2', 'pl_bmasselim', 'pl_bmassj', 'pl_bmassjerr1', 'pl_bmassjerr2', 'pl_bmassjlim', 'pl_orbeccenerr1', 'pl_orbeccenerr2', 'pl_orbeccenlim', 'pl_insolerr1', 'pl_insolerr2', 'pl_insollim', 'pl_eqterr1', 'pl_eqterr2', 'st_tefferr1', 'st_tefferr2', 'st_tefflim', 'st_raderr1', 'st_raderr2', 'st_radlim', 'st_masserr1', 'st_masserr2', 'st_metlim', 'st_loggerr1', 'st_loggerr2', 'st_logglim', 'rastr', 'decstr', 'sy_disterr1', 'sy_disterr2', 'sy_vmagerr1', 'sy_vmagerr2', 'sy_kmagerr1', 'sy_kmagerr2', 'sy_gaiamagerr1', 'sy_gaiamagerr2', 'st_masslim', 'st_meterr1', 'st_meterr2', 'pl_eqtlim', 'ttv_flag']
exoplanets_df = exoplanets_df.drop(columns=dropped_columns)
column_rename_map = {
    'pl_name': 'Planet Name',
    'pl_orbper': 'Orbital Period of Planet [days]',
    'pl_rade': 'Planet Radius [Earth Radius]',
    'pl_bmasse': 'Planet Mass [Earth Mass]',
    'pl_eqt': 'Equilibrium Temperature [K]',
    'pl_orbeccen': 'Orbital Eccentricity',
    'pl_orbsmax': 'Orbit Semi-Major Axis [AU]',
    'pl_insol': 'Incident Flux [Earth Flux]',
    'st_teff': 'Stellar Effective Temperature [K]',
    'st_rad': 'Stellar Radius [Solar Radius]',
    'st_mass': 'Stellar Mass [Solar Mass]',
    'st_met': 'Stellar Metallicity [dex]',
    'st_logg': 'Stellar Surface Gravity [log10(cm/s^2)]',
    'st_spectype': 'Star Spectral Type',
    'sy_dist': 'Distance [pc]',
    'sy_vmag': 'V-band Apparent Magnitude',
    'sy_kmag': 'K-band Apparent Magnitude',
    'discoverymethod': 'Discovery Method',
    'disc_year': 'Discovery Year',
    'default_flag': 'Default Parameter Set',
    'pl_bmassprov': 'Planet Mass Provenance',
    'st_metratio': 'Star Metal Ratio',
    'sy_snum': 'Number of Central Stars',
    'sy_pnum': 'Number of Planets',
    'sy_gaiamag': 'Gaia Magnitude'
}
exoplanets_df.rename(columns=column_rename_map, inplace=True)
hist_features = exoplanets_df.drop(columns=['Planet Name', 'Discovery Method', 'Planet Mass Provenance', 'Star Spectral Type', 'Star Metal Ratio', 'Distance [pc]']).columns.tolist()
bar_features = exoplanets_df[['Star Spectral Type', 'Star Metal Ratio']].columns.tolist()

if "selected_var" not in st.session_state:
    st.session_state.selected_var = exoplanets_df.columns[0]

import base64

def set_bg(jpg_file):
    with open(jpg_file, "rb") as f:
        data = f.read()
    b64_encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:Life_Search_App_Background/avif;base64,{b64_encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("Life_Search_App_Background.avif")

tab_intro, tab_eda, tab_preprocessing, tab_model, tab_unlabeled = st.tabs(["Intro", "EDA", "Preprocess", "Model", "Unlabel"])

with tab_intro:
    st.subheader("Project Overview")
    st.markdown("## Project Goals") 
    st.markdown(
        """
        ● **Executive Summary** - The purpose of this project is to build a model that predicts whether or not an 
            exoplanet is able to support life based on the features provided in the dataset by NASA’s Exoplanet 
            Archive.  
        
        ● **MVP** - A supervised or unsupervised model that can correctly predict whether an exoplanet can 
                    potentially support life or not.  
        """
       )

    st.markdown("---") 

    st.markdown("## Sources")

    st.markdown(
    """  
    ● Source of Main Dataframe: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars  
    
    ● Source of Habitable Worlds Catalog: https://phl.upr.edu/hwc  
    """     
    )
with tab_eda:
    st.subheader("EDA")
    col1, col2 = st.columns(2)
    with col1:
        selected_var = st.selectbox("Select numeric variable:", exoplanets_df.select_dtypes(include='number').columns)

    with col2:
        st.subheader(f"Histogram of {selected_var}")
        fig, ax = plt.subplots()
        ax.hist(exoplanets_df[selected_var], bins=10, color='skyblue', edgecolor='black')
        ax.set_xlabel(selected_var)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        var2 = st.selectbox("Select categorical variable:", exoplanets_df.drop(columns='Planet Name').select_dtypes(include='object').columns,       key="var2")
    with col4:
         st.subheader(f"Bar Chart of {var2}")
         fig2, ax2 = plt.subplots()
         exoplanets_df.groupby(var2)['Planet Name'].count().plot(kind='bar', ax=ax2, color='salmon', edgecolor='black')
         ax2.set_title(f'Count of Exoplanets by {var2}')
         ax2.set_ylabel('Exoplanet Count')
         ax2.set_xlabel(var2)
         st.pyplot(fig2)

slide_data = [
    """
    #Features in Common
    
  \n● **Discovery Method** – Astronomical method used to discover exoplanet.
     \n - **Transit Method** – Observes a star’s brightness, detects temporary dips caused by a planet passing in front (a transit).
     \n - **Radial Velocity Method** – Detects exoplanets by measuring the wobble of a star from the gravitational pull of orbiting planets.
    
 \n ● **Mass** – Mass of the planet in Earth masses (Habitable Range = 0.39–3.19 ME).
 \n ● **Radius** – Radius in Earth radii (Habitable Range = 0.92–1.60 RE).
  \n● **Flux** – Stellar flux (Habitable Range = 0.25–1.48 SE).
  \n● **Tsurf** – Estimated surface temperature in K (Habitable Range = 203–316 K).
 \n ● **Period** – Orbital period in days (Habitable Range = 4.05–267 days).
    """,
    st.markdown("---"),
    """
    #Target Variable Formation

   \n ● Used the ranges above and applied them using a mask function.
   \n ● Set aside an unlabeled dataframe of NaN rows for later predictions.
   \n ● Rows with values in all required ranges were labeled `1` (potentially habitable), others as `0`.
   \n ● Dropped rows with NaNs from the labeled set to keep training clean.
    """,
    st.markdown("---"),
    """
    #Preprocessing Techniques

   \n ● Dropped all variables that had a NaN percentage greater than 70%, which was 4 features
   \n ● Dropped the features used to create the Target Variable
   \n ● Used Simple Imputer with a strategy parameter of mean for all numeric variables
   \n ● Used Simple Imputer with a strategy parameter of most frequent for all categorical and 
      boolean variables
   \n ● Created a pipeline that applied Standardized Scaler to all numeric variables and One Hot 
      Encoder to all categorical variables
   \n ● Used a train/test split technique with a test size parameter of 0.3 and fit the pipeline to the 
         training data
   \n ● Used the fitted pipeline to transform X_train and X_test
   \n ● Applied SMOTE to fit and resample X_train_transformed and y_train since the data was very 
      imbalanced (about 0.2% of observations were True in target value)
      """
]

with tab_preprocessing:
    st.subheader("Target Variable Formation and Preprocessing")
    st.markdown(
         """
    ## Features in Common
    
  \n● **Discovery Method** – Astronomical method used to discover exoplanet.
     \n - **Transit Method** – Observes a star’s brightness, detects temporary dips caused by a planet passing in front (a transit).
     \n - **Radial Velocity Method** – Detects exoplanets by measuring the wobble of a star from the gravitational pull of orbiting planets.
    
 \n ● **Mass** – Mass of the planet in Earth masses (Habitable Range = 0.39–3.19 ME).
 \n ● **Radius** – Radius in Earth radii (Habitable Range = 0.92–1.60 RE).
  \n● **Flux** – Stellar flux (Habitable Range = 0.25–1.48 SE).
  \n● **Tsurf** – Estimated surface temperature in K (Habitable Range = 203–316 K).
 \n ● **Period** – Orbital period in days (Habitable Range = 4.05–267 days).
    """
    )

    st.markdown('---')

    st.markdown(
       """
    ## Target Variable Formation

   \n ● Used the ranges above and applied them using a mask function.
   \n ● Set aside an unlabeled dataframe of NaN rows for later predictions.
   \n ● Rows with values in all required ranges were labeled `1` (potentially habitable), others as `0`.
   \n ● Dropped rows with NaNs from the labeled set to keep training clean.
    """
    )

    st.markdown('---')

    st.markdown(
         """
    ## Preprocessing Techniques

   \n ● Dropped all variables that had a NaN percentage greater than 70%, which was 4 features
   \n ● Dropped the features used to create the Target Variable
   \n ● Used Simple Imputer with a strategy parameter of mean for all numeric variables
   \n ● Used Simple Imputer with a strategy parameter of most frequent for all categorical and 
      boolean variables
   \n ● Created a pipeline that applied Standardized Scaler to all numeric variables and One Hot 
      Encoder to all categorical variables
   \n ● Used a train/test split technique with a test size parameter of 0.3 and fit the pipeline to the 
         training data
   \n ● Used the fitted pipeline to transform X_train and X_test
   \n ● Applied SMOTE to fit and resample X_train_transformed and y_train since the data was very 
      imbalanced (about 0.2% of observations were True in target value)
      """
    )

with tab_model:
    st.subheader("Model Results and Feature Importance")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image("Screenshot 2025-07-09 131735.png", width=300)

        with col2:
            st.image("Screenshot 2025-07-09 131756.png", width=300)


    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image("Screenshot 2025-07-09 131812.png", width=300)

        with col2:
            st.markdown(
            """
          \n ● Along with these models, I also used Randomized Search Cross Validation 
               to get the best possible parameters for each model. 
          \n ● Logistic Regression predicted the most true positives or habitable 
               planets, but has the worst f1 score, which is the biggest indicator of a 
               model’s effectiveness. 
          \n ● Gradient Boosting Classifier Model only correctly predicted one potential 
               habitable planet, but since it has the best f1 score, it is the best model out 
               of the three, followed by the MLP Classifier Model and then the Logistic 
               Regression Model.
            """
        )

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
            """
             ● st_rad or stellar radius is many 
               times more powerful as a 
               variable in determining if an 
               exoplanet is habitable than the 
               other top ten variables. 
            """
               )

        with col2:
             st.image("Screenshot 2025-07-09 131844.png", width=400)

with tab_unlabeled:
    st.subheader("Prediction Results on Unlabeled Dataframe")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
             st.markdown(
            """
             ● The same preprocessing methods and 
               model was applied to the unlabeled 
               dataframe 
             
             ● The resulting positive predictions are 
               listed to the right, a total of 9 exoplanets 
               were predicted as being habitable, with 
               4 having a 100% probability of being 
               habitable based on the model
            """
             )

        with col2:
            st.image("Screenshot 2025-07-09 170812.png", use_column_width=True)

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("## Click on Planet Name Below for more Information")
            st.markdown("[GJ 1002 b](https://science.nasa.gov/exoplanet-catalog/gj-1002-b/)")
            st.markdown("[Kepler-1229 b](https://science.nasa.gov/exoplanet-catalog/kepler-1229-b/)")
            st.markdown("[Kepler-1652 b](https://science.nasa.gov/exoplanet-catalog/kepler-1652-b/)")
            st.markdown("[Kepler-438 b](https://science.nasa.gov/exoplanet-catalog/kepler-438-b/)")
            st.markdown("[Proxima Cen b](https://science.nasa.gov/exoplanets/other-stars-other-worlds/our-nearest-celestial-neighbor-an-exotic-3-star-system/)")
            st.markdown("[TOI-715 b](https://science.nasa.gov/exoplanet-catalog/toi-715-b/)")
            st.markdown("[TRAPPIST-1 e](https://science.nasa.gov/exoplanet-catalog/trappist-1-e/)")
            st.markdown("[TOI-700 d](https://www.nasa.gov/universe/nasa-planet-hunter-finds-its-1st-earth-size-habitable-zone-world/)")
        
        with col2:
            st.markdown(
          """
           ● All three models that I used predicted extremely few exoplanets as habitable correctly, 
             which agrees with the expectation I had that planets having the proper conditions to 
             support life is extremely rare.
          
           ● This also concurs with the fact that the ranges of values of the variables recorded in the 
             Habitable World’s Catalog are extremely narrow compared to the ranges of values of those 
             same variables in the exoplanets dataset.
          """
        )




