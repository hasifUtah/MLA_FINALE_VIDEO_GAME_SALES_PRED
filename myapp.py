import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats

# Initialize LabelEncoders
le_platform = LabelEncoder()
le_genre = LabelEncoder()
le_publisher = LabelEncoder()

df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')


df = df.rename(columns={"Year_of_Release": "Year",
                            "NA_Sales": "NA",
                            "EU_Sales": "EU",
                            "JP_Sales": "JP",
                            "Other_Sales": "Other",
                            "Global_Sales": "Global"})
df = df[df["Year"].notnull()]
df = df[df["Genre"].notnull()]
df["Year"] = df["Year"].apply(int)
df["Age"] = 2018 - df["Year"]


# Deletion for 'Publisher' column
df.dropna(subset=['Publisher'], inplace=True)


# Imputation for 'Year' column
median_year = df['Year'].median()
df['Year'] = df['Year'].fillna(median_year)


df.replace('tbd', np.nan, inplace=True)

# Convert numeric columns to float
numeric_columns = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
df[numeric_columns] = df[numeric_columns].astype(float)

# Numeric Columns Imputation
numeric_imputer = SimpleImputer(strategy='median')  # Change strategy to 'mean' or 'most_frequent' if needed
df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

# Categorical Columns Imputation
categorical_columns = ['Developer', 'Rating']
categorical_imputer = SimpleImputer(strategy='most_frequent')  # Change strategy to 'constant' and add fill_value='Unknown' if needed
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])


# Encode categorical features
df['Platform'] = le_platform.fit_transform(df['Platform'])
df['Genre'] = le_genre.fit_transform(df['Genre'])
df['Publisher'] = le_publisher.fit_transform(df['Publisher'])

# Define features (X) and target (y)
X = df[['Platform', 'Genre', 'NA', 'EU', 'JP', 'Other']].values
y = df['Global'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model
rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train, y_train)

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train, y_train)

st.write("""
# Video Game Sales Prediction App

This app predicts the **Video Game Sales** all over the world !
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Platform = st.sidebar.selectbox('Platform', le_platform.classes_)
    Genre = st.sidebar.selectbox('Genre', le_genre.classes_)
    #Publisher = st.sidebar.selectbox('Publisher', le_publisher.classes_)
    NA = st.sidebar.slider('NA Sales', 0.0, 10.0, 0.5)
    EU = st.sidebar.slider('EU Sales', 0.0, 10.0, 0.5)
    JP = st.sidebar.slider('JP Sales', 0.0, 10.0, 0.5)
    Other = st.sidebar.slider('Other Sales', 0.0, 10.0, 0.5)
    
    data = {
        'Platform': le_platform.transform([Platform])[0],
        'Genre': le_genre.transform([Genre])[0],
        #'Publisher': le_publisher.transform([Publisher])[0],
        'NA': NA,
        'EU': EU,
        'JP': JP,
        'Other': Other
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input parameters')
st.write(df_input)

# Predict using the trained model
prediction1 = rfr.predict(df_input)
prediction2 = dtr.predict(df_input)

st.subheader('Prediction')
st.write(f'Predicted Global Sales using RandomForestRegressor: {prediction1[0]:.2f} million units')
st.write(f'Predicted Global Sales using DecisionTreeRegressor: {prediction2[0]:.2f} million units')