import streamlit as st
import pandas as pd
import statsmodels.api as sm
#import statsmodels.tsa.statespace.api as sm
import numpy as np

def load_model():
    # Replace the following lines with your model training code and return the trained model
    # For demonstration purposes, I'm using the model training code from the previous answer
    df = pd.read_csv("test.csv")
    df['Geography'] = df['Geography'].str.strip()
    df['Education'] = df['Education'].str.strip()
    df["Geography"] = df["Geography"].replace({"British Columbia": 0, "Ontario": 1})
    df["Education"] = df["Education"].replace({"Bachelor's degree": 0, "High school graduate": 1}).astype('int64')
    
    X = df[["Year", "Geography", "Education", "M_(1)_or_F_(0)"]]
    y = df["Weekly_Wage"]
    model = sm.OLS(y, sm.add_constant(X)).fit()
    return model

model = load_model()

def app():
    st.title("Weekly Wage Prediction Model")

    # Input controls
    year = st.slider("Year:", min_value=1997, max_value=2019, value=2000)
    geography = st.selectbox("Region:", options=["British Columbia", "Ontario"])
    education = st.selectbox("Education Level:", options=["Bachelor's degree", "High school graduate"])
    sex = st.radio("Sex:", options=["Female", "Male"])

    # Prepare data for prediction
    newdata = pd.DataFrame({
        "const": [1],
        "Year": [year],
        "Geography": [geography],
        "Education": [education],
        "M_(1)_or_F_(0)": [1 if sex == "Male" else 0]
    })

    newdata["Geography"] = newdata["Geography"].replace({"British Columbia": 0, "Ontario": 1})
    newdata["Education"] = newdata["Education"].replace({"Bachelor's degree": 0, "High school graduate": 1})
    

    # Make prediction using the model
    prediction = model.predict(newdata)

    # Display prediction
    st.write("Predicted weekly wage:", round(prediction[0]-5.26, 2))

if __name__ == "__main__":
    app()