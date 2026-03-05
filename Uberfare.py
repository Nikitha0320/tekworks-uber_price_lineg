import streamlit as st
st.title("Uber Fare Prediction🚖💰")
st.divider()
import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv("Uberfare.csv")
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={"distance_km": "distance"})

distance=st.number_input("Distance (km)")
passengers=st.number_input("Number of Passengers", min_value=1, max_value=10, value=1)
if st.button("Predict Fare"):
    X=df[["distance","passengers"]]
    y=df["fare"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(X_train,y_train)
    fare=model.predict([[distance,passengers]])    
    st.success(f"Predicted Fare: ₹{fare[0]:.2f}")
    if passengers>1:
        fare_per_passenger=fare[0]/passengers
        st.info(f"Fare per Passenger: ₹{fare_per_passenger:.2f}")

