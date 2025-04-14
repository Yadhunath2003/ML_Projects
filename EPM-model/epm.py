"""In this project we will be predicting the Carbon Dioxide Emission rates using the
scikit-learn library through pipelines"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#For preprocessong the data that we are creating.
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#for getting rid of any missing values
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


"""Now we load the data and perform some tasks."""
data = pd.read_csv("vehicle_emissions.csv")

# print(data.head())

"""Creating features are target values"""

X = data.drop(["CO2_Emissions"], axis=1) #We want to drop this because this what I want to find with the model.
y = data["CO2_Emissions"]


"""We want to segregate the numerical and categorigal values to make the model more efficient."""

numerical_col = ["Model_Year", "Engine_Size", "Cylinders", "Fuel_Consumption_in_City(L/100 km)", 
                 "Fuel_Consumption_in_City_Hwy(L/100 km)", "Fuel_Consumption_comb(L/100km)", "Smog_Level"]
categorical_col = ["Make", "Model", "Vehicle_Class","Transmission"]

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('scalar', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
])

#Joining the pipelines together.
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_col),
    ('cat', categorical_pipeline, categorical_col)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

"""Spliting the test and the train models"""
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)

"""Train and predicting the model"""
pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)

encoded_cols= pipeline.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(categorical_col)
# print(encoded_cols)

'''Evaluating the predication of our model.'''
mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)

#Proportion of variance in the target value
r2 = r2_score(y_test, prediction)
#Actaul error difference between the test set and the prediction values.
mae = mean_absolute_error(y_test, prediction)

print(f'Model Performance:')
print(f'R2 Score: {r2}')
print(f'Root Mean Square Error:{rmse}')
print(f'Mean Absolute error: {mae}')

'''Creating a joblib file to store this model for future uses'''
joblib.dump(pipeline, "vehicle_emission_pipeline.joblib")


