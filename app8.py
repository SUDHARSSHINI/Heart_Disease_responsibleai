import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from responsibleai import RAIInsights, FeatureMetadata
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
import streamlit as st
import os

# Load the synthetic dataset
csv_file_path = 'heart_disease_prediction.csv'

if os.path.isfile(csv_file_path):
    data = pd.read_csv(csv_file_path)
else:
    st.error("CSV file not found.")
    st.stop()

# Prepare the dataset
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']
y = y.astype(int)

# Define categorical features
categorical_features = ['Sex']
X[categorical_features] = X[categorical_features].astype(str)

# Check and handle missing values
X = X.dropna()
y = y[X.index]

# Create a column transformer to preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Define a function to predict heart disease
def predict_heart_disease(age, sex, cholesterol, blood_pressure):
    new_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure]
    })
    new_data['Sex'] = new_data['Sex'].astype(str)
    try:
        prediction = pipeline.predict(new_data)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
    return "Heart Disease" if prediction[0] == 1 else "No Heart Disease"

# Initialize Responsible AI insights
def initialize_rai_insights():
    test_df = pd.concat([pd.DataFrame(X_test), pd.Series(y_test, name='Heart Disease')], axis=1)
    test_data_aif360 = BinaryLabelDataset(
        df=test_df,
        label_names=['Heart Disease'],
        protected_attribute_names=['Sex']
    )
    feature_metadata = FeatureMetadata(
        categorical_features=['Sex']
    )
    rai_insights = RAIInsights(
        model=pipeline,
        train=pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name='Heart Disease')], axis=1),
        test=test_df,
        target_column='Heart Disease',
        task_type='classification',
        feature_metadata=feature_metadata
    )
    rai_insights.explainer.add()
    rai_insights.error_analysis.add()
    rai_insights.counterfactual.add(total_CFs=10, desired_class='opposite')
    rai_insights.causal.add(treatment_features=['Cholesterol', 'Blood Pressure'])
    rai_insights.compute()
    return rai_insights, test_data_aif360

# Initialize RAI insights
rai_insights, test_data_aif360 = initialize_rai_insights()

# Streamlit app for heart disease prediction
def app():
    st.title('Heart Disease Prediction')
    
    age = st.number_input("Enter age:", min_value=20, max_value=80, step=1)
    sex = st.selectbox("Select sex:", [0, 1])  # 0 for female, 1 for male
    cholesterol = st.number_input("Enter cholesterol level:", min_value=150, max_value=300, step=1)
    blood_pressure = st.number_input("Enter blood pressure:", min_value=80, max_value=180, step=1)
    
    if st.button('Predict'):
        prediction = predict_heart_disease(age, sex, cholesterol, blood_pressure)
        if prediction:
            st.write("Prediction Result:", prediction)

    st.header("Responsible AI Insights")
    
    st.subheader("Fairness Metrics")
    metric = ClassificationMetric(test_data_aif360, rai_insights.counterfactual.get(), privileged_groups=[{'Sex': 1}], unprivileged_groups=[{'Sex': 0}])
    st.write("Disparate Impact:", metric.disparate_impact())
    st.write("Statistical Parity Difference:", metric.statistical_parity_difference())
    st.write("Equal Opportunity Difference:", metric.equal_opportunity_difference())
    st.write("Average Odds Difference:", metric.average_odds_difference())
    
    st.subheader("Explainer Insights")
    st.write(rai_insights.explainer.get())
    
    st.subheader("Error Analysis Insights")
    st.write(rai_insights.error_analysis.get())
    
    st.subheader("Counterfactual Insights")
    st.write(rai_insights.counterfactual.get())
    
    st.subheader("Causal Analysis Insights")
    st.write(rai_insights.causal.get())

# Run the Streamlit app
if __name__ == "__main__":
    app()
