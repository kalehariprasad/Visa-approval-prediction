import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import streamlit as st
import mlflow
from pathlib import Path
from app.logger import logging
from app.exception import CustomException
from app.utils import streamlitutilies



utilities = streamlitutilies()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
   raise CustomException("No MLFLOW_TRACKING_URI in environment", sys)
mlflow.set_tracking_uri(tracking_uri)

curr_dir = Path(__file__).resolve()
home_dir = curr_dir.parents[2]
models_folder = home_dir / 'models'
reports_dir = home_dir / 'app' / 'reports'

prprocessor_path = models_folder /'preprocessor.pkl'
model_info_path = reports_dir / 'model_run_info.json'
preprocessor =utilities.load_object(prprocessor_path)

logging.info('Model info Loaded for model inference')
model_info = utilities.load_model_info(model_info_path)
model = utilities.load_model_for_inference(model_info)



st.title('Visa Approval Prediction')


# Input fields
continent = st.selectbox("Continent", ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"])
education_of_employee = st.selectbox("Education of Employee", ["Bachelor's", "Master's", "Doctorate", "High School", "Other"])
has_job_experience = st.radio("Has Job Experience?", ['Y', 'N'])
requires_job_training = st.radio("Requires Job Training?", ['Y', 'N'])
no_of_employees = st.number_input("Number of Employees", min_value=1, value=100)
region_of_employment = st.selectbox("Region of Employment", ["South", "North", "East", "West", "Central", "Other"])
prevailing_wage = st.number_input("Prevailing Wage", min_value=0.0, value=50000.0)
unit_of_wage = st.selectbox("Unit of Wage", ["Hour", "Week", "Month", "Year"])
full_time_position = st.radio("Full-time Position?", ['Y', 'N'])
company_age = st.number_input("Company Age", min_value=0, value=5)

if st.button("Predict Visa Approval"):
    try:
        input_data = {
            "continent": continent,
            "education_of_employee": education_of_employee,
            "has_job_experience": has_job_experience,
            "requires_job_training": requires_job_training,
            "no_of_employees": no_of_employees,
            "region_of_employment": region_of_employment,
            "prevailing_wage": prevailing_wage,
            "unit_of_wage": unit_of_wage,
            "full_time_position": full_time_position,
            "company_age": company_age
        }
        input_df = pd.DataFrame([input_data])
        input_array = preprocessor.transform(input_df)
        prediction = model.predict(input_array)
        if prediction[0] == 0:
            st.error("Visa Status: Denied")
        else:
            st.success("Visa Status: Approved")
    except Exception as e:
        logging.info(f"Prediction failed: {e}")
        st.error(f"An error occurred during prediction: {e}")
        raise CustomException(e,sys)




