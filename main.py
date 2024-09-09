import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib

model = tf.keras.models.load_model('94.69_accuracy_model.keras')
scaler = joblib.load('standard_scaler.pkl')

st.title('Credit Risk Prediction App')
st.subheader('Please enter the details below')

loan_grade_data = {
    'loan_grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'rate_average': [7.328423, 10.995756, 13.464579, 15.360698, 17.008409, 18.609159, 20.251525]
}

loan_grade_df = pd.DataFrame(loan_grade_data)

def categorize_loan_rate(rate, grade_df):
    # Hitung selisih absolut antara rate dan setiap rata-rata
    grade_df['difference'] = abs(grade_df['rate_average'] - rate)
    # Temukan kategori dengan selisih terkecil
    closest_grade = grade_df.loc[grade_df['difference'].idxmin(), 'loan_grade']
    return closest_grade

# Create a form
with st.form('my_form'):
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Place input widgets inside the form
    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, key='person_age')
    
    with col2:
        income = st.number_input('Income (Rupiah)', key='person_income')
        
    col1, col2 = st.columns(2)
    
    with col1:
        ownership = st.radio('Person Home Ownership', ['MORTGAGE', 'RENT', 'OWN', 'OTHER'], key='person_home_ownership')
    
    with col2:
      loan_intent = st.radio('Loan Intention', ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'], key='loan_intent')
      
    emp_length = st.slider('Employment Length', min_value=0, max_value=100, key='person_emp_length')
    loan_amnt = st.number_input('Loan Amount', min_value=0.01, key='loan_amnt')
    loan_int_rate = st.number_input('Interest Rate', min_value=0.0, max_value=100.0, key='loan_int_rate')
    
    cb_person_default_on_file = st.radio('Default on File', ['Y', 'N'], key='cb_person_default_on_file')
        
    
    cb_person_cred_hist_length = st.number_input('Credit History Length', min_value=0, key='cb_person_cred_hist_length')
    
    # Add a submit button
    submit_button = st.form_submit_button(label='Submit')
    

    # Check if the form was submitted
    if submit_button:
        loan_grade = categorize_loan_rate(loan_int_rate, loan_grade_df)
        loan_percent_income = income / loan_amnt
        # Prepare the input data
        input_data = pd.DataFrame([[age, income, ownership, emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length]], columns=['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length'])
        
        # Convert categorical features to one-hot encoding
        input_data_encoded = pd.get_dummies(input_data, columns=['person_home_ownership', 'loan_intent', 'loan_grade'])
                 
        input_data_encoded = input_data_encoded.replace({
          'cb_person_default_on_file': {
            'Y': 1,
            'N': 0
          }
        })
        
        input_data_encoded = pd.get_dummies(input_data_encoded, columns=['cb_person_default_on_file'])
        
        # Ensure the columns match with the model's input features
        # You may need to re-align the columns to match the model's training data
        expected_columns = [
            'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
            'loan_percent_income', 'cb_person_cred_hist_length',
            'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 'person_home_ownership_OWN',
            'person_home_ownership_RENT',
            'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
            'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
            'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E',
            'loan_grade_F', 'loan_grade_G',
            'cb_person_default_on_file_0', 'cb_person_default_on_file_1'
        ]
        
        # Align input_data_encoded to expected columns
        input_data_encoded = input_data_encoded.reindex(columns=expected_columns, fill_value=0)

        # st.dataframe(input_data_encoded)
        
        # Scale the input data
        scaled_data = scaler.transform(input_data_encoded)
        
        # Make predictions using the model
        prediction = model.predict(scaled_data)
        
        result = (prediction[0][0] > 0.5).astype(int)
        
        if result == 0:
            st.subheader('Bad Loan')
        else:
            st.subheader('Good Loan')
        
        # Display the prediction
        # st.write(f'Prediction: {(prediction[0][0] > 0.5).astype(int)}')
        