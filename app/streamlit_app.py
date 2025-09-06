import streamlit as st
import requests


st.set_page_config(layout="wide")
def create_selector(label, options_str, mapping=None):
    options = [opt.strip() for opt in options_str.split(',')]
    options = [int(opt) if opt.isdigit() else opt for opt in options if opt and opt.upper() != 'NULL']
    if mapping:
        options = [mapping.get(opt, opt) for opt in options]
    return st.selectbox(label, options)

def create_selector_mappings(label, options, mapping):
    selected_option = st.selectbox(label, options)
    return mapping.get(selected_option, selected_option)

def create_integer_input(label, min_value=None, max_value=None, value=0):
    return st.number_input(label, min_value=min_value, max_value=max_value, value=int(value))

def create_float_input(label, min_value=None, max_value=None, value=0.0, format="%.2f"):
    return st.number_input(label, min_value=min_value, max_value=max_value, value=value, format=format)

def create_checkbox(label, key=None):
    return st.checkbox(label, key=key)


product_mapping = {'Personal loan': 1, 'Mortgage loan': 2, 'Line of credit': 7}  
sex_mapping = {'Male': 'M', 'Female': 'F', 'Not declared': 'N'}  
education_mapping = {'Primary': 1, 'Secondary': 2, 'Undergraduate': 3, 'Graduate': 4, 'Doctorate': 5}
marital_status_mapping = {'Single': 1, 'Married': 2, 'Divorced': 3, 'Widowed': 4, 'Separated': 5, 'Common-law': 6, 'Other': 7}
occupation_type_mapping = {'Employee': 1, 'Entrepreneur': 2, 'Self-employed': 3, 'Student': 4, 'Retired': 5}
residence_type_mapping = {'Own': 1, 'Rent': 2, 'Family': 3, 'Mortgage': 4, 'Other': 5}
professional_code_mapping = {'Engineering': 0, 'Medicine': 1, 'Teaching': 2, 'Arts': 3, 'Computer Science': 4, 'Law': 5, 'Accounting': 6, 'Marketing': 7, 'Sales': 8, 'Construction': 9, 'Administration': 10, 'Agriculture': 11, 'Research': 12, 'Human Resources': 13, 'Consulting': 14, 'Tourism': 15, 'Culinary': 16, 'Sports': 17, 'Other': 18}


brazil_states_mapping = {
 'Acre': 'AC',
 'Alagoas': 'AL',
 'Amapá': 'AP',
 'Amazonas': 'AM',
 'Bahia': 'BA',
 'Ceará': 'CE',
 'Distrito Federal': 'DF',
 'Espírito Santo': 'ES',
 'Goiás': 'GO',
 'Maranhão': 'MA',
 'Mato Grosso': 'MT',
 'Mato Grosso do Sul': 'MS',
 'Minas Gerais': 'MG',
 'Pará': 'PA',
 'Paraíba': 'PB',
 'Paraná': 'PR',
 'Pernambuco': 'PE',
 'Piauí': 'PI',
 'Rio de Janeiro': 'RJ',
 'Rio Grande do Norte': 'RN',
 'Rio Grande do Sul': 'RS',
 'Rondônia': 'RO',
 'Roraima': 'RR',
 'Santa Catarina': 'SC',
 'São Paulo': 'SP',
 'Sergipe': 'SE',
 'Tocantins': 'TO'
}

st.title(':violet[Bora Bora Bank - Loan Approval Portal]')
st.header('Lets embark on this journey together towards your financial goals!', divider='rainbow')


with st.form("credit_app_form"):
    col1, col2, col3= st.columns([1, 1, 1],gap="large" )  # Ajuste de proporciones para mejor visualización

    with col1:
        st.subheader('Personal and Residential Information')
        
        # Información personal
        name = st.text_input('Full Name')
        age = create_integer_input('Age')
        sex = create_selector_mappings('Sex', list(sex_mapping.keys()), sex_mapping)
        state_of_birth = create_selector_mappings('State of Birth', list(brazil_states_mapping.keys()), brazil_states_mapping)
        city_of_birth = st.text_input('City of Birth')
        marital_status = create_selector_mappings('Marital Status', list(marital_status_mapping.keys()), marital_status_mapping)
        quant_dependants = create_integer_input('Quantity of Dependents')
        education_level = create_selector_mappings('Education Level', list(education_mapping.keys()), education_mapping)
        
        # Información residencial
        residential_state = create_selector_mappings('Residential State', list(brazil_states_mapping.keys()), brazil_states_mapping)
        residential_city = st.text_input('Residential City')
        residential_borough = st.text_input('Residential Borough')
        flag_residential_phone = create_checkbox('Has Residential Phone')
        residence_type = create_selector_mappings('Residence Type', list(residence_type_mapping.keys()), residence_type_mapping)
        months_in_residence = create_integer_input('Months in Residence',min_value=0)

    with col2:
        st.subheader('Financial Information')
        
        # Información financiera
        personal_monthly_income = create_float_input('Personal Monthly Income')
        other_incomes = create_float_input('Other Incomes')
        payment_day = create_selector('Payment Day', '1,5,10,15,20,25')
        quant_cars = create_integer_input('Quantity of Cars')
        quant_banking_accounts = create_selector('Quantity of Banking Accounts', '0,1,2')
        quant_additional_cards = create_selector('Quantity of Additional Cards', '0,1,2,3,4,5')
        personal_assets_value = create_float_input('Personal assets value)')
        application_submission_type = create_selector('Application Submission Type', 'Web, Carga, Other')

    with col3:
        st.subheader('Credit Card and Professional Information')
        
        # Información de tarjetas de crédito
        flag_visa = create_checkbox('Has Visa')
        flag_mastercard = create_checkbox('Has MasterCard')
        flag_diners = create_checkbox('Has Diners')
        flag_american_express = create_checkbox('Has American Express')
        flag_other_cards = create_checkbox('Other Cards')
        
        # Información profesional
        company = create_checkbox('Provided Company Name')
        professional_state = create_selector_mappings('Professional State', list(brazil_states_mapping.keys()), brazil_states_mapping)
        flag_professional_phone = create_checkbox('Has Professional Phone')
        months_in_the_job = create_integer_input('Months in Current Job')
        profession_code = create_selector_mappings('Profession Code', list(professional_code_mapping.keys()), professional_code_mapping)
        occupation_type = create_selector_mappings('Occupation Type', list(occupation_type_mapping.keys()), occupation_type_mapping)
        mate_profession_code = create_selector_mappings('Mate Profession Code', list(professional_code_mapping.keys()), professional_code_mapping)
        mate_education_level = create_selector_mappings('Mate Education Level', list(education_mapping.keys()), education_mapping)
        product = create_selector_mappings('Product', list(product_mapping.keys()), product_mapping)
    
    submit_button = st.form_submit_button(label='📝 Submit Application', help='Click to submit your loan application')

debug_mode = st.checkbox('Debug Mode')

if submit_button:
    form_data = {
        "ID_CLIENT": 1,  # Asumo que este valor es estático, ajustarlo según sea necesario
        "CLERK_TYPE": "C",  # Valor estático, ajustar según sea necesario
        "PAYMENT_DAY": int(payment_day),
        "APPLICATION_SUBMISSION_TYPE": application_submission_type,
        "QUANT_ADDITIONAL_CARDS": int(quant_additional_cards),
        "POSTAL_ADDRESS_TYPE": 1,  # Valor estático, ajustar según sea necesario
        "SEX": sex,
        "MARITAL_STATUS": int(marital_status),
        "QUANT_DEPENDANTS": quant_dependants,
        "EDUCATION_LEVEL": education_level,
        "STATE_OF_BIRTH": state_of_birth,
        "CITY_OF_BIRTH": city_of_birth,
        "NACIONALITY": 1,  # Valor estático, ajustar según sea necesario
        "RESIDENCIAL_STATE": residential_state,
        "RESIDENCIAL_CITY": residential_city,
        "RESIDENCIAL_BOROUGH": residential_borough,
        "FLAG_RESIDENCIAL_PHONE": "Y" if flag_residential_phone else "N",
        "RESIDENCIAL_PHONE_AREA_CODE": "20",  # Valor estático, ajustar según sea necesario
        "RESIDENCE_TYPE": residence_type,
        "MONTHS_IN_RESIDENCE": months_in_residence,
        "FLAG_MOBILE_PHONE": "N",  # Asumiendo que no capturas esta información, ajustar según sea necesario
        "FLAG_EMAIL": 1,  # Valor estático, ajustar según sea necesario
        "PERSONAL_MONTHLY_INCOME": personal_monthly_income,
        "OTHER_INCOMES": 0,
        "FLAG_VISA": 1 if flag_visa else 0,
        "FLAG_MASTERCARD": 1 if flag_mastercard else 0,
        "FLAG_DINERS": 1 if flag_diners else 0,
        "FLAG_AMERICAN_EXPRESS": 1 if flag_american_express else 0,
        "FLAG_OTHER_CARDS": int(flag_other_cards),
        "QUANT_BANKING_ACCOUNTS": int(quant_banking_accounts),
        "QUANT_SPECIAL_BANKING_ACCOUNTS": 0,  # Valor estático, ajustar según sea necesario
        "PERSONAL_ASSETS_VALUE": 0.0,
        "QUANT_CARS": quant_cars,
        "COMPANY": "Y",  # Valor estático, ajustar según sea necesario
        "PROFESSIONAL_STATE": professional_state,
        "PROFESSIONAL_CITY": "",  # Asumiendo que no capturas esta información, ajustar según sea necesario
        "PROFESSIONAL_BOROUGH": "",  # Asumiendo que no capturas esta información, ajustar según sea necesario
        "FLAG_PROFESSIONAL_PHONE": "N",  # Asumiendo que no capturas esta información, ajustar según sea necesario
        "PROFESSIONAL_PHONE_AREA_CODE": " ",  # Valor estático, ajustar según sea necesario
        "MONTHS_IN_THE_JOB": months_in_the_job,
        "PROFESSION_CODE": profession_code,
        "OCCUPATION_TYPE": occupation_type,
        "MATE_PROFESSION_CODE": mate_profession_code,
        "FLAG_HOME_ADDRESS_DOCUMENT": 0,  # Valor estático, ajustar según sea necesario
        "FLAG_RG": 0,  # Valor estático, ajustar según sea necesario
        "FLAG_CPF": 0,  # Valor estático, ajustar según sea necesario
        "FLAG_INCOME_PROOF": 0,  # Valor estático, ajustar según sea necesario
        "PRODUCT": int(product),
        "FLAG_ACSP_RECORD": "N",  # Asumiendo que no capturas esta información, ajustar según sea necesario
        "AGE": age,
        "RESIDENCIAL_ZIP_3": 230,  # Valor estático, ajustar según sea necesario
        "PROFESSIONAL_ZIP_3": 230  # Valor estático, ajustar según sea necesario
    }
    

    if debug_mode and submit_button:
        st.subheader("Form Data Summary:")
        st.write(form_data)
    
    for key, value in form_data.items():
        if isinstance(value, bool):
            form_data[key] = 1 if value else 0
       
    response = requests.post("http://api_cr:5001/prediction", json=form_data)
    if response.status_code == 200:
        prediction = response.json()
        if debug_mode:
            st.write(prediction)
        st.success("Application submitted successfully!")
        st.success(f"Application Status: {prediction}")
    else:
        st.error("An error occurred while submitting the application.")
    if  debug_mode and response.status_code == 422:
        prediction = response.json()
        st.error("An error occurred while submitting the application. 422 Unprocessable Content")
        st.error(f"Application Status: {prediction}")
    
