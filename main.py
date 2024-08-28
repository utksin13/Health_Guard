import os
import pickle
import streamlit as st # type: ignore
from streamlit_option_menu import option_menu # type: ignore

#set page config
st.set_page_config(page_title="Health Guard",layout="wide")

#setting the working directory

working_dir=os.path.dirname(os.path.abspath(__file__))
#loading the save models
diabetes_model=pickle.load(open(f'{working_dir}/diabetes_model.sav','rb'))

#sidebar for navigation

with st.sidebar:
    selected=option_menu('Multiple Disease Prediction System',['Diabetes Prediction','Heart Disease prediction'],menu_icon='hospital-fill',icons=['activity','heart'],default_index=0)

#diabetes prediction page

if selected == 'Diabetes Prediction':
    #page title
    st.title('Diabetes Prediction Using ML')
    #setting up the numebr of column with name
    col1,col2,col3=st.columns(3)

    with col1:
        pregnancies=st.text_input('Number of Pregnrncies')
    with col2:
        glucose=st.text_input('clucose level')
    with col3:
        bloodpressure=st.text_input('blood pressure value')
    with col1:
        skinthickness=st.text_input('skin thickness valur')
    with col2:
        insolin=st.text_input('insoliun value')
    with col3:
        bmi=st.text_input('bmi value')
    with col1:
        diabetespedigreefunction=st.text_input('diabatese pedigree value')
    with col2:
        ageofperson=st.text_input('age of person')
    
 
    #code for pridition
    diab_diagnosis=''

    #creation of button
    if st.button('Diabetes Test Result'):
        user_input=[pregnancies,glucose,bloodpressure,skinthickness,insolin,bmi,diabetespedigreefunction,ageofperson]
        user_input=[float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0]==1:
            diab_diagnosis='the person i diabetic'
        else:
            diab_diagnosis='the person is not diabetic'
    st.success(diab_diagnosis)
