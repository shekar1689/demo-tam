import streamlit as st
import numpy as np
import pandas as pd
# import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import contractions
from num2words import num2words
# from gensim.models import Word2Vec
import re
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from utilsfordemo import *

# Define the app title
st.title('TCP Prediction App')

# Add an upload file widget
uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)
    
    # Display the uploaded DataFrame
    st.write(df)
    
    output = testModel(df)

    import io

    # Save the updated DataFrame to Excel
    excel_buffer = io.BytesIO()
    output.to_excel(excel_buffer, index=False)
    excel_data = excel_buffer.getvalue()
    st.download_button(
        label="Download Predictions",
        data=excel_data,
        file_name='predictions.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


