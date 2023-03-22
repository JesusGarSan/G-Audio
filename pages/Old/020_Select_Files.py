import streamlit as st
import numpy as np
import pandas as pd

if 'uploaded_files' not in st.session_state:
    st.write('Please upload a file before trying to select one')
    st.stop()


#---------------------------------
if 'chosen_file' not in st.session_state:
    st.session_state['chosen_file'] = [False] *len(st.session_state['uploaded_files'])

if 'hydrophone_index' not in st.session_state:
    st.session_state['hydrophone_index'] = [1] *len(st.session_state['uploaded_files'])  
if 'BITS' not in st.session_state:
    st.session_state['BITS'] = [24] *len(st.session_state['uploaded_files'])
if 'calibration_parameters' not in st.session_state:
    st.session_state['calibration_parameters'] = [False] *len(st.session_state['uploaded_files'])
#----------------------------------

st.write('## Select the file you want to analyze:')

col1, col2, col3 = st.columns(3)

col1.write('Audio Name')
col2.write('Bits')
col3.write('Hydrophone')

bits=[12,24,32]
hydrophones=['00009', '00010']

for i, file in enumerate(st.session_state['uploaded_files']):
    col1.write('')
    BITS = col2.selectbox('', options = bits, index=bits.index(st.session_state['BITS'][i]), key=i+10e3)
    hydrophone = col3.selectbox('', options=hydrophones, index=st.session_state['hydrophone_index'][i], key=i+20e3)
    
    
    st.session_state['hydrophone_index'][i]=hydrophones.index(hydrophone)
    st.session_state['BITS'][i]=BITS

    if(col1.checkbox(f"{file.name}", key=i, value = st.session_state['chosen_file'][i])):
        st.session_state['chosen_file'][i]=True
        if hydrophone =='00009':
                GAIN= 1.29  #dB
                SH = -184.8 #dB re 1V/uPa
        if hydrophone =='00010':
                GAIN= 1.28  #dB
                SH = -184.9 #dB re 1V/uPa

        st.session_state['calibration_parameters'][i] = {
            'hydrophone': hydrophone,
            'BITS': BITS,
            'GAIN': GAIN,
            'SH'  : SH
        }

    else:
        st.session_state['chosen_file'][i]=False
        st.session_state['calibration_parameters'][i]=False


#col4.write(f"File {i+1} selected.")


#col4.write(st.session_state['calibration_parameters'])

st.session_state['chosen_file_indexes'] = np.where(np.array(st.session_state['chosen_file'])==True)[0].tolist()



