import streamlit as st
import numpy as np
import pandas as pd

if 'uploaded_files' not in st.session_state:
    st.write('Please upload a file before trying to select one')
    st.stop()


#---------------------------------
if 'chosen_file' not in st.session_state:
    st.session_state['chosen_file'] = [False] *len(st.session_state['uploaded_files'])
if 'audio_name' not in st.session_state:
    st.session_state['audio_name'] = [''] *len(st.session_state['uploaded_files'])  
if 'hydrophone_index' not in st.session_state:
    st.session_state['hydrophone_index'] = [1] *len(st.session_state['uploaded_files'])  
if 'BITS' not in st.session_state:
    st.session_state['BITS'] = [24] *len(st.session_state['uploaded_files'])
if 'calibration_parameters' not in st.session_state:
    st.session_state['calibration_parameters'] = [False] *len(st.session_state['uploaded_files'])
if 'to_plot' not in st.session_state:
    st.session_state['to_plot'] = [True] * len(st.session_state['uploaded_files'])
#----------------------------------

st.write('## Select the file you want to analyze:')


hydrophones=['00009', '00010']
bits=['12', '24', '32']

data=[]

selected=[]
names=[]
BITS=[]
hydrophone=[]

with st.form(key='data'):
    for i, file in enumerate(st.session_state['uploaded_files']):

        if st.session_state['audio_name'][i] == '': names.append(file.name)
        else: names.append(st.session_state['audio_name'][i])

        BITS.append(str(st.session_state['BITS'][i]))
        hydrophone.append(hydrophones[st.session_state['hydrophone_index'][i]])

        selected.append(st.session_state['chosen_file'][i])
        data.append([selected[i], names[i], BITS[i], hydrophone[i]])

    df=pd.DataFrame(data, columns=['Selected', 'File Name', 'Bits', 'Hydrophone'])
    df.Hydrophone=pd.Categorical(df.Hydrophone, hydrophones)
    df.Bits=pd.Categorical(df.Bits, bits)
    #This causes severe issues with the name fo the files, swithcing the around. I'll wait 
    #df=df.sort_values('File Name')
    #df=df.reset_index(drop=True) 

    edited_df = st.experimental_data_editor(df)


    submit_button = st.form_submit_button(label='Save')
if submit_button:
    for i in range(len( edited_df.iloc[:] )):
        st.session_state['chosen_file'][i] = edited_df.iloc[i]['Selected']
        st.session_state['audio_name'][i] = edited_df.iloc[i]['File Name']
        st.session_state['BITS'][i] = edited_df.iloc[i]['Bits']
        st.session_state['hydrophone_index'][i] = hydrophones.index(edited_df.iloc[i]['Hydrophone'])
    
    
    st.experimental_rerun()



for i in range(len( edited_df.iloc[:] )):
    st.session_state['BITS'][i]=edited_df.iloc[i]['Bits']

    hydrophone = edited_df.iloc[i]['Hydrophone']
    if hydrophone =='00009': hydrophone_index=0 
    if hydrophone =='00010': hydrophone_index=1 
    st.session_state['hydrophone_index'][i] = hydrophone_index

    if edited_df.iloc[i]['Selected'] == False:
        st.session_state['chosen_file'][i]=False
        st.session_state['calibration_parameters'][i]=False
    else:
        st.session_state['audio_name'][i]= edited_df.iloc[i]['File Name']
        st.session_state['chosen_file'][i]=True
        if hydrophone =='00009':
                GAIN= 1.29  #dB
                SH = -184.8 #dB re 1V/uPa
        if hydrophone =='00010':
                GAIN= 1.28  #dB
                SH = -184.9 #dB re 1V/uPa

        st.session_state['calibration_parameters'][i] = {
            'hydrophone': hydrophone,
            'BITS': int(st.session_state['BITS'][i]),
            'GAIN': GAIN,
            'SH'  : SH
        }


st.session_state['chosen_file_indexes'] = np.where(np.array(st.session_state['chosen_file'])==True)[0].tolist()
st.session_state['to_plot'] = [True] * len(st.session_state['chosen_file_indexes'])




