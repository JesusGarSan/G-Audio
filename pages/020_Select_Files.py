import streamlit as st
import numpy as np
import pandas as pd


from audio import audio

if 'uploaded_files' not in st.session_state:
    st.write('Please upload a file before trying to select one')
    st.stop()


#-------------------GLOBAL VARIABLES INITIALIZATION-------------
if 'chosen_file' not in st.session_state:
    st.session_state['chosen_file'] = [False] *len(st.session_state['uploaded_files'])
if 'audio_name' not in st.session_state:
    st.session_state['audio_name'] = [''] *len(st.session_state['uploaded_files'])  
if 'hydrophone_index' not in st.session_state:
    st.session_state['hydrophone_index'] = [1] *len(st.session_state['uploaded_files'])  
if 'BITS' not in st.session_state:
    st.session_state['BITS'] = [24] *len(st.session_state['uploaded_files'])
if 'ref_audio' not in st.session_state:
    st.session_state['ref_audio'] = [None] *len(st.session_state['uploaded_files'])
if 'background_noise' not in st.session_state:
    st.session_state['background_noise'] = [None] *len(st.session_state['uploaded_files'])
if 'audio_group' not in st.session_state:
    st.session_state['audio_group'] = [None] *len(st.session_state['uploaded_files'])
if 'attributes' not in st.session_state:
    st.session_state['attributes'] = [False] *len(st.session_state['uploaded_files'])


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
file_size=[]
ref_audio=[]
background_noise=[]
audio_group=[]


with st.form(key='data'):
    for i, file in enumerate(st.session_state['uploaded_files']):

        selected.append(st.session_state['chosen_file'][i])
        if st.session_state['audio_name'][i] == '': names.append(file.name)
        else: names.append(st.session_state['audio_name'][i])
        BITS.append(str(st.session_state['BITS'][i]))
        hydrophone.append(hydrophones[st.session_state['hydrophone_index'][i]])
        #file_size.append(len(file.getvalue())/1e6)

        ref_audio.append(st.session_state['ref_audio'][i])
        background_noise.append(st.session_state['background_noise'][i])
        audio_group.append(st.session_state['audio_group'][i])


        data.append([selected[i], names[i], BITS[i], hydrophone[i], ref_audio[i], background_noise[i], audio_group[i] ])
    
    #columns=['Selected', 'File Name', 'Bits', 'Hydrophone', 'File Size (MB)']
    columns=['Selected', 'File Name', 'Bits', 'Hydrophone', 'Reference Audio', 'Background Noise', 'Audio Group']

    df=pd.DataFrame(data, columns=columns)
    df.Hydrophone=pd.Categorical(df.Hydrophone, hydrophones)
    df.Bits=pd.Categorical(df.Bits, bits)


    df['Reference Audio']=pd.Categorical(df['Reference Audio'], names)
    df['Background Noise']=pd.Categorical(df['Background Noise'], names)


    #This causes severe issues with the name fo the files, swithcing the around. I'll wait 
    #df=df.sort_values('File Name')
    #df=df.reset_index(drop=True) 
    col1, col2, col3, col4 =st.columns(4)
    autofill_button = col1.form_submit_button(label='Autofill audio groups & Hydrophone')
    select_all_button = col2.form_submit_button(label='Select all')

    edited_df = st.experimental_data_editor(df)


    submit_button = st.form_submit_button(label='Save')

#-----------Auto-Populate audio Groups:------------------------
if select_all_button:
    st.session_state['chosen_file'] = [True] *len(st.session_state['uploaded_files'])
    st.experimental_rerun()

#----------------------------------------------------------------



#-----------Auto-Populate audio Groups:------------------------
if autofill_button:
    for i, file in enumerate(st.session_state['uploaded_files']):
        #name = 'MOTRIL2_POS1_WN_CONT_4_H09.wav'
        name = st.session_state['audio_name'][i]

        attributes=[]
        while True:
            try:
                index= name.index('_')
            except:
                try:
                    index= name.index('.')
                except:
                    break
                
            attributes.append(name[0:index])
            name=name[index+1:]

        #tag=f"{attributes[0]}_{attributes[1]}" #No agrupamos conforme a hidrófono para el ruido de fondo (TEMPPORAL)
        if attributes[2]=='BN':
            tag=None #No agrupamos los ruidos de fondo (TEMPORAL)
        else:
            tag=f"{attributes[0]}_{attributes[1]}_{attributes[5]}"
            if attributes[5]=='H09': st.session_state['hydrophone_index'][i] = 0
            if attributes[5]=='H10': st.session_state['hydrophone_index'][i] = 1

        st.session_state['attributes'][i]=attributes
        st.session_state['audio_group'][i] = tag

    attributes= st.session_state['attributes']
    #attributes[i][0:2]
    for i, file in enumerate(st.session_state['uploaded_files']):
        if attributes[i][2]=='WN' and attributes[i][3]=='CONT':    
            for j, file_2 in enumerate(st.session_state['uploaded_files']):
                if attributes[j][2]=='BN' and attributes[i][0:2]==attributes[j][0:2]:
                    st.session_state['background_noise'][i] = st.session_state['audio_name'][j] 


    
    st.experimental_rerun()

#----------------------------------------------------------------






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





    


if submit_button:
    for i in range(len( edited_df.iloc[:] )):
        st.session_state['chosen_file'][i] = edited_df.iloc[i]['Selected']
        st.session_state['audio_name'][i] = edited_df.iloc[i]['File Name']
        st.session_state['BITS'][i] = edited_df.iloc[i]['Bits']
        st.session_state['hydrophone_index'][i] = hydrophones.index(edited_df.iloc[i]['Hydrophone'])
        st.session_state['ref_audio'][i] = edited_df.iloc[i]['Reference Audio']
        st.session_state['background_noise'][i] = edited_df.iloc[i]['Background Noise']
        st.session_state['audio_group'][i] = edited_df.iloc[i]['Audio Group']


    st.markdown('Saving...')

    st.session_state['chosen_file_indexes'] = np.where(np.array(st.session_state['chosen_file'])==True)[0].tolist()
    st.session_state['to_plot'] = [True] * len(st.session_state['chosen_file_indexes'])
    
    if len(st.session_state['chosen_file_indexes'])==1:
        uploaded_file=st.session_state['uploaded_files'][0]
        audio_object = audio(st.session_state['audio_name'][st.session_state['chosen_file_indexes'][0]], uploaded_file, st.session_state['calibration_parameters'][st.session_state['chosen_file_indexes'][0]])
        st.session_state['audio_object'] = [audio_object]

    else:
        audio_object=[0]*len(st.session_state['chosen_file_indexes'])
        for i,index in enumerate(st.session_state['chosen_file_indexes']):
            audio_object[i] = audio(st.session_state['audio_name'][index], 
                                    st.session_state['uploaded_files'][index], 
                                    st.session_state['calibration_parameters'][index], 
                                    st.session_state['ref_audio'][index], 
                                    st.session_state['background_noise'][index], 
                                    st.session_state['audio_group'][index])

        st.session_state['audio_object'] = audio_object

    # Hacemos esto para descargar toda  la memoria de los archivos subidos. (pero manteniendo la longitud del número de archivos subidos)
    # st.session_state['uploaded_files'] = range(len(st.session_state['uploaded_files']))
    st.experimental_rerun()
