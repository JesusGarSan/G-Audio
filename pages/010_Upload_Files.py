import streamlit as st




st.title('Upload Files')
uploaded_files = st.file_uploader('Upload your .wav files here', accept_multiple_files = True)


if ('uploaded_files' not in st.session_state or st.session_state['uploaded_files']==[]) and uploaded_files!=[]:
    st.session_state['uploaded_files'] = uploaded_files

elif 'uploaded_files' in st.session_state and uploaded_files!=[]:
    col1, col2 = st.columns(2)
    if col1.button('Append to previously uploaded files'):
        for i in range(len(uploaded_files)):
            st.session_state['uploaded_files'].append(uploaded_files[i])
            st.session_state['chosen_file'].append( False )         
            st.session_state['hydrophone_index'].append( 1 )   
            st.session_state['BITS'].append( 24 )
            st.session_state['calibration_parameters'].append( False )
            st.session_state['audio_name'].append('')



            st.session_state['ref_audio'].append(None)
            st.session_state['background_noise'].append(None)
            st.session_state['audio_group'].append(None)
            st.session_state['attributes'].append(False)
            if 'BN_behaviour' in st.session_state:
                st.session_state['BN_behaviour'].append(0)
            

          #  attributes=[]
          #  while True:
          #      try:
          #          index= name.index('_')
          #      except:
          #          try:
          #              index= name.index('.')
          #          except:
          #              break
          #          
          #      attributes.append(name[0:index])
          #      name=name[index+1:]
#
          #  st.session_state['position'].append('')
          #  st.session_state['noise'].append('')
          #  st.session_state['iteration'].append('')

    if col2.button('Overwrite previously uploaded files'):

        st.session_state['uploaded_files'] = (uploaded_files)

        st.session_state['chosen_file'] = [False] *len(st.session_state['uploaded_files'])           
        st.session_state['hydrophone_index'] = [1] *len(st.session_state['uploaded_files'])    
        st.session_state['BITS'] = [24] *len(st.session_state['uploaded_files'])
        st.session_state['calibration_parameters'] = [False] *len(st.session_state['uploaded_files'])
        st.session_state['audio_name'] = [''] *len(st.session_state['uploaded_files'])


        st.session_state['campaign'] = [''] *len(st.session_state['uploaded_files'])
        st.session_state['position'] = [''] *len(st.session_state['uploaded_files'])
        st.session_state['noise'] = [''] *len(st.session_state['uploaded_files'])
        st.session_state['iteration'] = [''] *len(st.session_state['uploaded_files'])
        
        st.session_state['ref_audio'] = [None] *len(st.session_state['uploaded_files'])
        st.session_state['background_noise'] = [None] *len(st.session_state['uploaded_files'])
        st.session_state['audio_group'] = [None] *len(st.session_state['uploaded_files'])
        st.session_state['attributes'] = [False] *len(st.session_state['uploaded_files'])
        st.session_state['BN_behaviour'] = [0] *len(st.session_state['uploaded_files'])

