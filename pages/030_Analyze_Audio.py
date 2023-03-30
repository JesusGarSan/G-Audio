import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import io
from datetime import datetime

from audio import audio

#--------------------------

def write_parameters(top_col3):
    file = open('STFT Parameters.txt', 'w')
    file.write(f"## Short Time Fourier Transform:\n")
    file.write(f"nperseg = {st.session_state['nperseg']}\n")
    file.write(f"noverlap = {st.session_state['noverlap']}\n")
    file.write(f"window_index = {st.session_state['window_index']}\n")
    file.write(f"\n")
    file.write(f"## Frequency Range:")
    file.write(f"[{st.session_state['low_freq']}, {st.session_state['high_freq']}] Hz")
    file.close()


    file = open('STFT Parameters.txt', 'r')
    return file

def display_save_parameters_button(file, position):

    file_name = f"STFT Parameters_{str(datetime.now())[:-7]}.txt"

    with file as f:
        position.download_button('Download STFT Parameters', f,file_name=file_name, use_container_width=True)  # Defaults to 'text/plain'
        file.close()
    

def parameters(audio_object):


    audio_object_list=audio_object
    audio_object=audio_object[0]

    if 'temp' not in st.session_state:
        st.session_state['temp'] = False




    col1, col2, top_col3 = st.columns(3)


    with st.form(key='parameters'):
        
        st.markdown('## Time Range:')
        col1, col2 = st.columns(2)

        low_time=col1.number_input('Lowest Time (s)', min_value = 0.0, max_value=audio_object.time[len(audio_object.time)-1], value=st.session_state['low_time'], step=1.0)
        #high_time=col2.number_input('Highest Time (s)', min_value = 0.0, max_value=audio_object.time[len(audio_object.time)-1], value=st.session_state['high_time'], step=1.0)
        high_time=col2.number_input('Highest Time (s)', min_value = 0.0, value=st.session_state['high_time'], step=1.0)



        st.session_state['low_time']=low_time
        st.session_state['high_time']=high_time


        st.markdown('## Short Time Fourier Transform:')
        col1, col2, col3 = st.columns(3)

        factor_nperseg =  col1.number_input('nperseg', min_value = 0, value=st.session_state['nperseg'])
        nperseg=factor_nperseg
        
        factor_noverlap =  col2.number_input('% noverlap', min_value = 0.0, max_value=90.0, value=st.session_state['noverlap'], step=1.0)
        noverlap= nperseg*(factor_noverlap/100)

        good_windows= ['blackman', 'hann', 'parzen', 'nuttall']
        window_name =     col3.selectbox('window', options = good_windows, index=0)
        window = f"scipy.signal.windows.{window_name}({nperseg},True)"


        for i in range(len(audio_object_list)):
            if not (low_time == 0 and high_time==0):
                audio_object_list[i].shorten_audio([low_time, high_time])
            
            f,t,Zxx = audio_object_list[i].audio_stft(nperseg=nperseg, noverlap=noverlap,
                                            window=eval(window),
                                            nfft=nperseg, scaling='psd')

        st.session_state['nperseg'] = factor_nperseg
        st.session_state['noverlap'] = factor_noverlap
        st.session_state['window_index'] = good_windows.index(window_name)

        #Frequency range filter
        st.markdown('## Frequency Range Filter:')
        col1, col2 = st.columns(2)

        low_freq=col1.number_input('Lowest Frequency (Hz)', min_value = f[0], max_value=st.session_state['high_freq'], value=st.session_state['low_freq'], step=1.0)
        high_freq=col2.number_input('Highest Frequency (Hz)', min_value = low_freq+1, max_value=f[len(f)-1], value=st.session_state['high_freq'], step=1.0)
        
        submit_button = st.form_submit_button(label='Run')


    if submit_button or st.session_state['STFT_ran']:

        # We create the file with the STFT parameters and allow the user to download it
        file = write_parameters(top_col3)
        display_save_parameters_button(file, top_col3)
        

        st.session_state['STFT_ran']=True

        for i in range(len(audio_object_list)):
            audio_object_list[i].filter_frequencies([low_freq, high_freq])

        st.session_state['low_freq']=low_freq
        st.session_state['high_freq']=high_freq


        stats = [audio_object.duration, audio_object.sample_rate]
        stats_labels = ['Audio duration (s)', 'Sample rate (Hz)']


        if len(f)>1:
            stats.append(f[1]-f[0])
            stats_labels.append('Frequency resolution (Hz):')
            
            stats.append([audio_object.freq_fft[0], audio_object.freq_fft[len(audio_object.freq_fft)-1] ])
            stats_labels.append('Frequency Range (Hz):')
        if len(t)>1:
            stats.append(t[1]-t[0])
            stats_labels.append('Time resolution (s):')

        df = pd.DataFrame(
            stats, stats_labels, columns=['Stats'])
        
        st.session_state['stats'] = df

        return audio_object

def spectogram(audio_object):
    # Spectogram representation
    #audio_object.spectogram()
    st.pyplot(audio_object.spectogram())
    

def avg_group(audio_object_list, Pts, BN=False):
    if BN==False:
        SPLs=[]
        errors=[]
        for i, audio_object in enumerate(audio_object_list):
                if not hasattr(audio_object, 'data_freq_SPL'):   audio_object.SPL_vs_freq(Pts=Pts, time_avg=1, plot=False)
                SPLs.append(audio_object.data_freq_SPL[1])
                errors.append(audio_object.data_freq_SPL[2])

        freq=audio_object.data_freq_SPL[0]
        avg=np.mean(SPLs, axis=0)


        std=np.std(SPLs, axis=0)
        error=std/np.sqrt(len(SPLs))

        return freq, avg, error

    if BN==True:
        st.markdown('DOING IT FOR BN')
        SPLs=[]
        errors=[]

        # we have to find the background noise object
        for i in range(len(audio_object_list)):
            for j, background_object in enumerate(st.session_state['audio_object']):
                if audio_object_list[i].background_noise == background_object.audio_name:
                    st.session_state['audio_object'][j].SPL_vs_freq(source=audio_object_list[i], method='TL', Pts=Pts, plot=False)
                    SPLs.append(st.session_state['audio_object'][j].data_freq_SPL_TL[1])
                    errors.append(st.session_state['audio_object'][j].data_freq_SPL_TL[2])
                    freq=st.session_state['audio_object'][j].data_freq_SPL_TL[0]
        
        avg=np.mean(SPLs, axis=0)

        std=np.std(SPLs, axis=0)
        error=std/np.sqrt(len(SPLs))


        len(freq)
        len(avg)
        len(error)
        return freq, avg, error


def SPL_vs_freq(audio_object, Pts, xlim=False, ylim=False):

    audio_object_list=audio_object
    audio_object=audio_object[0]

    #SPL vs freq representation
    fig, ax = plt.subplots(1, figsize=(14,6))
    fig2=False

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    
    #-----------Ploting with refrence to Background Noises-----------
    for i in range(len(audio_object_list)):
        if st.session_state['to_plot'][i]==True:

            if type(audio_object_list[i].background_noise)==str and (st.session_state['BN_behaviour'][i]==1 or st.session_state['BN_behaviour'][i]==2):
                audio_object_list[i].SPL_vs_freq(Pts=Pts)
            
            if type(audio_object_list[i].background_noise)==str and (st.session_state['BN_behaviour'][i]==0 or st.session_state['BN_behaviour'][i]==2):

                for j, object in enumerate(audio_object_list):
                    if audio_object_list[i].background_noise == object.audio_name:
                        audio_object_list[j].SPL_vs_freq(source=audio_object_list[i], method='TL', Pts=Pts,
                                                        legend=f"{audio_object_list[i].audio_name}-BN")

            elif type(audio_object_list[i].background_noise)!=str:
                audio_object_list[i].SPL_vs_freq(Pts=Pts)

    #-----------Ploting Audio Groups-----------

    audio_groups=[]
    for i in range(len(audio_object_list)):
        if st.session_state['to_plot'][i]==True:
            if type(audio_object_list[i].audio_group) == str:
                if audio_object_list[i].audio_group not in audio_groups:
                    audio_groups.append(audio_object_list[i].audio_group)
        
    if audio_groups!=[]:
        fig4, ax = plt.subplots(1, figsize=(14,6))
        plt.title('Audio Groups')

        for index, group in enumerate(audio_groups):
            sublist=[]
            for i in range(len(audio_object_list)):
                if audio_object_list[i].audio_group == group and st.session_state['to_plot'][i]==True:
                    sublist.append(audio_object_list[i])


            #SOLUCIÓN TMEPORAL A LA GESTION DE DATOS
            subtract_BN_in_groups=True
            if subtract_BN_in_groups:
                freq, avg, error=avg_group(sublist, Pts, BN=True)
                plt.plot(freq, avg, label=f"{group}-BN")
            else:          
                freq, avg, error=avg_group(sublist, Pts, BN=False)
                plt.plot(freq, avg, label=group)

            if xlim:
                plt.xlim(xlim[0], xlim[1])
            if ylim:
                plt.ylim(ylim[0], ylim[1])
            plt.fill_between(freq, avg-error, avg+error, alpha=0.5)
            plt.grid(True)
            plt.legend()

        st.pyplot(fig4)
    #-------------------------------------------



    #Plot de atenuación
    # -- TEMPORAL --
    #Si detecta que solo tiene que dibujar 2 lineas, calcula la atenuación automáticamente.
    #Detecta aquella con intensidad media mayor y la asigna como fuente
    temp=np.array(st.session_state['to_plot'])
    #temp
    #f len(temp[np.where(temp==True)])==2:
    #   idx0=np.where(temp==True)[0][0]
    #   idx1=np.where(temp==True)[0][1]

    #   st.markdown('the conditio')
    #   if audio_object_list[idx0].background_noise == audio_object_list[idx1].audio_name:
    #       st.markdown('the condition 1 has been met')
    #       audio_object_list[idx0].SPL_vs_freq(source=audio_object_list[idx1] , Pts=Pts)
    #       

    #   if audio_object_list[idx1].background_noise == audio_object_list[idx0].audio_name:
    #       st.markdown('the condition 2 has been met')
    #       audio_object_list[idx1].SPL_vs_freq(source=audio_object_list[idx0] , Pts=Pts)


    st.pyplot(fig)
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')

    img2=False
   #if len(temp[np.where(temp==True)])==2:
   #    idx0=np.where(temp==True)[0][0]
   #    idx1=np.where(temp==True)[0][1]
   #    fig2, ax = plt.subplots(1, figsize=(14,6))
   #
   #    #-----------------------------    
   #    if xlim:
   #        plt.xlim(xlim[0], xlim[1])
   #    #-----------------------------    
   #    if np.mean(audio_object_list[idx0].signal_fft) > np.mean(audio_object_list[idx1].signal_fft):
   #        legend=f"{audio_object_list[idx0].audio_name} - TL_{audio_object_list[idx1].audio_name}"
   #        audio_object_list[idx1].SPL_vs_freq(source=audio_object_list[idx0], method='Source-TL' , Pts=Pts, legend=legend)
   #    else:
   #        legend=f"{audio_object_list[idx1].audio_name} - TL_{audio_object_list[idx0].audio_name}"
   #        audio_object_list[idx0].SPL_vs_freq(source=audio_object_list[idx1], method='Source-TL' , Pts=Pts, legend=legend)

   #    st.pyplot(fig2)
   #    img2 = io.BytesIO()

    return img1, img2

def SPL_vs_time(audio_object, frequency, ponderation, xlim=False):
    audio_object_list=audio_object
    audio_object=audio_object[0]


    #SPL vs time representation
    fig, ax = plt.subplots(1, figsize=(14,6))
    #plt.title(audio_object.audio_name)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    
    for i in range(len(audio_object_list)):
        if st.session_state['to_plot'][i]==True:
            aux = audio_object_list[i].SPL_vs_time(frequency=frequency, ponderation=ponderation)
            if aux[0]==0 and len(aux)<2:
                st.write('Choose a higher ponderation. The current value is too low');
            elif aux[0]==1 and len(aux)<2:
                st.write('Choose a lower ponderation. The current value is too high')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)


def reverb_time(audio_object,RT, drop, frequency):
    #fig, ax = plt.subplots(1, figsize=(14,6))
#    plt.title(audio_object.audio_name)
    fig, RT_interpolated= audio_object.reverb_time_schroeder(RT=RT, drop=drop, frequency=frequency)
    if fig!=None:
        st.write(f"RT{st.session_state['RT']}: {RT_interpolated} s")
        plt.grid()
        st.pyplot(fig)
    else:
        st.write('The given signal does not drop enough dB')


#---------------------------

#st.title('G-Audio Processor')
#st.text('This is a web app to explore audio files')
#st.markdown('## This is a  **markdown**')

#st.sidebar.title('Upload File')
#uploaded_files = st.sidebar.file_uploader('Upload your .wav file', accept_multiple_files = True)


#------------------Initialization--------------------
if 'uploaded_files' not in st.session_state:
    st.write('Please upload and select the file you want to analyze')
    st.stop()
    
if 'chosen_file' not in st.session_state or st.session_state['chosen_file'] == [False] *len(st.session_state['uploaded_files']):
    st.write('Please select the file you want to analyze')
    st.stop()
#-----------------------------------------------------

if 'nperseg' not in st.session_state:
    st.session_state['nperseg'] = 4800
if 'noverlap' not in st.session_state:
    st.session_state['noverlap'] = 0.0
if 'window_index' not in st.session_state:
    st.session_state['window_index'] = 0
if 'low_time' not in st.session_state:
    st.session_state['low_time'] = 0.0
if 'high_time' not in st.session_state:
    st.session_state['high_time'] = 0.0
if 'low_freq' not in st.session_state:
    st.session_state['low_freq'] = 20.0
if 'high_freq' not in st.session_state:
    st.session_state['high_freq'] = 20000.0
if 'Pts' not in st.session_state:
    st.session_state['Pts'] = 100
if 'STFT_ran' not in st.session_state:
    st.session_state['STFT_ran'] = True
if 'BN_behaviour' not in st.session_state:
    st.session_state['BN_behaviour'] = [0] *len(st.session_state['uploaded_files'])

#---------------------------------------------------

uploaded_files = st.session_state['uploaded_files']

#-----------------------------------------------------

try:
    uploaded_file = st.session_state['uploaded_files'][st.session_state['chosen_file_indexes'][0]]
   

except:
    st.write('UPLOAD A FILE')
    uploaded_file=None

#---------------------------------------------------------

if uploaded_file:


    audio_object = st.session_state['audio_object']

   ##We create the audio Objects
   #if len(st.session_state['chosen_file_indexes'])==1:
   #    audio_object = audio(st.session_state['audio_name'][st.session_state['chosen_file_indexes'][0]], uploaded_file, st.session_state['calibration_parameters'][st.session_state['chosen_file_indexes'][0]])

   #else:
   #    audio_object=[0]*len(st.session_state['chosen_file_indexes'])
   #    for i,index in enumerate(st.session_state['chosen_file_indexes']):
   #        audio_object[i] = audio(st.session_state['audio_name'][index], 
   #                                uploaded_files[index], 
   #                                st.session_state['calibration_parameters'][index], 
   #                                st.session_state['ref_audio'][index], 
   #                                st.session_state['background_noise'][index], 
   #                                st.session_state['audio_group'][index])






    #Área de navegación

    options = st.sidebar.radio('Audio analysis', options=['Parameters', 'Spectogram', 'SPL vs. Frequency', 'SPL vs. Time', 'Reverb Time'])





    if options == 'Parameters':
        if len(st.session_state['chosen_file_indexes'])==1:
            st.write(f"Current Audio file: {audio_object[0].audio_name}")
        else:
            st.write(f"Current Audio files:")
            for i,index in enumerate(st.session_state['chosen_file_indexes']):
                st.write(f"{audio_object[i].audio_name}")


        if len(st.session_state['chosen_file_indexes'])==1 or True:
            parameters(audio_object)
            st.session_state['audio_object'] = audio_object

    elif options =='Spectogram':
        st.header('Spectogram')
        audio_object = st.session_state['audio_object'] 


        try:
            if(len(audio_object)>1):
                audio_object_list=audio_object
        except:
            audio_object_list=[audio_object]

        for i in range(len(audio_object_list)):
            spectogram(audio_object_list[i])

            st.audio(audio_object_list[i].signal, sample_rate = audio_object_list[i].sample_rate)
            image_name= f"Spectogram-{audio_object_list[i].audio_name}.png"
            image_name = st.text_input('Image name:', value=image_name)
            img = io.BytesIO()
            plt.savefig(img, format='png')
            save_button = st.download_button(
                        label='Download Image',
                        data=img,
                        file_name=image_name,
                        mime="image/png"
                        )

    elif options =='SPL vs. Frequency':
        st.header('SPL vs. Frequency')

        try:
            if(len(audio_object)>1):
                audio_object_list=audio_object
        except:
            audio_object_list=[audio_object]

    
        with st.form(key='SPL_vs_freq'):
            try:
                freq_fft=st.session_state['audio_object'].freq_fft
            except:
                freq_fft=st.session_state['audio_object'][0].freq_fft

        
            #xlim=[freq_fft[0], freq_fft[len(freq_fft)-1]]

            if 'SPL_vs_freq_xlim' not in st.session_state:
                st.session_state['SPL_vs_freq_xlim'] = [freq_fft[0], freq_fft[len(freq_fft)-1]]
            if 'SPL_vs_freq_ylim' not in st.session_state:
                st.session_state['SPL_vs_freq_ylim'] = [60, 120]

            st.session_state['Pts'] =  st.number_input('Number of Points. Select 0 for all', min_value = 0, value=st.session_state['Pts'])
            Pts = st.session_state['Pts']
            if Pts == 0: Pts = 'all'
            audio_object = st.session_state['audio_object'] 
            col1, col2 = st.columns(2)
            st.session_state["SPL_vs_freq_xlim"][0] = col1.number_input('Lowest frequency:', value=st.session_state["SPL_vs_freq_xlim"][0], step=freq_fft[1]-freq_fft[0])
            st.session_state["SPL_vs_freq_xlim"][1] = col2.number_input('Highest frequency:', value=st.session_state["SPL_vs_freq_xlim"][1], step=freq_fft[1]-freq_fft[0])
            
            st.session_state["SPL_vs_freq_ylim"][0] = col1.number_input('Lowest SPL:', value=st.session_state["SPL_vs_freq_ylim"][0])
            st.session_state["SPL_vs_freq_ylim"][1] = col2.number_input('Highest SPL:', value=st.session_state["SPL_vs_freq_ylim"][1])



            #---------------------Selection Table----------------
            plot_y_n=[]
            name=[]
            BN_options=['Apply', 'Not Apply', 'Plot Both']
            BN=[]

            #Apply: Plotea la curva - su ruido de fondo

            data=[]
            for i in range(len(st.session_state['chosen_file_indexes'])):
                plot_y_n.append(True)
                name.append(audio_object[i].audio_name)
                BN.append('Apply')

                data.append([ plot_y_n[i], name[i], BN[i] ])

            columns=['Plot?', 'Audio Name', 'Subtract BN']
            df=pd.DataFrame(data, columns=columns)
            df['Subtract BN']=pd.Categorical(df['Subtract BN'], BN_options)
            edited_df = st.experimental_data_editor(df)
            #------------------------------------------------------


            col1, col2 = st.columns(2)
            col2col1, col2col2, col2col3, col2col4 = col2.columns(4) 
            submit_button = col2.form_submit_button(label='Run', use_container_width=True)



            show_buttons=False
            if submit_button:
                for i in range(len( edited_df.iloc[:] )):
                    st.session_state['to_plot'][i] = edited_df.iloc[i]['Plot?']
                    st.session_state['BN_behaviour'][i] = BN_options.index(edited_df.iloc[i]['Subtract BN'])

                img1, img2 = SPL_vs_freq(audio_object, Pts, xlim=st.session_state["SPL_vs_freq_xlim"], ylim=st.session_state["SPL_vs_freq_ylim"])
                show_buttons=True



        if show_buttons:
            #Download the image
            image_name= f"SPL_vs_freq-{st.session_state['Pts']}Pts-Freq[{st.session_state['SPL_vs_freq_xlim'][0]},{st.session_state['SPL_vs_freq_xlim'][1]}]-SPL[{st.session_state['SPL_vs_freq_ylim'][0]},{st.session_state['SPL_vs_freq_ylim'][1]}].png"
            image_name = st.text_input('Image name:', value=image_name)
            #img = io.BytesIO()
            #plt.savefig(img, format='png')
            save_button = st.download_button(
                        label='Download Image',
                        data=img1,
                        file_name=image_name,
                        mime="image/png"
                        )
            if img2:
                image_name= f"Source-TL_freq-{st.session_state['Pts']}Pts-Freq[{st.session_state['SPL_vs_freq_xlim'][0]},{st.session_state['SPL_vs_freq_xlim'][1]}]-SPL[{st.session_state['SPL_vs_freq_ylim'][0]},{st.session_state['SPL_vs_freq_ylim'][1]}].png"
                image_name = st.text_input('2nd Image name:', value=image_name, key='img2')
                img2 = io.BytesIO()
                plt.savefig(img2, format='png')
                save_button = st.download_button(
                            label='Download Image',
                            data=img2,
                            file_name=image_name,
                            mime="image/png"
                            )
        

    elif options =='SPL vs. Time':
        st.header('SPL vs. Time')

        freq_resolution = st.session_state['stats']['Stats'][2]

        try:
            freq_fft=st.session_state['audio_object'].freq_fft
            time_fft=st.session_state['audio_object'].time_fft

        except:
            freq_fft=st.session_state['audio_object'][0].freq_fft
            time_fft=st.session_state['audio_object'][0].time_fft

        if 'SPL_vs_time_xlim' not in st.session_state:
            st.session_state['SPL_vs_time_xlim'] = [time_fft[0], time_fft[len(time_fft)-1]]

        
        #freq_fft=st.session_state['audio_object'].freq_fft

        if 'ponderation' not in st.session_state:
            st.session_state['ponderation'] = 0.125
        if 'choice' not in st.session_state:
            st.session_state['choice'] = False
        if 'frequency' not in st.session_state:
            st.session_state['frequency'] = False



        choices=['All', 'Choose frequency:']
        st.session_state['choice']   =  st.checkbox('Choose Frequency', value=st.session_state['choice'])

        with st.form('SPL_vs_time'):
            col1, col2 = st.columns(2)
            st.session_state['ponderation'] =  col1.number_input('Ponderation', min_value = 0.001, value=st.session_state['ponderation'], format="%0.3f")
            if not st.session_state['choice']:
                st.session_state['frequency'] = False
            else:
                if st.session_state['frequency']==False: st.session_state['frequency'] = freq_fft[0]
                st.session_state['frequency']   =  col2.number_input('Frequency', min_value = freq_fft[0], max_value=freq_fft[len(freq_fft)-1], value=st.session_state['frequency'], step=freq_resolution*1.0)
            #if Pts == 0: Pts = 'all'
            audio_object = st.session_state['audio_object'] 

            col1, col2 = st.columns(2)
            st.session_state["SPL_vs_time_xlim"][0] = col1.number_input('Lowest time:', value=st.session_state["SPL_vs_time_xlim"][0], step=0.1)
            st.session_state["SPL_vs_time_xlim"][1] = col2.number_input('Highest time:', value=st.session_state["SPL_vs_time_xlim"][1], step=0.1)

            col1, col2 = st.columns(2)

            col2col1, col2col2, col2col3, col2col4 = col2.columns(4) 
            submit_button = col2.form_submit_button(label='Run', use_container_width=True)

            try:
                for i in range(len(st.session_state['chosen_file_indexes'])):
                    if col1.checkbox(f"{audio_object[i].audio_name}", True):
                        st.session_state['to_plot'][i] = True
                    else:
                        st.session_state['to_plot'][i] = False

            except:
                pass

        if submit_button or False:
            SPL_vs_time(audio_object, frequency=st.session_state['frequency'], ponderation=st.session_state['ponderation'], xlim=st.session_state["SPL_vs_time_xlim"])

        #Download the image
        image_name= f"SPL_vs_time-Ponderation{st.session_state['ponderation']}-Time[{st.session_state['SPL_vs_time_xlim'][0]},{st.session_state['SPL_vs_time_xlim'][1]}].png"
        image_name = st.text_input('Image name:', value=image_name)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        save_button = st.download_button(
                    label='Download Image',
                    data=img,
                    file_name=image_name,
                    mime="image/png"
                    )

    elif options =='Reverb Time':
        st.header('Reverb Time')

        freq_resolution = st.session_state['stats']['Stats'][1]

        time_fft=st.session_state['audio_object'].time_fft
        if 'SPL_vs_time_xlim' not in st.session_state:
            st.session_state['SPL_vs_time_xlim'] = [time_fft[0], time_fft[len(time_fft)-1]]

        
        freq_fft=st.session_state['audio_object'].freq_fft
        if 'RT' not in st.session_state:
            st.session_state['RT'] = 60
        if 'drop' not in st.session_state:
            st.session_state['drop'] = 20
        if 'choice' not in st.session_state:
            st.session_state['choice'] = False
        if 'frequency' not in st.session_state:
            st.session_state['frequency'] = False


        audio_object = st.session_state['audio_object'] 
        col1,col2,col3, col4 = st.columns(4)

        st.session_state['RT'] = col1.number_input('RT to calculate:', value=st.session_state['RT'], step=5)

        st.session_state['drop'] = col2.number_input('Drop to locate (dB):', value=st.session_state['drop'])


        choices=['All', 'Choose frequency:']
        st.session_state['choice']   =  col3.selectbox('Frequency?', options=choices, index=choices.index(st.session_state['choice']))
        if st.session_state['choice'] =='All':
            st.session_state['frequency'] = False
        else:
            if st.session_state['frequency']==False: st.session_state['frequency'] = freq_fft[0]
            st.session_state['frequency']=  col4.number_input('', min_value = freq_fft[0], max_value=freq_fft[len(freq_fft)-1], value=st.session_state['frequency'], step=freq_resolution*1.0)


        reverb_time(audio_object, RT=st.session_state['RT'], drop=st.session_state['drop'], frequency=st.session_state['frequency'])





if 'stats' in st.session_state and options!='Files':
    st.table(st.session_state['stats'])

