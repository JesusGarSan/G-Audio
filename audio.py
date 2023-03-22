# Importación de librerías
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from   scipy.io import wavfile
import pandas as pd


# Clase de Audio
class audio:
    def __init__(self, audio_name ,path, calibration_parameters):
        self.audio_name=audio_name
        self.path=path

        #Method not suported in old python 3.7
        self.sample_rate, self.signal = wavfile.read(path)
        if(np.size(self.signal) > len(self.signal)): self.signal= self.signal[:, 0] #convierte canal doble en mono
        self.signal=self.re_normalize_bits(self.signal)
        self.signal=self.data_to_uPa(calibration_parameters)

        self.duration=len(self.signal)/self.sample_rate
        self.time=np.linspace(0, len(self.signal)/(self.sample_rate) , len(self.signal))
        self.espectogram_data=None

    # Recorta el audio al intervalo indicado en segundos
    def shorten_audio(self, duration=[0,3]):
        signal_length=[int(duration[0]*self.sample_rate), int(duration[1]*self.sample_rate)]
        self.signal=self.signal[signal_length[0]:signal_length[1]]
        self.duration=duration[1]-duration[0]
        self.time=np.linspace(0, len(self.signal)/(self.sample_rate) , len(self.signal))

    #Convertimos la señal de (32) bits a (24) bits
    def re_normalize_bits(self, signal, initial_bits=32, bits=24):
        signal=signal/2**(initial_bits-1)
        signal=signal*2**(bits-1)
        return signal
    
    #Conversión de la señal digital a micro Pascales
    def data_to_uPa(self, calibration_parameters):
        NBBITS= calibration_parameters['BITS']
        GAIN= calibration_parameters['GAIN'] #dB
        SH= calibration_parameters['SH'] #dB re 1V/uPa
        uPa=self.signal/(2**(NBBITS-1) * 10**(GAIN/20) * 10**(SH/20) ) #PUEDE QUE EL SIGNO DE SH ESTÉ DUPLICADO Y SEA "-" SÓLO UNA VEZ
        return uPa

    #Realiza una Transformada de Fourier de tiempo corto a la señal del audio
    def audio_stft(self, window=[None], nperseg=None, noverlap=None, nfft=None, scaling='spectrum'):
        if(nperseg==None): nperseg=256
        if(window[0]==None): window = scipy.signal.windows.blackman(nperseg,True)
        if(noverlap==None): noverlap = nperseg//2
        if(nfft==None): nfft = nperseg


        f, t, Zxx= scipy.signal.stft(self.signal, fs=self.sample_rate,
                                    window=window, nperseg=nperseg, noverlap=noverlap,
                                    nfft=nfft, detrend=False, return_onesided=True, boundary=None, padded=False,
                                    scaling=scaling)
         
        Zxx_abs=np.abs(Zxx)

        self.freq_fft, self.time_fft, self.signal_fft = f,t, Zxx_abs
        self.RT=np.array([[0]*len(self.freq_fft), [0.0]*len(self.freq_fft)])
        return f,t,Zxx_abs

    #Filtra las frecuencias de la señal. Se queda con el rango de frecuencias especificado
    def filter_frequencies(self, frequency_range=[20, 20000]):

        f_min=frequency_range[0]
        f_max=frequency_range[1]

        idx_min = (np.abs(self.freq_fft - f_min)).argmin()
        idx_max = (np.abs(self.freq_fft - f_max)).argmin() + 1

        self.freq_fft=self.freq_fft[idx_min:idx_max]
        self.signal_fft=self.signal_fft[idx_min:idx_max, :]
        self.RT=np.array(self.RT[:, idx_min:idx_max])
        return

    # Conversión directa de Presión a dB
    def uPa_to_dB(self, signal, reference=1):
        dB=10*np.log10(signal/reference) #Intensity in dB. P_ref= 1 dB SPL re 1μ Pa
        return dB 
    
    #Representa la intensidad de la señal en dB frente al tiempo
    def dB_vs_time(self, frequency=False):
        #if frequency=False it plots for every frequency
        if(frequency==False):
            time=self.time
            dB = self.uPa_to_dB(self.signal)
            
        else:
            index=np.where(self.freq_fft==frequency)[0]
            dB = self.uPa_to_dB(self.signal_fft[index][0])
            time=self.time_fft
            plt.title(f"f= {frequency} Hz")

        plt.plot(time, dB)
        plt.ylabel('Intensity (dB)')
        plt.xlabel('Time (s)')
        #plt.show()
        return dB

    # Conversión de un conjunto de valores a su rms
    def signal_to_rms(self, signal):
        signal=np.array(signal)
        rms= np.sqrt( np.mean( signal**2 ) )
        return rms #escalar

    # Cálculo de SPL a partir de rms (o uPa cualquiera)
    def rms_to_SPL(self, signal, reference=1):
        SPL = 20*np.log10(signal/reference)
        return SPL

    #Representa SPL en dB frente al tiempo
    def SPL_vs_time(self, frequency=False, ponderation='fast'):
        if(ponderation=='slow'): ponderation=1  #s
        if(ponderation=='fast'): ponderation=.125 #s
        signal=self.signal_fft

        #Get the set of points for the rms
        avg_points = int(ponderation * len(signal[0])/self.duration)
        if (avg_points==0):
            print('Choose a higher ponderation. The current value is too low')
            return np.array([0])
        total_points = int( len(signal[0])/avg_points )
        if(total_points==0):
            print('Choose a lower ponderation. The current value is too high')
            return np.array([1])

        #print(f"signal.shape: {signal.shape}")
        #print(f"Calculating rms over {avg_points} values")
        #print(f"Total # of graphed points: {total_points}")

        #lost_points=len(signal[0])-total_points*avg_points
        #if(lost_points!=0): print(f"The last {lost_points} points were lost")

        Prms=[]
        if(frequency==False):
            for i in range(total_points):
                Prms.append(self.signal_to_rms(signal[:,i*avg_points:(i+1)*avg_points]))

        else:
            index= (np.abs( self.freq_fft - frequency)).argmin()
            plt.title(f"f= {self.freq_fft[index]} Hz")
            for i in range(total_points):
                Prms.append(self.signal_to_rms(signal[index,i*avg_points:(i+1)*avg_points]))

        Prms=np.array(Prms)
        SPL=self.rms_to_SPL(Prms)

        plt.plot(np.linspace(0, self.duration, len(SPL)), SPL, label=self.audio_name)
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("SPL (dB re 1 uPa)")
        #plt.show()
        return SPL
    
    def SPL_vs_freq(self, source=False, method='log', legend=False, Pts='all', time_avg=1):
        if(Pts=='all'): Pts=len(self.signal_fft)
        step=int(len(self.signal_fft)/Pts)
        if(legend==False): legend=self.audio_name
        plt.ylabel('SPL (dB re 1μPa)')
        plt.xlabel('Frequency (Hz)')
        
        SPLs=[]
        time_step = np.abs(self.time_fft-time_avg).argmin()
        if time_step > len(self.time_fft)/2: time_step = len(self.time_fft) -1
        j=0

        while (j+1)*time_step<len(self.time_fft):

            #time_step = 5
            freq=[]
            SPL=[]
            for i in range(Pts):
                if (source==False):
                    SPL.append(self.rms_to_SPL(self.signal_to_rms(self.signal_fft[i*step:(i+1)*step , j*time_step:(j+1)*time_step])))
                    freq.append(np.mean(self.freq_fft[i*step:(i+1)*step]))
                
                #elif(method=='log'):
                #    SPL.append(self.rms_to_SPL(self.signal_to_rms(source.signal_fft[i*step:(i+1)*step][:] - self.signal_fft[i*step:(i+1)*step][:])))
                #    freq.append(np.mean(self.freq_fft[i*step:(i+1)*step]))
                #if(legend==False): legend+=' TL'
                #elif(method=='lineal'):
                #    SPL_source = self.rms_to_SPL(self.signal_to_rms(source.signal_fft[i*step:(i+1)*step][:]))
                #    SPL_TL = self.rms_to_SPL(self.signal_to_rms(source.signal_fft[i*step:(i+1)*step][:]
                #                                                - self.signal_fft[i*step:(i+1)*step][:]))
                #    freq.append(np.mean(self.freq_fft[i*step:(i+1)*step]))
                #    if(legend==False): legend+=' (Source - Measure)'
                #    SPL.append(SPL_source-SPL_TL)
                #    plt.ylabel('ΔSPL (dB re 1μPa)')

            SPLs.append(SPL)
            j+=1


        SPLs=np.array(SPLs)
        print(SPLs.shape)
        print(len(freq))
        mean=[]
        std=[]

        for f in range(SPLs.shape[1]):
            mean.append(np.mean(SPLs[:,f]))
            std.append(np.std(SPLs[:,f]))

        error=std/np.sqrt(len(std))
        print(std)

        plt.plot(freq, mean, label=legend)
        plt.fill_between(freq, mean-error, mean+error, alpha=0.5)
        plt.grid()

        return SPL
    


    def shroeder_integration(self, frequency=False, start_time=0, end_time=False, plot=True):
        start= (np.abs( self.time_fft - start_time)).argmin()
        if(end_time==False): end_time=self.duration
        end= (np.abs( self.time_fft - end_time)).argmin()

        if(frequency==False):
            signal=self.signal[start:end]

        else:
            index= (np.abs( self.freq_fft - frequency)).argmin()
            if(plot==True): plt.title(f"f= {self.freq_fft[index]} Hz")
            signal=self.signal_fft[index,start:end]
            
        abs_signal = np.abs(signal) / np.max(np.abs(signal))
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))

        time=np.linspace(start_time, end_time, len(signal))

        if(plot==True): plt.plot(time,sch_db)

        return sch_db

    #Calcula el tiempo de reverberación se la señal completa o una de sus frecuencias (requiere calcular stft antes)
    def reverb_time_raw(self, peak_time, scale='SPL', ponderation='fast', drop=30, scout_range=50, sensibility=10, frequency=False, attempts=999):
        fig, ax = plt.subplots()
        if(scale=='SPL' and ponderation=='slow'): ponderation=1  #s
        if(scale=='SPL' and ponderation=='fast'): ponderation=.125 #s
        #We determine if the calcultion has to be done for 1 frequency or the entire signal
        if(frequency==False):
            if(scale=='dB'):
                dB = self.dB_vs_time(frequency=frequency)
                time=self.time
            if(scale=='SPL'):
                dB = self.SPL_vs_time(ponderation=ponderation)
                time = np.linspace(0, self.duration, len(dB))
            #dB=self.signal
        else:
            index= (np.abs( self.freq_fft - frequency)).argmin()
            frequency=self.freq_fft[index]
            if(scale=='dB'): 
                dB = self.dB_vs_time(frequency=frequency)
                time=self.time_fft
            if(scale=='SPL'):
                dB = self.SPL_vs_time(frequency=frequency , ponderation=ponderation)
                time = np.linspace(0, self.duration, len(dB))
            plt.title(f"f= {frequency} Hz")

        #Get the nearest time value to the peak time given
        idx = (np.abs(time - peak_time)).argmin()
        
        #Autoselect the nearest time with the highest intensity dB
        while(any(dB[idx-scout_range:idx] > dB[idx]) ): idx-=1 #Select previous time values if the intensity is higher there
        while(any(dB[idx:idx+scout_range] > dB[idx]) ): idx+=1 #Select next time values if the intensity is higher there

        peak_time=time[idx]

        peak_dB=dB[np.where(time==peak_time)][0]
        print(f"Peak: {peak_dB} dB at {peak_time}s")
        ax.axvline(x=time[idx], c='r', label=f"{round(time[idx],3)}s, {round(dB[idx],2)}dB")
        ax.legend()

        try:
            for i in range (attempts):
                #We get the first instance where the dB drop the desired amount
                indexes=np.where(dB<=peak_dB - drop)
                a=np.where((idx<indexes) == True)[1][0+i]
                idx_drop=indexes[0][a]

                #We check if that drop is stable. If not, we check for future times
                if(np.mean(dB[idx_drop:idx_drop+sensibility]) <= peak_dB - drop or (sensibility==0 and dB[idx_drop] <= peak_dB - drop) ): 

                    reverb_time=time[idx_drop] - time[idx]
                    print(f"Reverberation time T{drop}: {reverb_time}s")

                    ax.axvline(x=time[idx_drop], c='g', label=f"{round(time[idx_drop],3)}s, {round(dB[idx_drop],2)}dB")
                    #ax.axvline(x=time[idx], c='r', label=f"{round(time[idx],3)}s, {round(dB[idx],2)}dB")
                    ax.axhline(y=peak_dB, c='black', label=None)
                    ax.axhline(y=peak_dB-drop, c='black', label=f"{round(dB[idx],2)}dB, {round(dB[idx],2)-drop}dB")
                    #ax.fill_between(time, dB[idx_drop], peak_dB, where=(y<peak_dB) & (dB>dB[idx_drop]), alpha=0.5)
                    plt.legend()
                    plt.show()

                    return reverb_time
            
            raise ValueError()
        except:
            print(f"The given signal does not drop {drop}dB consistently.")
            return None

    #Calcula el tiempo de reverberación interpolando
    def reverb_time_interpolated(self, peak_time, RT=60, drop=20, ponderation='fast', frequency=False, sensibility=10, scout_range=50, margin=5 ,attempts=999):
        fig, ax = plt.subplots()
        if(ponderation=='slow'): ponderation=1  #s
        if(ponderation=='fast'): ponderation=.125 #s
        if(frequency==False):
            dB = self.SPL_vs_time(ponderation=ponderation)
            time = np.linspace(0, self.duration, len(dB))
        else:
            index= (np.abs( self.freq_fft - frequency)).argmin()
            frequency=self.freq_fft[index]
            self.RT[0,index]=RT
            dB = self.SPL_vs_time(frequency=frequency , ponderation=ponderation)
            time = np.linspace(0, self.duration, len(dB))
            plt.title(f"f= {frequency} Hz")

        #Get the nearest time value to the peak time given
        idx = (np.abs(time - peak_time)).argmin()
        
        #Autoselect the nearest time with the highest intensity dB
        while(any(dB[idx-scout_range:idx] > dB[idx]) ): idx-=1 #Select previous time values if the intensity is higher there
        while(any(dB[idx:idx+scout_range] > dB[idx]) ): idx+=1 #Select next time values if the intensity is higher there

        peak_time=time[idx]

        peak_dB=dB[np.where(time==peak_time)][0]
        #print(f"Peak: {peak_dB} dB at {peak_time}s")
        ax.axvline(x=time[idx], c='r', label=f"{round(time[idx],3)}s, {round(dB[idx],2)}dB")
        ax.legend()

        try:
            for i in range (attempts):
                #We get the values -5dB and -25dB of peak
                idx_drop = [None, None]
                #We use clip to make sure the margin is AT LEAST the one desired
                idx_drop[0] = (np.abs( (dB[idx:] - (peak_dB- margin)).clip(min=0) )).argmin() +idx
                #We search for the point like this lo let the sensibility parameter do it's thing
                indexes=np.where(dB<=peak_dB - margin - drop)
                found=np.where((idx<indexes) == True)[1][0+i]
                idx_drop[1]=indexes[0][found]

                if(np.mean(dB[idx_drop[1]:idx_drop[1]+sensibility]) <= peak_dB - margin - drop or (sensibility==0 and dB[idx_drop[1]] <= peak_dB - margin - drop) ):
                    
                    #We are going to fit a line between the margin point and the accepted drop
                    a,b = np.polyfit(time[idx_drop[0]:idx_drop[1]] , dB[idx_drop[0]:idx_drop[1]],1)
                    plt.plot(np.linspace(time[idx_drop[0]], time[idx_drop[1]],(idx_drop[1]-idx_drop[0])),
                            a*np.linspace(time[idx_drop[0]], time[idx_drop[1]],(idx_drop[1]-idx_drop[0]))+b,
                            linestyle='--', c='grey')
                    #print(f"fit: dB = {a}(dB/time)*time + {b}(dB)")
                    
                    reverb_time = (dB[idx_drop[0]]-RT-b)/a - time[idx_drop[0]]
                    #print(f"RT{RT}: {reverb_time}s")
                    
                    # Simple multiplication
                    #RT_drop = time[idx_drop[1]] - time[idx_drop[0]]
                    #real_drop=dB[idx_drop[0]] - dB[idx_drop[1]]
                    #reverb_time=RT_drop*(60/real_drop)
                    #print(f"RT{round(real_drop, 2)}: {RT_drop} s")
                    #print(f"RT{RT}: {reverb_time} s")

                    ax.axvline(x=time[idx_drop[0]], c='orange', label=f"{round(time[idx_drop[0]],3)}s, {round(dB[idx_drop[0]],2)}dB")
                    ax.axvline(x=time[idx_drop[1]], c='g', label=f"{round(time[idx_drop[1]],3)}s, {round(dB[idx_drop[1]],2)}dB")
                                
                    plt.legend()
                    #plt.show()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                    if(frequency!=False): self.RT[1,index]=reverb_time                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

                    return reverb_time
            raise ValueError()
        except:
            print(f"The given signal does not drop {drop}dB consistently.")
            if(frequency!=False): self.RT[1,index]=-1
            return None

    #Calcula el tiempo de reverberación interpolando schroeder
    def reverb_time_schroeder(self, peak_time=0, RT=60, drop=20, frequency=False, sensibility=10, margin=5 ,attempts=999, plot=True):
        if(plot==True): fig, ax = plt.subplots()
        if(frequency==False):
            dB = self.shroeder_integration(plot=plot)
            time = np.linspace(0, self.duration, len(dB))
        else:
            index= (np.abs( self.freq_fft - frequency)).argmin()
            frequency=self.freq_fft[index]
            self.RT[0,index]=RT
            dB = self.shroeder_integration(frequency=frequency, plot=plot)
            time = np.linspace(0, self.duration, len(dB))
            if(plot==True): plt.title(f"f= {frequency} Hz")

        #Get the nearest time value to the peak time given
        idx = (np.abs(time - peak_time)).argmin()
        peak_time=time[idx]

        peak_dB=dB[np.where(time==peak_time)][0]

        try:
            for i in range (attempts):
                #We get the values -5dB and -25dB of peak
                idx_drop = [None, None]
                #We use clip to make sure the margin is AT LEAST the one desired
                idx_drop[0] = (np.abs( (dB[idx:] - (peak_dB- margin)).clip(min=0) )).argmin() +idx
                #We search for the point like this lo let the sensibility parameter do it's thing
                indexes=np.where(dB<=dB[idx_drop[0]]  - drop)
                found=np.where((idx<indexes) == True)[1][0+i]
                idx_drop[1]=indexes[0][found]

                if(np.mean(dB[idx_drop[1]:idx_drop[1]+sensibility]) <= peak_dB - margin - drop or (sensibility==0 and dB[idx_drop[1]] <= peak_dB - margin - drop) ):
                    
                    #We are going to fit a line between the margin point and the accepted drop
                    a,b = np.polyfit(time[idx_drop[0]:idx_drop[1]] , dB[idx_drop[0]:idx_drop[1]],1)
                    if(plot==True):
                        plt.plot(np.linspace(time[idx_drop[0]], time[idx_drop[1]],(idx_drop[1]-idx_drop[0])),
                                a*np.linspace(time[idx_drop[0]], time[idx_drop[1]],(idx_drop[1]-idx_drop[0]))+b,
                                linestyle='--', c='grey')
                    
                    reverb_time = (dB[idx_drop[0]]-RT-b)/a - time[idx_drop[0]]

                    if(plot==True):
                        ax.axvline(x=time[idx_drop[0]], c='orange', label=f"{round(time[idx_drop[0]],3)}s, {round(dB[idx_drop[0]],2)}dB")
                        ax.axvline(x=time[idx_drop[1]], c='g', label=f"{round(time[idx_drop[1]],3)}s, {round(dB[idx_drop[1]],2)}dB")
                        plt.legend()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                        
                    if(frequency!=False): self.RT[1,index]=reverb_time                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

                    return fig, reverb_time
            raise ValueError()
        except:
            print(f"The given signal does not drop {drop}dB consistently.")
            if(frequency!=False): self.RT[1,index]=-1
            return None, None

    #Calcula el tiempo de reverberación interpolando schroeder via xiang
    def reverb_time_xiang(self,p0=[1,1,1] ,peak_time=0, RT=60, drop=20, frequency=False, start_time=0, end_time=False, sensibility=10, margin=5 ,attempts=999, plot=True):
        if(plot==True): fig, ax = plt.subplots()
        if(frequency!=False):
            index= (np.abs( self.freq_fft - frequency)).argmin()
            frequency=self.freq_fft[index]
            self.RT[0,index]=RT
            if(plot==True): plt.title(f"f= {frequency} Hz")

        if(end_time==False): end_time=self.duration
        dB = self.shroeder_integration(frequency=frequency, plot=plot, start_time=start_time, end_time=end_time)
        time = np.linspace(start_time, end_time, len(dB))

        #Get the nearest time value to the peak time given
        idx = (np.abs(time - peak_time)).argmin()
        peak_time=time[idx]

        peak_dB=dB[np.where(time==peak_time)][0]


        for i in range (attempts):
            #We get the values -5dB and -25dB of peak
            idx_drop = [None, None]
            #We use clip to make sure the margin is AT LEAST the one desired
            idx_drop[0] = (np.abs( (dB[idx:] - (peak_dB- margin)).clip(min=0) )).argmin() +idx
            #We search for the point like this lo let the sensibility parameter do it's thing
            indexes=np.where(dB<=dB[idx_drop[0]]  - drop)
            found=np.where((idx<indexes) == True)[1][0+i]
            idx_drop[1]=indexes[0][found]

            if(np.mean(dB[idx_drop[1]:idx_drop[1]+sensibility]) <= peak_dB - margin - drop or (sensibility==0 and dB[idx_drop[1]] <= peak_dB - margin - drop) ):
                
                #Linear fit to get the guess values
                a,b = np.polyfit(time[idx_drop[0]:idx_drop[1]] , dB[idx_drop[0]:idx_drop[1]],1)
                reverb_time = (dB[idx_drop[0]]-RT-b)/a - time[idx_drop[0]]

                x1_guess, x2_guess, x0_guess = p0

                print(a,b)


                # Fit Xiang
                try:
                    popt, pcov = scipy.optimize.curve_fit(lambda t, x0, x1, x2: x1*np.exp(-x2*t) + x0*(len(dB)-t),
                                                        time[idx_drop[0]:], dB[idx_drop[0]:],
                                                        p0=(x0_guess, x1_guess, x2_guess))
                except:
                    print("No se han encontrado valores optimos para el ajuste")
                    return -1
                

                x0=popt[0]
                x1=popt[1]
                x2=popt[2]
                if(plot==True):
                #    plt.plot(np.linspace(time[idx_drop[0]], time[idx_drop[1]],(idx_drop[1]-idx_drop[0])),
                #            a*np.linspace(time[idx_drop[0]], time[idx_drop[1]],(idx_drop[1]-idx_drop[0]))+b,
                #            linestyle='--', c='grey')
                
                    plt.plot(time[idx_drop[0]:],
                            x1 * np.exp(-x2 * time[idx_drop[0]:]) + x0*(len(time[idx_drop[0]:])-time[idx_drop[0]:]),
                            linestyle='--', c='grey')
                
                print(x0,x1,x2)
                error=np.sqrt(np.diag(pcov))
                print(error)
                reverb_time = 13.8/b

                if(plot==True):
                    ax.axvline(x=time[idx_drop[0]], c='orange', label=f"{round(time[idx_drop[0]],3)}s, {round(dB[idx_drop[0]],2)}dB")
                    ax.axvline(x=time[idx_drop[1]], c='g', label=f"{round(time[idx_drop[1]],3)}s, {round(dB[idx_drop[1]],2)}dB")
                    plt.legend()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                    
                if(frequency!=False): self.RT[1,index]=reverb_time                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

                return reverb_time

    #Calcula y representa el espectograma de la señal. Si se especifica un audio fuente, representa el espectrograma de TL
    def spectogram(self, source=None):

        plot_name=self.audio_name
        t=np.linspace(0,self.duration, num=len(self.signal))
        fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [1, 2.5]}, sharex=True, figsize=(14,6))
        axs[0].set(ylabel='Amplitude')
        #axs[0].set(ylabel='SPL (dB re 1uPa)')
        cmap_custom = matplotlib.cm.get_cmap('hot_r')

        if(source!=None):
            axs[0].plot(t, (source.signal-self.signal)/np.max(source.signal-self.signal)) #plot uPa
            #axs[0].plot(self.time_fft,source.SPL_vs_time() - self.SPL_vs_time()) #plot SPL
            plot_name+=f"source: {source.audio_name}"
        else:
            axs[0].plot(t,self.signal /np.max(self.signal)) # plot uPa
            #axs[0].plot(self.time_fft,self.SPL_vs_time()) # plot SPL
        fig.suptitle(plot_name)

        f,t,Zxx=self.freq_fft, self.time_fft, self.signal_fft
        

        if (source!=None):
            """ 
            #log subtraction
            Zxx_abs = 10*np.log10(10**(source.spectogram_data[2]/10) - 10**(self.spectogram_data[2]/10)) """
            #subtraction uPa
            Zxx = source.signal_fft - self.signal_fft

        Zxx_dB=self.rms_to_SPL(Zxx)

        Zxx_dB[np.isnan(Zxx_dB)] = 0.0
        Zxx_dB[np.isinf(Zxx_dB)] = 0.0
        min_amp=np.min(Zxx_dB)
        max_amp=np.max(Zxx_dB)
        #min_amp=np.min(Zxx_abs[~np.isnan(Zxx_abs)])
        #max_amp=np.max(Zxx_abs[~np.isnan(Zxx_abs)])
        min_amp=0
        max_amp=120

        colormap=axs[1].pcolormesh(t, f, Zxx_dB, vmin=min_amp, vmax=max_amp, shading='gouraud', cmap=cmap_custom)
        cbar=fig.colorbar(colormap, orientation='horizontal', shrink=0.6, pad=0.2)
        cbar.set_label('SPL (dB re 1uPa)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        #plt.ylim(20, 20e3)
        #plt.yscale('log')

        #plt.savefig(f"plots/{plot_name}.png")
        plt.show()

        return fig

    #Calcula PSD mediante uno de 2 métodos    
    def PSD(self, method='Welch', nperseg=None, noverlap=None):
        if(nperseg==None): nperseg=256
        if(noverlap==None): noverlap = nperseg//2
        fig, ax = plt.subplots(figsize=(10,6))

        if(method=='stft'):
            plt.title('stft')
            Sxx=np.array([0]*len(self.signal_fft))

            for i in range(len(Sxx)):
                Sxx[i] = self.signal_to_rms(self.signal_fft[i,:])

            freq=self.freq_fft
            PSD=self.rms_to_SPL(Sxx)

        if(method=='Welch'):

            PSD=scipy.signal.welch(self.signal, self.sample_rate, nperseg=nperseg, noverlap=noverlap, window=scipy.signal.windows.blackman(nperseg,True),
                                        nfft=nperseg, scaling='density')
            PSD=np.array(PSD)
            
            print(f"frequency resolution:{PSD[0,1]-PSD[0,0]}Hz")
            PSD[1] = self.uPa_to_dB(PSD[1], reference=1)
            freq=PSD[0]
            PSD=PSD[1]
            plt.title('Welch')


        ax.plot(freq, PSD)
        ax.grid()
        plt.xlim(0,25000)
        plt.ylim(40,120)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (1dB re 1 uPa^2/Hz)')
        plt.show()

