'''
Audio Signal Processing Tools 1
MSc Acoustical Engineering 2019-2020
Generic Toolbox for Python Signal Processing
'''
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile as wav

#==============================================================================
#2-Channel Wav File Reader
#Convert .wav file into two arrays
#Enable Mono file to two channels / Split 2-channel .wav file
#To read mono file, use scipy.io.wavefile.read instead
#==============================================================================
def wav_read2(file_name):
    fs, data = wav.read(file_name)
    shape=data.shape
    if len(shape)==1:
        ch_l=data
        ch_r=data
    elif len(shape)==2:
        ch_l=data[:, 0]
        ch_r=data[:, 1]
    else:
        ch_l=0
        ch_r=0
    return (fs,ch_l,ch_r)
    
#==============================================================================
#Wav File Generator
#Convert any Numpy Array into Audible wav File
#Essentially corrects the data type needed for an audible wav file
#==============================================================================
def wav_gen(array,file_name,fs):
    array=np.asarray(array, dtype=np.int16)
    wav.write(file_name,fs,array)
    return 0

#==============================================================================
#HRIR Filtering
#Filter 
#Essentially corrects the data type needed for an audible wav file
#==============================================================================
def hrirfilt(data_left,data_right,hrir_left,hrir_right,hrir_gain):
    a=np.array([1.0])
    outl=signal.lfilter(hrir_left*hrir_gain,a,data_left)
    outr=signal.lfilter(hrir_right*hrir_gain,a,data_right)
    out=np.vstack((outl,outr))
    out=np.transpose(out)
    return(outl,outr,out)

#==============================================================================
#Sine Wave Generation Corresponding to Sampling Frequency
#Return x[n]=sin[2*pi*f*t] with length t*fs
#==============================================================================
def sineGen(frequency,time,fs):
    t_vec=np.arange(0,time,1/fs)
    x=np.sin(2*np.pi*frequency*t_vec)
    return(x)

#==============================================================================
#Impulse Generation
#Generate a pulse with zero paddings
#
#==============================================================================
def impulseGen(length):
    i=[1]+[0]*(length-1)
    impulse=np.array(i)
    return(impulse)

#==============================================================================
#Generate double sided response from single sided response
#So that IFFT & IRFFT can be computed
#
#==============================================================================
def doubleSide(h):
    H=[0]*(len(h)-1)*2
    H[0]=h[0]
    for i in range (len(h)):
        H[i]=h[i]
        H[-i]=np.conj(h[i])
    return(H)