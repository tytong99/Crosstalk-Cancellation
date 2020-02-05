'''
Crosstalk cancellation filter implementation with two loudspeakers, FIR
Tianyou Tong, University of Southampton
v1.000
'''
import audio
import numpy as np
import math as m
import scipy.signal as spy
from scipy.io import wavfile
import logging
import wave
'''
Logger Configuration
'''
logging.basicConfig(filename="testCTC.log",format='%(asctime)s %(message)s',
                    filemode='w')
logger=logging.getLogger

'''Universal Parameters'''
#desired filter length
L=20000
#sampling frequency
fs=44100
#acoustic speed[m/s]
c0=343
#bulk particle density[kg/m^3]
rho0=1.225
'''
Read wav files with already splitted channels
'''
fs,data=wavfile.read('')
wav_left=data[:,0]
wav_right=data[:,1]
'''
Loudspeaker Setup
'''
#distance between two loudspeakers(sources)[m]
ld=0            
#distance from listener to the centre point between two sources[m]
cd=0            
#effective distance between ear canals (head diameter)[m]
r_del=0
#distance from two sources to centre of listener[m]
l=m.sqrt(m.pow(cd,2)+m.pow(ld,2)/4)
#azimuth angle[rads]
theta=m.atan(ld/(2*cd))
#loudspeaker span
Theta=2*theta
#distance to desirable audio source[m]
l1=m.sqrt(m.pow(l,2)+m.pow(r_del,2)-r_del*l*m.sin(theta))
#distance to source of crosstalk[m]
l2=m.sqrt(m.pow(l,2)+m.pow(r_del,2)+r_del*l*m.sin(theta))            
#path length difference[m]
l_del=l2-l1     
#path length ratio
g=l1/l2         
#time delay between two ears reaching the listener for single plane wave[s]
tau=l_del/c0
'''
Acoustical Parameters in Frequency Domain
'''
#initialise frequency bins
dt=1.0
f=np.arange(20,20000,1)
#normalised angular frequency
om=2*m.pi*f
#wavenumber k

#ipsilateral spatial phasor phi
phi=m.pow(m.e,(1j*k*l1)
#normalised phasor over transmission alpha
alpha=phi/l1
#transfer matrix parameter
s=g*pow(m.e,(-1j*om*tau))
'''
Plant Matrix between Ipsilateral / Contralateral Cues
'''
#1/(1-s^2),s/(1-s^2),s/(1-s^2),1/(1-s^2)
'''
CTC Filter Matrix Generation
'''
#HLL
HLL=1/alpha*(1/(1-np.square(s)))
#HLR
HLL=1/alpha*(s/(1-np.square(s)))
#HRL
HLL=1/alpha*(s/(1-np.square(s)))
#HRR
HLL=1/alpha*(1/(1-np.square(s)))
'''
Generate FIR filter
'''
#Window function
w=spy.hanning(L)
#FLL
FLL=np.fft.ifft(HLL*w)
#FLR
FLR=np.fft.ifft(HLR*w)
#FRL
FRL=np.fft.ifft(HRL*w)
#FRR
FRR=np.fft.ifft(HRR*w)
'''
Process Data Through Filter
'''
wav_out_left=spy.convolve(wav_left,FLL,same)+spy.convolve(wav_right,FLR,same)
wav_out_right=spy.convolve(wav_left,FRL,same)+spy.convolve(wav_right,FRR,same)