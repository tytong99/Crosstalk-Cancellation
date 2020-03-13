import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile as wav
import ctc
import genericsp as gsp

nfft=4096
c0=343
fs,l,r = gsp.wav_read2('violin.wav') #Mono audio file, should have fs=44100
rate1,hrirl=wav.read('L0e090a.wav') #HRIR left, should have sampling freq 44100
rate2,hrirr=wav.read('R0e090a.wav') #HRIR right, should have sampling freq 44100

outl,outr,out=gsp.hrirfilt(l,r,hrirl,hrirr,1/60000)
#convolve the mono file with HRIRs, out is the matrix ready to be converted to wav.
#outl is the left channel vector, outr is the right channel vector

f=np.arange(0,fs/2+fs/nfft,fs/nfft) #create frequency vector

ld=1.2 #length of loudspeaker array
cd=1.0 #distance from listener to loudspeaker array (from centre)
r_del=0.2 #head diameter of listener's head

#shadowless, constant regularization, 
r=ctc.pathcalc_52(ld,cd,r_del) #calculate the transmission path lengths for 2*5 plant matrix
c=ctc.plant_shadowless(f,r,5,2) #generate 2*5 plant matrix from the calculated path lengths
h=ctc.filtGen_constreg(c,0.001) #filter generation from pseudo inverse+regularization (beta=0.001)

h=h*np.exp(-1j*2*np.pi*f*0.5) #modelling delay
H=ctc.filt_doubleSide(h) #append the entire frequency response to double-sided response

Ht=ctc.multiIRFFT(H,nfft) #IRFFT of the entire filter matrix with nfft points
plt.plot(Ht[0,0,:])

output=ctc.ctcConvolve_single(Ht,out) 
#loudspeaker output matrix acquired, shape = 5*(nfft+length of audio-1)
#generate wav files
gsp.wav_gen(output[0,:],'loudspeaker_output1.wav',fs)
gsp.wav_gen(output[1,:],'loudspeaker_output2.wav',fs)
gsp.wav_gen(output[2,:],'loudspeaker_output3.wav',fs)
gsp.wav_gen(output[3,:],'loudspeaker_output4.wav',fs)
gsp.wav_gen(output[4,:],'loudspeaker_output5.wav',fs)


Z=ctc.performMat(c,h)
plt.figure()
plt.plot(f,np.abs(Z[0,0,:]))
plt.plot(f,np.abs(Z[0,1,:]))
plt.xscale('log')
