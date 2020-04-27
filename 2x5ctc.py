import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile as wav
import ctc

nfft=4096
c0=343
fs=44100

f=np.arange(0,fs/2+fs/nfft,fs/nfft) #create frequency vector
t=np.arange(0,nfft/fs,1/fs) #Time vector for plotting filters

ld=1.2 #length of loudspeaker array
cd=1.0 #distance from listener to loudspeaker array (from centre)
r_del=0.2 #head diameter of listener's head

#shadowless, constant regularization, 
r=ctc.pathcalc_52(ld,cd,r_del) #calculate the transmission path lengths for 2*5 plant matrix
C=ctc.plant_shadowless(f,r,5,2) #generate 2*5 plant matrix from the calculated path lengths

plt.figure()
plt.plot(f,20*np.log10(abs(C[0,0,:])))
plt.title("Shadowless Head Modelled HRTF C11 and C21")
plt.plot(f,20*np.log10(abs(C[0,1,:])))
plt.ylabel("Magnitude [dB]")
plt.xscale('log')

c=ctc.multiIRFFT(C,nfft)

plt.figure()
plt.plot(t,c[0,0,:])
plt.title("Shadowless Head Modelled HRIR c11 -- c15")
plt.plot(t,c[0,1,:])
plt.plot(t,c[0,2,:])
plt.plot(t,c[0,3,:])
plt.plot(t,c[0,4,:])
plt.ylabel("Magnitude")
plt.xlabel('time [s]')

H=ctc.filtGen_constreg(C,0.0001) #filter generation from pseudo inverse+regularization (beta=0.001)

H=H*np.exp(-1j*2*np.pi*f*0.5) #modelling delay

plt.figure()
plt.plot(f,20*np.log10(abs(H[0,0,:])))
plt.title("Frequency Response H11 and H12")
plt.plot(f,20*np.log10(abs(H[0,1,:])))
plt.ylabel("Magnitude [dB]")
plt.xscale('log')

plt.figure()
plt.plot(f,20*np.log10(abs(H[0,0,:])),label='H11')
plt.title("Frequency Response H11 -- H51")
plt.plot(f,20*np.log10(abs(H[1,0,:])),label='H21')
plt.plot(f,20*np.log10(abs(H[2,0,:])),label='H31')
plt.plot(f,20*np.log10(abs(H[3,0,:])),label='H41')
plt.plot(f,20*np.log10(abs(H[4,0,:])),label='H51')
plt.ylabel("Magnitude [dB]")
plt.xscale('log')
plt.legend()

plt.figure()
plt.plot(f,20*np.log10(abs(H[0,1,:])),label='H12')
plt.title("Frequency Response H12 -- H52")
plt.plot(f,20*np.log10(abs(H[1,1,:])),label='H22')
plt.plot(f,20*np.log10(abs(H[2,1,:])),label='H32')
plt.plot(f,20*np.log10(abs(H[3,1,:])),label='H42')
plt.plot(f,20*np.log10(abs(H[4,1,:])),label='H52')
plt.ylabel("Magnitude [dB]")
plt.xscale('log')
plt.legend()

h=ctc.multiIRFFT(H,nfft) #IRFFT of the entire filter matrix with nfft points

plt.figure()
plt.plot(t,h[0,0,:])
plt.title("Impulse Response h11 and h12")
plt.plot(t,h[0,1,:])
plt.ylabel("Magnitude")
plt.xlabel('time [s]')

#Reproduced Pressure
D,d=ctc.targetSig(len(f),nfft)
P=ctc.repPressure(C,H,D)

plt.figure()
plt.plot(f,np.abs(P[0,0,:]))
plt.plot(f,np.abs(P[1,0,:]))
plt.ylabel("dB")
plt.xscale('log')
plt.title ("Reproduced Pressure")

#Crosstalk Matrix
Z=ctc.xtalkMat(C,H)
plt.figure()
plt.plot(f,20*np.log10(np.abs(Z[0,0,:])))
plt.plot(f,20*np.log10(np.abs(Z[0,1,:])))
plt.ylabel("dB")
plt.xscale('log')
plt.title("Xtalk Matrix")

#Crosstalk Spectrum
Epsi=np.abs(Z[0,0,:])/np.abs(Z[1,0,:])
plt.figure()
plt.plot(f,20*np.log10(Epsi))
plt.ylabel("dB")
plt.xscale('log')
plt.title("Xtalk Spectrum")

#Generate wav file
output=ctc.multiConvolve_test(h,d) #multi-convolve target signal with filters
ctc.wav_gen(output[0,:],'loudspeaker_output1.wav',fs)
ctc.wav_gen(output[1,:],'loudspeaker_output2.wav',fs)
ctc.wav_gen(output[2,:],'loudspeaker_output3.wav',fs)
ctc.wav_gen(output[3,:],'loudspeaker_output4.wav',fs)
ctc.wav_gen(output[4,:],'loudspeaker_output5.wav',fs)

ctc.wav_gen(c[0,1,:],'HRIR2.wav',fs)
