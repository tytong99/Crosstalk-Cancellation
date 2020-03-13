import ctc
import numpy as np
import matplotlib.pyplot as plt

fs=44100
c0=343
nfft=4096

D_loudspeaker=1.2
D_centre=1.0
D_head=0.2

f=np.arange(0,fs/2+fs/nfft,fs/nfft)

channel_1=ctc.impulseGen(1024)
channel_2=np.zeros(1024)

r=ctc.pathcalc_22(D_loudspeaker,D_centre,D_head)
c=ctc.plant_shadowless(f,r,2,2)
h=ctc.filtGen(c)

h=ctc.filtGen_constreg(c,0) #filter generation from pseudo inverse+regularization
h=h*np.exp(-1j*2*np.pi*f*0.3)
#??????
H=ctc.filt_doubleSide(h)
Ht=ctc.multiIRFFT(H,512)

R=ctc.performMat(c,h)
plt.plot(Ht[0,0,:])

r=ctc.performMat(c,h)
plt.figure()
plt.plot(f,np.abs(r[0,0,:]))
plt.plot(f,np.abs(r[0,1,:]))
plt.xscale('log')
