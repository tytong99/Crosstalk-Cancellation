'''
Crosstalk Cancellation Toolkit
MSc Acoustical Engineering 2019-2020
'''
import numpy as np
import math as m
import scipy.signal as signal
from scipy.io import wavfile as wav

c0=343

#==============================================================================
#Part 1: System Setup Description and Distances between Loudspeakers & Listener
#==============================================================================

#==============================================================================
#5x2 Shadowless Crosstalk Cancellation Filter Parameter Calculator
#Provide total distance between loudspeakers ld
#distance from listener centre to centre point of loudspeakers cd
#head diameter r_del
#return r 2x5 matrix
#[r11, r21, r31, r41, r51]
#[r12, r22, r32, r42, r52]
#==============================================================================
def pathcalc_52(ld,cd,r_del):
    l11=np.sqrt((ld/2-r_del/2)**2+cd**2)
    l12=np.sqrt((ld/2+r_del/2)**2+cd**2)
    l21=np.sqrt((ld/4-r_del/2)**2+cd**2)
    l22=np.sqrt((ld/4+r_del/2)**2+cd**2)
    l31=np.sqrt((r_del/2)**2+cd**2)
#   l11=l52,l12=l51,l21=l42,l22=l41
    r=np.array([[l11,l21,l31,l22,l12],[l12,l22,l31,l21,l11]])
    return (r)

#==============================================================================
#Part 2: CTC Filter Generation
#==============================================================================

#==============================================================================
#LxM Shadowless Plant Matrix Generation
#Provide frequency sequence f, path length matrix r, 
#loudspeaker number l, listening point number m,
#plant expression e^(-jwr/c0)/4pir

#for each frequency, generate a corresponding plant matrix,
#return a sequence of plant matrices (3D array 'C')
#[c11, c12,......, c1l]
#[c21, c22,......, c2l]
#[        ,......,    ]    [        ,......,    ]    [        ,......,    ]
#[cm1, cm2,......, cml]
#--------C-of-f1-------------------C-of-f2-------------------C-of-f3-------->
#                              frequency

#path length matrix r must be the same dimention with c
#==============================================================================
def plant_shadowless(f,r,l,m):
    C=np.ndarray(shape=(m,l,len(f)),dtype=complex)
    for k in range (len(f)):
        for i in range(m):
            for q in range (l):
                C[i][q][k]=np.exp(-1j*2*np.pi*f[k-1]*r[i-1][q-1]/c0)/(4*np.pi*r[i-1][q-1])
    return(C)

#==============================================================================
#Filter Matrix Generation from Moore-Penrose Pseudo Inverse of Matrix
#Provide sequence of plant matrix (3D array 'C' with dimensions m,l,len(f)),
#apply M-P pseudo inverse to the m*l matrix on each frequency bin

#return a sequence of filter matrices (3D array 'H')
#[H11, H12,......, H1m]
#[H21, H22,......, H2m]
#[        ,......,    ]    [        ,......,    ]    [        ,......,    ]
#[Hl1, Hl2,......, Hlm]
#--------H-of-f1-------------------H-of-f2-------------------H-of-f3-------->
#                              frequency

#third dimension of plant matrix should match with the frequency vector
#==============================================================================
def filtGen(C):
    m,l,length=np.shape(C)
    H=np.ndarray(shape=(l,m,length),dtype=complex)
    for k in range (length):
        H[:,:,k]=np.linalg.pinv(C[:,:,k])
    return (H)

#==============================================================================
#Filter Matrix Generation from Pseudo Inversion with Tiknohov Regularisation
#Provide sequence of plant matrix and constant regularisation factor

#return a sequence of filter matrices (3D array 'H')
#[H11, H12,......, H1m]
#[H21, H22,......, H2m]
#[        ,......,    ]    [        ,......,    ]    [        ,......,    ]
#[Hl1, Hl2,......, Hlm]
#--------H-of-f1-------------------H-of-f2-------------------H-of-f3-------->
#                              frequency

#third dimension of plant matrix should match with the frequency vector
#==============================================================================
def filtGen_constreg(C,beta):
    m,l,length=np.shape(C)
    H=np.ndarray(shape=(l,m,length),dtype=complex)
    I=np.eye(m)
#H=C^h x (C x C^h + beta x I)^-1
    for k in range (length):
        H[:,:,k]=np.matmul(C[:,:,k].T.conjugate(),np.linalg.pinv(np.matmul(C[:,:,k],C[:,:,k].T.conjugate())+beta*I))
    return (H)

#==============================================================================
#Perform IRFFT to a sequence of frequency domain filters
#==============================================================================
def multiIRFFT(H,nfft):
    l,m,length=np.shape(H)
    h=np.ndarray(shape=(l,m,nfft))
    for i in range(l):
        for j in range (m):
            h[i,j,:]=np.fft.irfft(H[i,j,:],nfft)
    return(h)

#==============================================================================
#Part 3: Testing and Evaluation
#==============================================================================

#==============================================================================
#Target Signal
#Generate a target impulse signal with a flat frequency response
#Generate both the frequency response D and the time domain signal d
#D=[]
#d is the IRFFT result of D
#==============================================================================
def targetSig(length,nfft):
    D=np.ndarray(shape=(2,1,length),dtype=complex)
    D[0,0,:]=1
    D[1,0,:]=0
    d=np.ndarray(shape=(2,1,nfft),dtype=complex)
    d[0,0,:]=np.fft.irfft(D[0,0,:])
    d[1,0,:]=np.fft.irfft(D[1,0,:])
    return (D,d)
    
#==============================================================================
#Performance Matrix Generation
#Provide sequence of plant matrix (3D array 'C' with dimensions m,l,len(f)),
#Provide sequence of crosstalk matrix (3D array 'H' with dimensions l,m,len(f))
#return a sequence of 2x2 performance matrices (3D array 'R')
#==============================================================================
def perfMatrix(C,H):
    m,l,lenFreq=np.shape(C)
    Z=np.ndarray(shape=(2,2,lenFreq),dtype=complex)
    for k in range(lenFreq):
        Z[:,:,k]=np.matmul(C[:,:,k],H[:,:,k])
    return (Z)

#==============================================================================
#Reproduced Pressure Generation
#Provide sequence of plant matrix (3D array 'C' with dimensions m,l,length),
#Provide sequence of crosstalk matrix (3D array 'H' with dimensions l,m,length)
#Provide target signal D (3D array 'D' with dimensions 2,1,length)
#return a sequence of 2x2 performance matrices (3D array 'R')
#==============================================================================
def repPressure(C,H,D):
    m,l,length=np.shape(C)
    P=np.ndarray(shape=(2,1,length),dtype=complex)
    for k in range(length):
        P[:,:,k]=np.matmul(np.matmul(C[:,:,k],H[:,:,k]),D[:,:,k])
    return (P)

#==============================================================================
#CTC Multiconvolver for single listener using wav files
#convolve a read-in two-channel wav. signal with a sequence of filter matrices
#
#==============================================================================
def multiConvolve_wav(h,d):
    l,m,lenfilt=np.shape(h)
    lenaudio,channel=np.shape(d)
    output=np.ndarray(shape=(l,lenaudio+lenfilt-1))
    for i in range (l):
        output[i,:]=np.convolve(h[i,0,:],d.T[0,:])+np.convolve(h[i,1,:],d.T[1,:])
    return (output)

#==============================================================================
#CTC Multiconvolver for single listener using wav files
#convolve a read-in two-channel wav. signal with a sequence of filter matrices
#
#==============================================================================
def multiConvolve_test(h,d):
    l,m,lenfilt=np.shape(h)
    two,one,lenaudio=np.shape(d)
    output=np.ndarray(shape=(l,lenaudio+lenfilt-1))
    for i in range (l):
        output[i,:]=np.convolve(h[i,0,:],d[0,0,:])+np.convolve(h[i,1,:],d[1,0,:])
    return (output)

#==============================================================================
#Crosstalk Matrix H^h x C^h x C x H
#==============================================================================
def xtalkMat(c,h):
    m,l,lenFreq=np.shape(c)
    Z=np.ndarray(shape=(2,2,lenFreq),dtype=complex)
    Z1=np.ndarray(shape=(2,2,lenFreq),dtype=complex)
    for k in range(lenFreq):
        Z[:,:,k]=np.matmul(c[:,:,k],h[:,:,k])
        Z1[:,:,k]=np.matmul(h[:,:,k].T.conjugate(),c[:,:,k].T.conjugate())
        Z[:,:,k]=np.matmul(Z1[:,:,k],Z[:,:,k])
    return (Z)        

#==============================================================================
#Part 4: Other Tools for manipulating filters
#==============================================================================

#==============================================================================
#Truncate a sequence of double sided filter matrices with length N
#into a sequence of single sided filter matrices with length (N/2+1)
#For obtaining frequency response from HRIR FFTs
#==============================================================================
def filt_singleSideTrunc(H_doubleside):
    l,m,length_doubleside=np.shape(H_doubleside)
    H=np.ndarray(shape=(l,m,int(length_doubleside/2)+1),dtype=complex)
    for k in range (int(lenFreq/2)+1):
        H[:,:,k]=H_doubleside[:,:,k]
    return(H)

#==============================================================================
#Generate a sequence of double sided filter matrices 
#from a sequence of single sided filter matrices
#So that IFFT & IRFFT can be computed
#
#==============================================================================
def filt_doubleSide(H):
    l,m,length=np.shape(H)
    H_doubleside=np.ndarray(shape=(l,m,(length-1)*2),dtype=complex)
    H_doubleside[:,:,0]=H[:,:,0]
    for i in range (l):
        for j in range (m):
            for k in range (length-1):
                H_doubleside[i,j,k+1]=H[i,j,k+1]
                H[i,j,-k-1]=np.conj(H[i,j,k+1])
    return(H_doubleside)

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
#Part 5: Generic Audio Signal Processing I/O toolkit
#==============================================================================

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
#wav file HRIR reader
#==============================================================================
def hrirRead(hrirA,hrirB):
    fs_A,A=wav.read(hrirA)
    fs_B,B=wav.read(hrirB)
    if fs_A!=fs_B:
        return("Unequal Sampling Frequency between HRIRs",0,0)
    elif len(A)!= len(B):
        return(0,"Filter Lengths do not Match","Filter Lengths do not Match")
    else:
        if np.shape(np.shape(A))[0]!=1:
            return(fs_A,"Wrong Audio File Dimensions, hrirA",B)
        elif np.shape(np.shape(B))[0]!=1:
            return(fs_A,A,"Wrong Audio File Dimensions, hrirA")
        else:
            return(fs_A,A,B)
        
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

