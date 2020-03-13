'''
Crosstalk Cancellation Toolkit
MSc Acoustical Engineering 2019-2020
'''
import numpy as np
import math as m

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
#2x2 Crosstalk Cancellation Filter Parameter Calculator
#Provide distance between loudspeakers,
#distance from listener centre to centre point of loudspeakers
#head diameter r_del
#return l1, ipsilateral path, l2 contralateral path, g ratio lo1/l2, ITD
#==============================================================================
def pathcalc_22_parametric(ld,cd,r_del):
    c0=343
#distance from loudspeaker to listener centre
    lc=m.sqrt(m.pow(cd,2)+m.pow(ld,2)/4)
    theta=m.atan(ld/(2*cd))
#loudspeaker span
    Theta=2*theta
    l1=m.sqrt(m.pow(lc,2)+m.pow(r_del,2)-r_del*lc*m.sin(theta))
    l2=m.sqrt(m.pow(lc,2)+m.pow(r_del,2)+r_del*lc*m.sin(theta))
    l_del=l2-l1     
    g=l1/l2         
    ITD=l_del/c0
    return(lc,l1,l2,g,ITD)
#==============================================================================
#2x2 Shadowless Crosstalk Cancellation Filter Parameter Calculator
#Provide distance between loudspeakers,
#distance from listener centre to centre point of loudspeakers
#head diameter r_del
#return 2x2 matrix
#
#==============================================================================
def pathcalc_22(ld,cd,r_del):
    c0=343
#distance from loudspeaker to listener centre
    lc=m.sqrt(m.pow(cd,2)+m.pow(ld,2)/4)
    theta=m.atan(ld/(2*cd))
#loudspeaker span
    Theta=2*theta
    l1=m.sqrt(m.pow(lc,2)+m.pow(r_del,2)-r_del*lc*m.sin(theta))
    l2=m.sqrt(m.pow(lc,2)+m.pow(r_del,2)+r_del*lc*m.sin(theta))
    l_del=l2-l1     
    g=l1/l2         
    ITD=l_del/c0
    r=np.array([[l1,l2],[l2,l1]])
    return(r)

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
    c0=343
    l11=np.sqrt((ld/2-r_del/2)**2+cd**2)
    l12=np.sqrt((ld/2+r_del/2)**2+cd**2)
    l21=np.sqrt((ld/4-r_del/2)**2+cd**2)
    l22=np.sqrt((ld/4+r_del/2)**2+cd**2)
    l31=np.sqrt((r_del/2)**2+cd**2)
#   l11=l52,l12=l51,l21=l42,l22=l41
    r=np.array([[l11,l21,l31,l22,l12],[l12,l22,l31,l21,l11]])
    return (r)

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
    c0=343
    c=np.ndarray(shape=(m,l,len(f)),dtype=complex)
    for k in range (len(f)):
        for i in range(m):
            for q in range (l):
                c[i-1][q-1][k-1]=np.exp(-1j*2*np.pi*f[k-1]*r[i-1][q-1]/c0)/(4*np.pi*r[i-1][q-1])
    return(c)

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
def filtGen(c):
    m,l,lenFreq=np.shape(c)
    h=np.ndarray(shape=(l,m,lenFreq),dtype=complex)
    for k in range (lenFreq):
        h[:,:,k-1]=np.linalg.pinv(c[:,:,k-1])
    return (h)

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
def filtGen_constreg(c,beta):
    m,l,lenFreq=np.shape(c)
    h=np.ndarray(shape=(l,m,lenFreq),dtype=complex)
    I=np.eye(m)
    for k in range (lenFreq):
        h[:,:,k-1]=np.matmul(c[:,:,k-1].T.conjugate(),np.linalg.pinv(np.matmul(c[:,:,k-1],c[:,:,k-1].T.conjugate())+beta*I))
    return (h)
    
#==============================================================================
#Performance Matrix Generation
#Provide sequence of plant matrix (3D array 'C' with dimensions m,l,len(f)),
#Provide sequence of crosstalk matrix (3D array 'H' with dimensions l,m,len(f))
#return a sequence of 2x2 performance matrices (3D array 'R')
#==============================================================================
def performMat(c,h):
    m,l,lenFreq=np.shape(c)
    Z=np.ndarray(shape=(2,2,lenFreq),dtype=complex)
    for k in range(lenFreq):
        Z[:,:,k-1]=np.matmul(c[:,:,k-1],h[:,:,k-1])
    return (Z)
        
#==============================================================================
#Generate a sequence of double sided filter matrices 
#from a sequence of single sided filter matrices
#So that IFFT & IRFFT can be computed
#
#==============================================================================
def filt_doubleSide(h):
    l,m,lenFreq=np.shape(h)
    H=np.ndarray(shape=(l,m,(lenFreq-1)*2),dtype=complex)
    H[:,:,0]=h[:,:,0]
    for i in range (l):
        for j in range (m):
            for k in range (lenFreq-1):
                H[i,j,k]=h[i,j,k]
                H[i,j,-k]=np.conj(h[i,j,k])
    return(H)

#==============================================================================
#Perform IRFFT to a sequence of frequency domain filters
#==============================================================================
def multiIRFFT(Hw,nfft):
    l,m,lenFreq=np.shape(Hw)
    Ht=np.ndarray(shape=(l,m,nfft))
    for i in range(l):
        for j in range (m):
            Ht[i-1,j-1,:]=np.fft.irfft(Hw[i-1,j-1,:],nfft)
    return(Ht)

#==============================================================================
#Truncate a sequence of double sided filter matrices with length N
#into a sequence of single sided filter matrices with length (N/2+1)
#
#==============================================================================
def filt_singleSideTrunc(Ht):
    l,m,len2=np.shape(Ht)
    filter=np.ndarray(shape=(l,m,int(len2/2)+1),dtype=complex)
    for k in range (int(lenFreq/2)+1):
        filter[:,:,k-1]=Ht[:,:,k-1]
    return(H)

#==============================================================================
#CTC Multiconvolver for single listener
#convolve a read-in two-channel wav. signal with a sequence of filter matrices
#
#==============================================================================
def ctcConvolve_single(filter,d):
    l,m,lenfilt=np.shape(filter)
    lenaudio,channel=np.shape(d)
    output=np.ndarray(shape=(l,lenaudio+lenfilt-1))
    for i in range (l):
        output[i-1,:]=np.convolve(filter[i-1,0,:],d.T[0,:])+np.convolve(filter[i-1,1,:],d.T[1,:])
    return (output)
            