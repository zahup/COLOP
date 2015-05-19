import sys
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

%matplotlib inline

#Set computational variables

#Shape parameter for Kaiser window
beta=0

#Normalized amplitude treshold, smaller amplitudes will be skipped during computation
treshold=0.2

#Operator phase spectrum depending on polarity of the seismic data
#SEG normal(AI increase = through): phase=90; SEG reverse (AI increase = Peak): phase=-90
phase=-90

#Number of samples of the operator
num=100


#Input spectrum files
seisfile="C:\\users\\zahup\desktop\\F2_01_seismic_amplitude_spectrum.dat"
wellfile="C:\\users\\zahup\desktop\\F2_01_well_AI_amplitude_spectrum.dat"

#Ouput operator file
operatorfile="C:\\users\\zahup\desktop\\F2_01_colour_operator.dat"

#########################################################################

#Load exported spectrums from OpendTect
freqseis, ampseis=np.loadtxt(seisfile, unpack=True)
freqwell, ampwell=np.loadtxt(wellfile, unpack=True)


# dB to amplitude conversion
ampwell=np.power(10,ampwell/20)
ampseis=np.power(10,ampseis/20)

#Normalize seismic spectrum
normseis = ampseis / np.max(ampseis)

#Calculate logarithmic well spectrum
logfreq= np.log10(freqwell)
logamp=np.log10(ampwell)

#Linear regression on logarithmic well spectrum
slope, intercept, rvalue, pvalue, stderr = linregress(logfreq,logamp)
print ('Regression results:')
print ("Intercept:", intercept)
print ("Slope    :", slope)
print ("R-value  :", rvalue)

#Plot well based AI spectrum with regression line
lintrend=intercept+slope*logfreq
plt.figure(0)
plt.scatter(logfreq,logamp, label="AI impedance spectrum")
plt.xlabel("log10(frequency)")
plt.ylabel("log10(amplitude)")
plt.plot(logfreq,lintrend, label="Trend line", linewidth=3)
plt.xlim(np.min(logfreq),np.max(logfreq))
plt.legend()


#Calculate regression based trend well spectrum
WelltrendSpectrum = intercept*np.power(freqseis,slope)

#Calculate residual spectrum
treshold = 0.2
ResidualSpectrum=np.zeros(len(normseis))
for i in range(len(normseis)):
    if normseis[i]>treshold:
        ResidualSpectrum[i]= WelltrendSpectrum[i] / normseis[i]
        
#Normalize residual spectrum
ResidualSpectrum=ResidualSpectrum / np.max(ResidualSpectrum)

#Plot normalized seismic spectrum with well trend spectrum
plt.figure(1)
thold=np.ones(len(freqseis))
thold=treshold*thold
plt.plot(freqseis,normseis, label='Normalized seismic amplitude spectrum')
plt.plot(freqseis,WelltrendSpectrum, label='Regression based AI spectrum')
plt.plot(freqseis,ResidualSpectrum, label='Frequency domain raw operator')
plt.plot(freqseis,thold, label='Amplitude treshold')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Normalized Amplitude')
plt.ylim(0,1.5)
plt.xlim(0,np.max(freqseis))
plt.legend()

#Calculate dt
dt=1/(2*np.max(freqseis))

#Setup complex amplitude spectrum for ifft

phase=np.radians(phase)
cspectrum=ResidualSpectrum*(np.cos(phase)+1j*np.sin(phase))
cspectrum_neg=ResidualSpectrum*(np.cos(-1*phase)+1j*np.sin(-1*phase))
rev_cspectrum_neg=np.fliplr([cspectrum_neg])[0]
np.append(cspectrum,rev_cspectrum_neg)

#Calculate ifft and reorder arrays
t_op=np.fft.ifft(cspectrum)
start_t=(-1/2)*dt*len(cspectrum)
t_shift=np.linspace(start_t,-1*start_t,len(t_op))
t_op_shift=np.fft.ifftshift(t_op)

#Tapering
window=np.kaiser(len(t_shift),beta)
t_op_final=t_op_shift*window

start_i=(int(len(t_shift)/2))-int(num/2)
stop_i=(int(len(t_shift)/2))+int(num/2)

#Plot final time domain operator
plt.figure(2)
plt.plot(t_shift,t_op_final, label='Time domain operator')
plt.xlim(t_shift[start_i],t_shift[stop_i])
plt.ylim(-0.07,0.09)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

#Save final operator
np.savetxt(operatorfile,t_op_final[start_i:stop_i].real)

#QC operator
