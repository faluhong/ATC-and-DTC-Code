import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
pi=np.pi

# This is the example of GEM-type model (including GEM-eta and GEM-sigma models) writing with Python 3.7
# Both models have four parameters

def GEM_eta(time,*param):
    P, T_Ave, eta0, eta1 = param[0], param[1], param[2], param[3]
    sum = 0
    h1 = eta0 + eta1 * time_DTC
    for i in range(1, FourierNumber + 1):
        M = 1.0 / np.sqrt(i * OmegaCal * P * P + np.sqrt(2 * i * OmegaCal) * P * h1 + h1 * h1)
        FourierA = 1.0 / pi * quad(FourierKernalA, X1, X2, args=(i))[0]
        FourierB = 1.0 / pi * quad(FourierKernalB, X1, X2, args=(i))[0]
        Fi = np.arctan(P * np.sqrt(i * OmegaCal) / (np.sqrt(2.0) * h1 + P * np.sqrt(i * OmegaCal)))
        Gt = FourierA * np.cos(i * OmegaCal * time - Fi) + FourierB * np.sin(i * OmegaCal * time - Fi)

        sum = sum + Gt * M

    LSTReturn = T_Ave +  sum
    return LSTReturn

def GEM_sigma(time,*param):

    P, T_Ave, h1, sigma =param[0], param[1], param[2], param[3]
    sum=0
    for i in range(1,FourierNumber+1):

        M=1.0/np.sqrt(i*OmegaCal*P*P+np.sqrt(2*i*OmegaCal)*P*h1+h1*h1)
        FourierA = 1.0 / pi * quad(FourierKernalA, X1, X2, args=(i))[0]
        FourierB = 1.0 / pi * quad(FourierKernalB, X1, X2, args=(i))[0]
        Fi=np.arctan(P*np.sqrt(i*OmegaCal)/(np.sqrt(2.0)*h1+P*np.sqrt(i*OmegaCal)))
        Gt=FourierA*np.cos(i*OmegaCal*time-Fi)+FourierB*np.sin(i*OmegaCal*time-Fi)

        sum=sum+Gt*M

    LSTReturn=T_Ave+sigma*(time-43200)+sum
    return LSTReturn

def FourierKernalA(x,n):
    CosSZA=np.cos(delta/180*pi)*np.cos(Latitude/180*pi)*np.cos(x-pi)+np.sin(delta/180*pi)*np.sin(Latitude/180*pi)
    AtmosphereTransmittance=1-0.2*np.sqrt(1.0/CosSZA)
    return (1-albedo)*SolarConstant*CosSZA*AtmosphereTransmittance*np.cos(n*x)

def FourierKernalB(x,n):
    CosSZA=np.cos(delta/180*pi)*np.cos(Latitude/180*pi)*np.cos(x-pi)+np.sin(delta/180*pi)*np.sin(Latitude/180*pi)
    AtmosphereTransmittance=1-0.2*np.sqrt(1.0/CosSZA)
    return (1-albedo)*SolarConstant*CosSZA*AtmosphereTransmittance*np.sin(n*x)




# Example data
Latitude = 40.3581
Longitude = -3.9803
DOY = 139

# DTC model starts from the sunrise to the next sunrise, the values greater than 24 refer to the time in the next day
time_DTC=np.arange(4.75,28.5,1)
temperature_DTC=np.array([283,286,290,295,301,306,309,313,314,314,312,309,305,301,297,294,293,291,290,288,287,286,285,284],dtype=np.float)


# Basic settings of GEM-type model
FourierNumber=10
SolarConstant=1367
albedo=0.15

# OmegaEarth represents the angular velocity of the earth
OmegaEarth=2*pi/86400
DayNumber=1
OmegaCal=OmegaEarth/DayNumber


# delta is the solar declination
delta = 23.45 * np.sin(2 * pi / 366.0 * (284 + DOY))
# omega is the duration of daytime
omega = 2.0 / 15 * math.acos(-math.tan(Latitude / 180.0 * pi) * math.tan(delta / 180.0 * pi)) * 180.0 / pi
sunrisetime = 12 - omega / 2
sunsettime=12+omega/2

# Calculating the integrating range
# TimeChangeOne and TimeChangTwo cloud be regarded as the time range of daytime (represented by seconds)
TimeChangeOne= 43200 - np.arccos((0.04 - np.sin(delta / 180 * pi) * np.sin(Latitude / 180 * pi)) / (np.cos(delta / 180 * pi) * np.cos(Latitude / 180 * pi))) / OmegaEarth
TimeChangTwo= 86400 - TimeChangeOne
#X1 and X2 represent the intergrating range of FourierKernalA and FourierKernalA
X1 = TimeChangeOne * OmegaEarth
X2 = TimeChangTwo * OmegaEarth


# Setting the initial value
p0 = [1000 ,np.average(temperature_DTC) ,10, 0]

# Solving the free parameters of GEM-eta model. Bounds and max_nfev are optional
popt, pcov = curve_fit(GEM_eta, time_DTC*3600, temperature_DTC, p0, bounds=([100, 200, -100, -100], [5000, 350, 100, 100]),max_nfev=10000)
temperature_modelling = GEM_eta(time_DTC*3600, popt[0], popt[1], popt[2], popt[3])
print (popt)

# Solving the free parameters of GEM-sigma model. Bounds and max_nfev are optional
# popt, pcov = curve_fit(GEM_sigma, time_DTC*3600, temperature_DTC, p0, bounds = ([100, 200, -100, -100], [5000, 350, 100, 100]), max_nfev = 10000)
# temperature_modelling = GEM_sigma(time_DTC*3600, popt[0], popt[1], popt[2], popt[3])
# print (popt)

print('RMSE of GEM model:',np.sqrt(np.mean(np.square(temperature_DTC-temperature_modelling))))

plt.title('Example of GEM-type model')
plt.plot(time_DTC,temperature_DTC,'.g',label='LST observations')
plt.plot(time_DTC, temperature_modelling, 'r', label='DTC modelling results')
plt.legend(loc='best')
plt.show()

