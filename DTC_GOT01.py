import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
pi=np.pi

# This is the example of GOT01 model writing with Python 3.7

def GOT01(time, *param):

    T0, Ta, tm, ts,deltaT = param[0], param[1], param[2], param[3], param[4]
    k = omega / (pi * np.tan(pi / omega * (ts - tm)))
    temperature = np.zeros(np.shape(time),dtype=np.float)
    Mask=time<ts
    temperature[Mask] = T0 + Ta * np.cos(pi / omega * (time[Mask] - tm))

    #You can obtain the INA08 model by replacing the exponential function with hyperbolic function
    temperature[~Mask] = T0 +deltaT + [Ta * np.cos(pi / omega * (ts - tm))-deltaT] * np.exp(-(time[~Mask]- ts) / k)

    return temperature



# Example data
Latitude=40.3581
Longitude=-3.9803
DOY=139

# DTC model starts from the sunrise to the next sunrise, the values greater than 24 refer to the time in the next day
time_DTC=np.arange(4.75,28.5,1)
temperature_DTC=np.array([283,286,290,295,301,306,309,313,314,314,312,309,305,301,297,294,293,291,290,288,287,286,285,284],dtype=np.float)

# delta is the solar declination
delta = 23.45 * np.sin(2 * pi / 365.0 * (284 + DOY))

# omega is the duration of daytime
omega = 2.0 / 15 * math.acos(-math.tan(Latitude / 180.0 * pi) * math.tan(delta / 180.0 * pi)) * 180.0 / pi

sunrisetime = 12 - omega / 2
sunsettime=12+omega/2



# Setting the initial value
p0 = [np.average(temperature_DTC), np.max(temperature_DTC) - np.min(temperature_DTC), 13.0, sunsettime-1,0]

# Solving the free parameters of GOT09 model. Bounds and max_nfev are optional
popt, pcov = curve_fit(GOT01, time_DTC, temperature_DTC, p0, bounds = ([200, 0, 8,12,-5], [400, 60, 17, 23,5]), max_nfev = 10000)
print(popt)

temperature_modelling = GOT01(time_DTC, popt[0], popt[1], popt[2], popt[3],popt[4])

print('RMSE of GOT01 model:',np.sqrt(np.mean(np.square(temperature_DTC-temperature_modelling))))

plt.title('Example of GOT01 model')
plt.plot(time_DTC,temperature_DTC,'.g',label='LST observations')
plt.plot(time_DTC, temperature_modelling, 'r', label='DTC modelling results')
plt.legend(loc='best')
plt.show()


