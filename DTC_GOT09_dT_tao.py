import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
pi=np.pi

# This is the example of GOT09-dT-tao model writing with Python 3.7

# GOT09-dT-tao model is the recommended four-parameter DTC model
# GOT09-dT-tao model is modified from GOT09 model by fixing deltaT as 0 and tao as 0.01
def GOT09_dT_tao(time, *param):
    T0, Ta, tm, ts = param[0], param[1], param[2], param[3]

    theta = pi / 12 * (time - tm)
    theta_s = pi / 12 * (ts - tm)
    mask_LT = time < ts
    mask_GE = time >= ts

    #fixing tao (represent the total atmospheric optical thickness) as 0.01
    tao = 0.01

    Re, H = 6371, 8.43

    cos_sza = math.sin(delta / 180 * pi) * math.sin(Latitude / 180 * pi) + math.cos(delta / 180 * pi) * math.cos(
        Latitude / 180 * pi) * np.cos(theta)
    cos_sza_s = math.sin(delta / 180 * pi) * math.sin(Latitude / 180 * pi) + math.cos(delta / 180 * pi) * math.cos(
        Latitude / 180 * pi) * math.cos(theta_s)
    sin_sza_s = math.sqrt(1 - cos_sza_s * cos_sza_s)
    cos_sza_min = math.sin(delta / 180 * pi) * math.sin(Latitude / 180 * pi) + math.cos(delta / 180 * pi) * math.cos(
        Latitude / 180 * pi)

    m_val = -Re / H * cos_sza + np.sqrt(pow((Re / H * cos_sza), 2) + 2 * Re / H + 1)
    m_sza_s = -Re / H * cos_sza_s + math.sqrt(pow((Re / H * cos_sza_s), 2) + 2 * Re / H + 1)
    m_min = -Re / H * cos_sza_min + math.sqrt(pow((Re / H * cos_sza_min), 2) + 2 * Re / H + 1)

    sza_derive_s = math.cos(delta / 180 * pi) * math.cos(Latitude / 180 * pi) * math.sin(theta_s) / math.sqrt(
        1 - cos_sza_s * cos_sza_s)
    m_derive_s = Re / H * sin_sza_s - pow(Re / H, 2) * cos_sza_s * sin_sza_s / math.sqrt(
        pow(Re / H * cos_sza_s, 2) + 2 * Re / H + 1)

    k1 = 12 / pi / sza_derive_s
    k2 = tao * cos_sza_s * m_derive_s
    k = k1 * cos_sza_s / (sin_sza_s + k2)

    temperature1 = T0 + Ta * cos_sza[mask_LT] * np.exp(tao * (m_min - m_val[mask_LT])) / cos_sza_min

    temp1 = math.exp(tao * (m_min - m_sza_s)) / cos_sza_min
    temp2 = np.exp(-12 / pi / k * (theta[mask_GE] - theta_s))
    temperature2 = T0 + Ta * cos_sza_s * temp1 * temp2

    temperature = np.concatenate((temperature1, temperature2))

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
p0 = [np.average(temperature_DTC), np.max(temperature_DTC) - np.min(temperature_DTC), 13.0, sunsettime-1]

# Solving the free parameters of GOT09 model. Bounds and max_nfev are optional
popt, pcov = curve_fit(GOT09_dT_tao, time_DTC, temperature_DTC, p0, bounds = ([200, 0, 8,12], [400, 60, 17, 23]), max_nfev = 10000)
print(popt)

temperature_modelling = GOT09_dT_tao(time_DTC, popt[0], popt[1], popt[2], popt[3])

print('RMSE of GOT09-dT-tao model:',np.sqrt(np.mean(np.square(temperature_DTC-temperature_modelling))))

plt.title('Example of GOT09-dT-tao model')
plt.plot(time_DTC,temperature_DTC,'.g',label='LST observations')
plt.plot(time_DTC, temperature_modelling, 'r', label='DTC modelling results')
plt.legend(loc='best')
plt.show()


