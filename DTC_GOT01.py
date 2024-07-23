"""
    example of GOT01 model

    Ref:
    Göttsche, F. M., & Olesen, F. S. (2001). Modelling of diurnal cycles of brightness temperature extracted from METEOSAT data. Remote Sensing of Environment, 76(3), 337-348.
    https://www.sciencedirect.com/science/article/pii/S0034425700002145
"""

import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def GOT01(time, *param):
    """
        Structure of the GOT01 model

    :param time: time of the day, unit is hour
    :param param:
    :return:
    """

    # parameter of the GOT01 model
    # T0 is the residual temperature around sunrise
    # Ta is the temperature amplitude
    # tm is the time when temperature reaches its maximum
    # ts is the time when free attenuation begins
    T0, Ta, tm, ts, deltaT = param[0], param[1], param[2], param[3], param[4]

    k = omega / (np.pi * np.tan(np.pi / omega * (ts - tm)))
    temperature = np.zeros(np.shape(time), dtype=float)

    mask = time < ts    # mask to separating the daytime and nighttime segments

    temperature[mask] = T0 + Ta * np.cos(np.pi / omega * (time[mask] - tm))

    # You can obtain the INA08 model by replacing the exponential function with hyperbolic function
    temperature[~mask] = T0 + deltaT + [Ta * np.cos(np.pi / omega * (ts - tm))-deltaT] * np.exp(-(time[~mask] - ts) / k)

    return temperature


if __name__ == "__main__":

    # Example data
    latitude = 40.3581
    longitude = -3.9803
    doy = 139

    # DTC model starts from the sunrise to the next sunrise, the values greater than 24 refer to the time in the next day
    time_dtc = np.arange(4.75, 28.5, 1)
    temperature_dtc = np.array(object=[283, 286, 290, 295, 301, 306, 309,
                                       313, 314, 314, 312, 309, 305, 301,
                                       297, 294, 293, 291, 290, 288, 287,
                                       286, 285, 284],
                               dtype=float)

    # delta is the solar declination
    delta = 23.45 * np.sin(2 * np.pi / 365.0 * (284 + doy))

    # omega is the duration of daytime
    global omega
    omega = 2.0 / 15 * math.acos(-math.tan(latitude / 180.0 * np.pi) * math.tan(delta / 180.0 * np.pi)) * 180.0 / np.pi

    sunrise_time = 12 - omega / 2
    sunset_time = 12 + omega / 2

    # Setting the initial value
    p0 = [np.average(temperature_dtc), np.max(temperature_dtc) - np.min(temperature_dtc), 13.0, sunset_time - 1, 0]

    # Solving the free parameters of GOT01 model. Bounds and max_nfev are optional
    popt, pcov = curve_fit(GOT01, time_dtc, temperature_dtc, p0,
                           bounds=([200, 0, 8, 12, -5], [400, 60, 17, 23, 5]),
                           max_nfev = 10000)
    print('fitted parameter', popt)

    # The fitted GOT01 curve
    temperature_modeling = GOT01(time_dtc, popt[0], popt[1], popt[2], popt[3], popt[4])

    # calculate the root mean square error of the model fitting
    print('RMSE of GOT01 model:', np.sqrt(np.mean(np.square(temperature_dtc - temperature_modeling))))

    # plot the fitted curve
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    plt.plot(time_dtc, temperature_dtc, '.g', label='LST observations')
    plt.plot(time_dtc, temperature_modeling, 'r', label='DTC modelling results')

    ax.tick_params('x', labelsize=12, direction='out', length=4, width=1.5, bottom=True, which='major')
    ax.tick_params('y', labelsize=12, direction='out', length=4, width=1.5, left=True, which='major')
    ax.set_xlabel('Time (h)', size=15)
    ax.set_ylabel('LST', size=15)

    plt.title('Example of GOT01 model', fontsize=18)
    plt.legend(loc='best')
    plt.show()


