"""
    example of GOT09-dT-tao model

    GOT09-dT-tao model is the recommended four-parameter DTC model
    GOT09-dT-tao model is modified from GOT09 model by fixing deltaT as 0 and tao as 0.01

    Ref:
    [1] Göttsche, F.M. and Olesen, F.S., 2009. Modelling the effect of optical thickness on diurnal cycles of land surface temperature. Remote Sensing of Environment, 113(11), pp.2306-2316.
        https://www.sciencedirect.com/science/article/np.pii/S0034425709001850
    [2] Hong, F., Zhan, W., Göttsche, F.M., Liu, Z., Zhou, J., Huang, F., Lai, J. and Li, M., 2018. Comprehensive assessment of four-parameter diurnal land surface temperature cycle models under clear-sky. ISPRS Journal of Photogrammetry and Remote Sensing, 142, pp.190-204.
        https://www.sciencedirect.com/science/article/np.pii/S0924271618301710
"""


import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def GOT09_dT_tao(time, *param):
    """
        Structure of GOT09-dT-tao model

    :param time: time of the day, unit is hour
    :param param:
    :return:
    """

    # parameter of the GOT09-dt-tao model
    # T0 is the residual temperature around sunrise
    # Ta is the temperature amplitude
    # tm is the time when temperature reaches its maximum
    # ts is the time when free attenuation begins
    T0, Ta, tm, ts = param[0], param[1], param[2], param[3]

    theta = np.pi / 12 * (time - tm)
    theta_s = np.pi / 12 * (ts - tm)
    mask_LT = time < ts
    mask_GE = time >= ts

    # fixing tao (represent the total atmospheric optical thickness) as 0.01
    tao = 0.01

    Re, H = 6371, 8.43   # constant values of the Earth's radius and thickness of the atmosphere (“scale height”) in km

    # cosine of the solar zenith angle
    cos_sza = math.sin(delta / 180 * np.pi) * math.sin(latitude / 180 * np.pi) + math.cos(delta / 180 * np.pi) * math.cos(latitude / 180 * np.pi) * np.cos(theta)

    # cosine of the solar zenith angle at the beginning of the free attenuation, i.e., theta = theta_s
    cos_sza_s = math.sin(delta / 180 * np.pi) * math.sin(latitude / 180 * np.pi) + math.cos(delta / 180 * np.pi) * math.cos(latitude / 180 * np.pi) * math.cos(theta_s)
    sin_sza_s = math.sqrt(1 - cos_sza_s * cos_sza_s)

    # the minimum of the cosine of solor zenith angle, i.e., cos(theta) = 1 or time = tm
    cos_sza_min = math.sin(delta / 180 * np.pi) * math.sin(latitude / 180 * np.pi) + math.cos(delta / 180 * np.pi) * math.cos(latitude / 180 * np.pi)

    # the relative optical air mass
    m_val = -Re / H * cos_sza + np.sqrt(pow((Re / H * cos_sza), 2) + 2 * Re / H + 1)

    # the relative optical air mass at the beginning of the free attenuation
    m_sza_s = -Re / H * cos_sza_s + math.sqrt(pow((Re / H * cos_sza_s), 2) + 2 * Re / H + 1)

    # the relative optical air mass at the minimum of the cosine of solar zenith angle
    m_min = -Re / H * cos_sza_min + math.sqrt(pow((Re / H * cos_sza_min), 2) + 2 * Re / H + 1)

    # Eq.(21) in GOT09 paper
    sza_derive_s = math.cos(delta / 180 * np.pi) * math.cos(latitude / 180 * np.pi) * math.sin(theta_s) / sin_sza_s

    # Eq.(24) in GOT09 paper
    m_derive_s = Re / H * sin_sza_s - pow(Re / H, 2) * cos_sza_s * sin_sza_s / math.sqrt(pow(Re / H * cos_sza_s, 2) + 2 * Re / H + 1)

    k1 = 12 / np.pi / sza_derive_s     # first part of k
    k2 = tao * cos_sza_s * m_derive_s   # second part of k

    # assemble to calculate the final k value
    k = k1 * cos_sza_s / (sin_sza_s + k2)

    temperature_day = T0 + Ta * cos_sza[mask_LT] * np.exp(tao * (m_min - m_val[mask_LT])) / cos_sza_min

    temp1 = math.exp(tao * (m_min - m_sza_s)) / cos_sza_min
    temp2 = np.exp(-12 / np.pi / k * (theta[mask_GE] - theta_s))
    temperature_night = T0 + Ta * cos_sza_s * temp1 * temp2

    temperature = np.concatenate((temperature_day, temperature_night))

    return temperature


if __name__ == "__main__":

    global latitude, delta

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
    omega = 2.0 / 15 * math.acos(-math.tan(latitude / 180.0 * np.pi) * math.tan(delta / 180.0 * np.pi)) * 180.0 / np.pi

    sunrise_time = 12 - omega / 2
    sunset_time=12+omega/2

    # Setting the initial value
    p0 = [np.average(temperature_dtc), np.max(temperature_dtc) - np.min(temperature_dtc), 13.0, sunset_time - 1]

    # Solving the free parameters of GOT09-dt-tao model. Bounds and max_nfev are optional
    popt, pcov = curve_fit(GOT09_dT_tao, time_dtc, temperature_dtc, p0, bounds = ([200, 0, 8, 12], [400, 60, 17, 23]), max_nfev = 10000)
    print('fitted parameter', popt)

    temperature_modeling = GOT09_dT_tao(time_dtc, popt[0], popt[1], popt[2], popt[3])

    # calculate the root mean square error of the model fitting
    print('RMSE of GOT09-dT-tao model:', np.sqrt(np.mean(np.square(temperature_dtc - temperature_modeling))))

    # plot the fitted curve
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    plt.plot(time_dtc, temperature_dtc, '.g', label='LST observations')
    plt.plot(time_dtc, temperature_modeling, 'r', label='DTC modelling results')

    ax.tick_params('x', labelsize=12, direction='out', length=4, width=1.5, bottom=True, which='major')
    ax.tick_params('y', labelsize=12, direction='out', length=4, width=1.5, left=True, which='major')
    ax.set_xlabel('Time (h)', size=15)
    ax.set_ylabel('LST', size=15)

    plt.title('Example of GOT09-dT-tao model', fontsize=18)
    plt.legend(loc='best')
    plt.show()



