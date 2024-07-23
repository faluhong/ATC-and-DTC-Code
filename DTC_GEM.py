"""
    This is the example of GEM-type model (including GEM-eta and GEM-sigma models)
    Both models have four parameters

    Ref:
    [1] Hong, F., Zhan, W., Göttsche, F.M., Liu, Z., Zhou, J., Huang, F., Lai, J. and Li, M., 2018. Comprehensive assessment of four-parameter diurnal land surface temperature cycle models under clear-sky. ISPRS Journal of Photogrammetry and Remote Sensing, 142, pp.190-204.
        https://www.sciencedirect.com/science/article/pii/S0924271618301710
    [2] Zhan, W., Zhou, J., Ju, W., Li, M., Sandholt, I., Voogt, J. and Yu, C., 2014. Remotely sensed soil temperatures beneath snow-free skin-surface using thermal observations from tandem polar-orbiting satellites: An analytical three-time-scale model. Remote Sensing of Environment, 143, pp.1-14.
        https://www.sciencedirect.com/science/article/pii/S0034425713004380
    [3] Huang, F., Zhan, W., Duan, S.B., Ju, W. and Quan, J., 2014. A generic framework for modeling diurnal land surface temperatures with remotely sensed thermal observations under clear sky. Remote sensing of environment, 150, pp.140-151.
        https://www.sciencedirect.com/science/article/pii/S0034425714001655
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad


def GEM_eta(time, *param):
    """
        Structure of GEM-eta model

    :param time: time of the day, unit is second
    :param param:
    :return:
    """

    # parameter of GEM-eta model
    P, T_Ave, eta0, eta1 = param[0], param[1], param[2], param[3]

    h1 = eta0 + eta1 * time

    sum = 0
    for i in range(1, fourier_number + 1):
        m = 1.0 / np.sqrt(i * omega_cal * P * P + np.sqrt(2 * i * omega_cal) * P * h1 + h1 * h1)
        fourier_a = 1.0 / np.pi * quad(fourier_kernel_a, start_integration_point, end_integration_point, args=(i))[0]
        fourier_b = 1.0 / np.pi * quad(fourier_kernel_b, start_integration_point, end_integration_point, args=(i))[0]
        fi = np.arctan(P * np.sqrt(i * omega_cal) / (np.sqrt(2.0) * h1 + P * np.sqrt(i * omega_cal)))
        gt = fourier_a * np.cos(i * omega_cal * time - fi) + fourier_b * np.sin(i * omega_cal * time - fi)

        sum = sum + gt * m

    lst_return = T_Ave + sum

    return lst_return


def GEM_sigma(time, *param):
    """
        Structure of GEM-sigma model

    :param time: time of the day, unit is second
    :param param:
    :return:
    """

    # parameter of GEM-sigma model
    P, T_Ave, h1, sigma = param[0], param[1], param[2], param[3]

    sum = 0
    for i in range(1, fourier_number + 1):

        m = 1.0/np.sqrt(i * omega_cal * P * P + np.sqrt(2 * i * omega_cal) * P * h1 + h1 * h1)
        fourier_a = 1.0 / np.pi * quad(fourier_kernel_a, start_integration_point, end_integration_point, args=(i))[0]
        fourier_b = 1.0 / np.pi * quad(fourier_kernel_b, start_integration_point, end_integration_point, args=(i))[0]
        fi = np.arctan(P * np.sqrt(i * omega_cal) / (np.sqrt(2.0) * h1 + P * np.sqrt(i * omega_cal)))
        gt = fourier_a * np.cos(i * omega_cal * time - fi) + fourier_b * np.sin(i * omega_cal * time - fi)

        sum = sum + gt*m

    lst_return = T_Ave + sigma * (time-43200) + sum

    return lst_return


def fourier_kernel_a(x, n):
    """
        Fourier Kernel A

    :param x: the angle (radian) that the sun rotates in a day, between [0, 2*np.pi]
    :param n: n_th order of Fourier series, passed with the parameter i
    :return:
    """
    solar_constant = 1367  # solar constant

    # cosine of the solar zenith angle
    # (x - np.pi) is to make cosine(solar zenith angle) is maximum at noon, i.e., solar zenith angle is minimum at noon.
    cos_sza = np.cos(delta/180*np.pi) * np.cos(latitude / 180 * np.pi) * np.cos(x - np.pi) + np.sin(delta / 180 * np.pi) * np.sin(latitude / 180 * np.pi)

    atmosphere_transmittance = 1 - 0.2 * np.sqrt(1.0 / cos_sza)

    return (1-albedo)*solar_constant * cos_sza*atmosphere_transmittance * np.cos(n * x)


def fourier_kernel_b(x, n):
    """
        Fourier Kernel B

    :param x: the angle (radian) that the sun rotates in a day, between [0, 2*np.pi]
    :param n: n_th order of Fourier series, passed with the parameter i
    :return:
    """
    solar_constant = 1367  # solar constant

    # cosine of the solar zenith angle
    cos_sza = np.cos(delta / 180*np.pi) * np.cos(latitude / 180 * np.pi) * np.cos(x - np.pi) + np.sin(delta / 180 * np.pi) * np.sin(latitude / 180 * np.pi)

    atmosphere_transmittance = 1 - 0.2 * np.sqrt(1.0 / cos_sza)

    return (1-albedo) * solar_constant*cos_sza * atmosphere_transmittance * np.sin(n * x)


if __name__ == "__main__":

    global delta, latitude, albedo, fourier_number, omega_cal, start_integration_point, end_integration_point

    # Example data
    latitude = 40.3581
    longitude = -3.9803
    doy = 139

    # DTC model starts from the sunrise to the next sunrise, the values greater than 24 refer to the time in the next day
    time_dtc_full = np.arange(4.75, 28.5, 1)
    temperature_full = np.array(object=[283, 286, 290, 295, 301, 306, 309,
                                        313, 314, 314, 312, 309, 305, 301,
                                        297, 294, 293, 291, 290, 288, 287,
                                        286, 285, 284],
                                dtype=float)

    time_dtc = np.array([10.75, 13.75, 22.75, 25.75])
    temperature_dtc = np.array([309, 314, 290, 286], dtype=float)

    # Basic settings of GEM-type model
    fourier_number = 10   # number of items of Fourier series, set to 10 for simple use
    albedo = 0.15   # albedo of the surface

    omega_earth = 2 * np.pi / 86400  # the angular velocity of the earth in one day
    day_number = 1  # number of days for model, usually we model each day independently
    omega_cal = omega_earth / day_number    # the omega used for calculation

    # delta is the solar declination. The unit is degree. Need to convert to radian for computation
    delta = 23.45 * np.sin(2 * np.pi / 365.0 * (284 + doy))

    # get the solar zenith angle when cosine(solar_zenith_angle) = 0.04
    # cosine(solar_zenith_angle) = 0.04 is regarded as
    solar_zenith_angle_004 = np.arccos((0.04 - np.sin(delta / 180 * np.pi) * np.sin(latitude / 180 * np.pi)) / (np.cos(delta / 180 * np.pi) * np.cos(latitude / 180 * np.pi)))

    # get the time range of the day
    start_time_day = 43200 - solar_zenith_angle_004 / omega_earth   # get the start time of day, similar to the sunrise time
    end_time_day = 86400 - start_time_day   # get the end time of the day, similar to the sunset time

    # calculating the integrating range for Fourier Kernel A and Fourier Kernel B
    start_integration_point = start_time_day * omega_earth
    end_integration_point = end_time_day * omega_earth

    # Setting the initial value
    p0 = [1000, np.average(temperature_dtc), 10, 0]

    # Solving the free parameters of GEM-eta model. Bounds and max_nfev are optional
    popt_eta, pcov = curve_fit(GEM_eta, time_dtc * 3600, temperature_dtc, p0, bounds=([100, 200, -100, -100], [5000, 350, 100, 100]), max_nfev=10000)
    temperature_modeling_eta = GEM_eta(time_dtc_full * 3600, popt_eta[0], popt_eta[1], popt_eta[2], popt_eta[3])
    print('fitted parameter of GEM-eta model', popt_eta)

    # Solving the free parameters of GEM-sigma model. Bounds and max_nfev are optional
    popt_sigma, pcov = curve_fit(GEM_sigma, time_dtc*3600, temperature_dtc, p0,
                           bounds=([100, 200, -100, -100], [5000, 350, 100, 100]),
                           max_nfev=10000)
    temperature_modeling_sigma = GEM_sigma(time_dtc_full * 3600, popt_sigma[0], popt_sigma[1], popt_sigma[2], popt_sigma[3])
    print('fitted parameter of GEM-sigma model', popt_sigma)

    print('RMSE of GEM-eta model:', np.sqrt(np.mean(np.square(temperature_full - temperature_modeling_eta))))
    print('RMSE of GEM-sigma model:', np.sqrt(np.mean(np.square(temperature_full - temperature_modeling_sigma))))

    # plot the fitted curve
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    plt.scatter(time_dtc, temperature_dtc, color='g', marker="s", s=70, label='LST fitting points')
    plt.scatter(time_dtc_full, temperature_full, color='black', marker=".", s=80, label='LST observation points')
    plt.plot(time_dtc_full, temperature_modeling_eta, 'r', label='GEM-eta modeling results', linewidth=2)
    plt.plot(time_dtc_full, temperature_modeling_sigma, 'b', label='GEM-sigma modeling results', linewidth=2)

    ax.tick_params('x', labelsize=12, direction='out', length=4, width=1.5, bottom=True, which='major')
    ax.tick_params('y', labelsize=12, direction='out', length=4, width=1.5, left=True, which='major')
    ax.set_xlabel('Time (h)', size=15)
    ax.set_ylabel('LST', size=15)

    plt.title('Example of GEM-type model', fontsize=18)
    plt.legend(loc='best')
    plt.show()

