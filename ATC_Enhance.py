"""
    enhanced annual temperature cycle (ATC) model

    Ref:
    [1] Zou, Z., Zhan, W., Liu, Z., Bechtel, B., Gao, L., Hong, F., Huang, F. and Lai, J., 2018. Enhanced modeling of annual temperature cycles with temporally discrete remotely sensed thermal observations. Remote Sensing, 10(4), p.650.
        https://www.mdpi.com/2072-4292/10/4/650
    [2] Liu, Z., Zhan, W., Lai, J., Hong, F., Quan, J., Bechtel, B., Huang, F. and Zou, Z., 2019. Balancing prediction accuracy and generalization ability: A hybrid framework for modelling the annual dynamics of satellite-derived land surface temperatures. ISPRS Journal of Photogrammetry and Remote Sensing, 151, pp.189-206.
        https://www.sciencedirect.com/science/article/pii/S0924271619300826
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
from os.path import join


def atc_original(day, *param, len_day=365.25):

    """
        Original ATC model with single sinusoidal function

    :param day: array of day of year
    :param param: parameters of the original ATC model
    :param len_day: number of days in a year, 365 or 366, default value is 365.25
    :return:
    """

    # parameter of the original ATC model
    mast, yast, theta = param[0],param[1],param[2]
    lst = mast + yast*np.sin(2 * np.pi * day / len_day + theta)

    return lst


def atc_enhance(xdata, *param, len_day=365.25):
    """
        Enhanced ATC model with the incorporation of air temperature

    :param xdata:   array of DOY and DeltaTemperature
    :param param:   parameters of the enhanced ATC model
    :param len_day: number of days in a year, 365 or 366, default value is 365.25
    :return:
    """

    day, delta_temperature = xdata
    omega = 2 * np.pi * day / len_day

    # parameter of the enhanced ATC model
    mast, yast, theta, k= param[0], param[1], param[2], param[3]
    lst = mast + yast * np.sin(omega + theta) + k * delta_temperature

    return lst


if __name__ == '__main__':

    example_excel_file_path= os.getcwd()
    excel_file_file_name = join(example_excel_file_path, 'Example_ATC_data.xlsx')

    excel_file = pd.ExcelFile(excel_file_file_name)
    excel_sheet = excel_file.parse('Day')

    # Example for the nighttime
    # ExcelSheet=ExcelFile.parse('Night')

    doy = excel_sheet['DOY'].values
    len_day = len(doy)

    clear_mask = excel_sheet['Cloud_free_Flag'].values  # clear sky mask

    lst_all = excel_sheet['LST_Obs']

    # Cloud-free LSTs are used to drive the ATC model
    lst_cloud_free = lst_all[clear_mask]
    # Under-cloud LSTs are used to evaluate the accuracy of ATC model
    lst_under_cloud = lst_all[~clear_mask]

    # Air temperature cloud be obtained from in-situ measurements or reanalysis data
    air_temperature_all = excel_sheet['Air_temperature_Obs']

    # Modeling with original ATC model
    p0_ori_atc = [np.nanmean(lst_cloud_free), np.nanmax(lst_cloud_free) - np.nanmin(lst_cloud_free), 1.5 * np.pi]  # initial value of the parameters
    popt_ori, pcov = curve_fit(atc_original, doy[clear_mask], lst_cloud_free, p0_ori_atc)
    lst_ori_atc = atc_original(doy, popt_ori[0], popt_ori[1], popt_ori[2], len_day)

    # Modeling with enhanced ATC model
    # First step: fitting the air temperature with the original ATC model
    p0_ori_atc = [np.nanmean(air_temperature_all), np.nanmax(air_temperature_all) - np.nanmin(air_temperature_all), 1.5 * np.pi]
    popt_ori, pcov = curve_fit(atc_original, doy, air_temperature_all, p0_ori_atc)
    air_temperature_ori_atc = atc_original(doy, popt_ori[0], popt_ori[1], popt_ori[2], len_day)

    # Second step: calculating the DeltaTemperature
    delta_temperature = air_temperature_all - air_temperature_ori_atc

    # Third step: solving the parameters of enhanced ATC model
    xdata_mask = np.vstack([doy[clear_mask], delta_temperature[clear_mask]])   # get the clear sky data for input
    xdata = np.vstack([doy, delta_temperature])  # all the data for input

    p0_enhance_atc = [np.nanmean(lst_cloud_free), np.nanmax(lst_cloud_free) - np.nanmin(lst_cloud_free), 1.5 * np.pi, 1.5]  # initial value of the parameters
    popt_enhance, pcov = curve_fit(atc_enhance, xdata_mask, lst_cloud_free, p0_enhance_atc)

    lst_enhance_atc = atc_enhance(xdata, popt_enhance[0], popt_enhance[1], popt_enhance[2], popt_enhance[3], len_day)

    # Calculate the errors of the original and enhanced ATC models
    print('Error of original ATC model')
    print('Bias:', np.nanmean(lst_ori_atc[~clear_mask] - lst_under_cloud))
    print('MAE:', np.nanmean(np.abs(lst_ori_atc[~clear_mask] - lst_under_cloud)))

    print('Error of Enhanced ATC model')
    print('Bias:', np.nanmean(lst_enhance_atc[~clear_mask] - lst_under_cloud))
    print('MAE:', np.nanmean(np.abs(lst_enhance_atc[~clear_mask] - lst_under_cloud)))

    # plot the fitting results
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    plt.plot(doy[clear_mask], lst_cloud_free, '^', color='blue', label='Cloud-free LSTs')
    plt.plot(doy[~clear_mask], lst_under_cloud, '^', color='green', label='Under-cloud LSTs')
    plt.plot(doy, lst_ori_atc, 'dodgerblue', label='Original ATC modeling results', linewidth=2)
    plt.plot(doy, air_temperature_all, 'dimgrey', label='Air temperature')
    plt.plot(doy, lst_enhance_atc, 'r', label='Enhanced ATC modeling results', linewidth=2)

    ax.tick_params('x', labelsize=12, direction='out', length=4, width=1.5, bottom=True, which='major')
    ax.tick_params('y', labelsize=12, direction='out', length=4, width=1.5, left=True, which='major')
    ax.set_xlabel('Day of year', size=15)
    ax.set_ylabel('LST', size=15)

    plt.title('Example of Enhanced ATC model', fontsize=18)
    plt.legend(loc='best')
    plt.show()

