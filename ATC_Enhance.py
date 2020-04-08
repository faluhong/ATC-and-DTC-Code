import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from pandas import *
import os
pi=np.pi


# Original ATC model with single sinusoidal function
def ATC_Original(Day,*param):
    MAST,YAST,theta=param[0],param[1],param[2]
    LST=MAST+YAST*np.sin(2*pi*Day/LenDay+theta)
    return LST

# Enhanced ATC model with the incorporation of air temperature
def ATC_enhance(Xdata,*param):
    Day, DeltaTemperature = Xdata
    Omega=2*pi*Day/LenDay

    MAST,YAST,theta,k= param[0], param[1], param[2], param[3]
    LST = MAST + YAST * np.sin(Omega + theta)+k*DeltaTemperature
    return LST


ExcelFile_Path=os.getcwd()
ExcelFile_FileName=ExcelFile_Path+'\\'+'Example_ATC_data.xlsx'

ExcelFile = pd.ExcelFile(ExcelFile_FileName)
ExcelSheet=ExcelFile.parse('Day')

DOY=ExcelSheet['DOY'].values
LenDay=len(DOY)

ClearMask= ExcelSheet['Cloud_free_Flag'].values

LSTAll= ExcelSheet['LST_Obs']
# Cloud-free LSTs are used to drive the ATC model
LST_Cloud_free=LSTAll[ClearMask]
# Under-cloud LSTs are used to evaluate the accuracy of ATC model
LST_Under_Cloud=LSTAll[~ClearMask]

# Air temperature cloud be obtained from in-situ measurements or reanalysis data
AirTemperatureAll= ExcelSheet['Air_temperature_Obs']


# Modelling with original ATC model
p0_OriATC = [np.nanmean(LST_Cloud_free), np.nanmax(LST_Cloud_free) - np.nanmin(LST_Cloud_free), 1.5 * pi]
popt_ori, pcov = curve_fit(ATC_Original, DOY[ClearMask], LST_Cloud_free, p0_OriATC)
LST_OriATC=ATC_Original(DOY,popt_ori[0],popt_ori[1],popt_ori[2])


# Modelling with enhanced ATC model
# First step: calculating the DeltaTemperature
p0_OriATC = [np.nanmean(AirTemperatureAll), np.nanmax(AirTemperatureAll) - np.nanmin(AirTemperatureAll), 1.5 * pi]
popt_ori, pcov = curve_fit(ATC_Original, DOY, AirTemperatureAll, p0_OriATC)
AirTemperature_Ori_ATC=ATC_Original(DOY,popt_ori[0],popt_ori[1],popt_ori[2])

DeltaTemperature=AirTemperatureAll-AirTemperature_Ori_ATC

# Second step: solving the parameters of enhanced ATC model
Xdata_Mask = np.vstack([DOY[ClearMask], DeltaTemperature[ClearMask]])
Xdata = np.vstack([DOY, DeltaTemperature])

p0_enhanceATC = [np.nanmean(LST_Cloud_free), np.nanmax(LST_Cloud_free) - np.nanmin(LST_Cloud_free), 1.5 * pi, 1.5]
popt_enhance, pcov = curve_fit(ATC_enhance, Xdata_Mask, LST_Cloud_free, p0_enhanceATC)

LST_EnhanceATC = ATC_enhance(Xdata, popt_enhance[0], popt_enhance[1], popt_enhance[2], popt_enhance[3])


# Error printing
print('Error of original ATC model')
print('Bias:',np.nanmean(LST_OriATC[~ClearMask]-LST_Under_Cloud))
print('MAE:',np.nanmean(np.abs(LST_OriATC[~ClearMask] - LST_Under_Cloud)))

print('Error of Enhanced ATC model')
print('Bias:',np.nanmean(LST_EnhanceATC[~ClearMask] - LST_Under_Cloud))
print('MAE:',np.nanmean(np.abs(LST_EnhanceATC[~ClearMask] - LST_Under_Cloud)))

# Results plotting
plt.plot(DOY[ClearMask], LST_Cloud_free, '^', color='blue', label='Cloud-free LSTs')
plt.plot(DOY[~ClearMask], LST_Under_Cloud, '^', color='green', label='Under-cloud LSTs')
plt.plot(DOY,LST_OriATC,'dodgerblue',label='Original ATC modelling results')
plt.plot(DOY,AirTemperatureAll,'dimgrey',label='Air temperature')
plt.plot(DOY,LST_EnhanceATC,'r',label='Enhanced ATC modelling results')
plt.legend(loc='best')
plt.show()

