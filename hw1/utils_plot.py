from typing import List
import numpy as np
from utils_data_reader import prep_data_range
import matplotlib.pyplot as plt

def calc_date_index(month:int, date:int, year:int):
    # check if leap year
    leap = False if year%4 else True
    # days in each month
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # calculate the index
    return (sum(days[:month-1]) + date + (1 if leap and month > 2 else 0) - 1)

def calc_hour_index(month:int, date:int, year:int):
    return calc_date_index(month, date, year) * 24

# year = 2019
# date_idx = calc_date_index(12, 31, year)
# hour_idx = calc_hour_index(12, 31, year)
# print(f"Date index for 31st December {year} is {date_idx}")
# print(f"Hour index for 31st December {year} is {hour_idx}")

# year = 2020
# date_idx = calc_date_index(12, 31, year)
# hour_idx = calc_hour_index(12, 31, year)
# print(f"Date index for 31st December {year} is {date_idx}")
# print(f"Hour index for 31st December {year} is {hour_idx}")


def get_rho_at_date(data:np.ndarray, month:int, date:int, year:int):
    """
    Returns the density data for the selected date
    @ data: the density data for the selected year
    @ month: the month of the date
    @ date: the date of the month
    @ year: the year
    returns: the density data for the selected date (24, 20, 36) 
    """
    date_idx = calc_date_index(month, date, year)
    return data[date_idx]


# find the index of the element in the list that is closest to the value
def get_altitude_index(altitude:List[float], alt:float):
    return min(range(len(altitude)), key=lambda i: abs(altitude[i]-alt))


def plot_rho_24(rho_date:np.ndarray, alt:float):
    """
    Plots the density data for the selected hour and altitude
    @ rho_date: density data for the selected date
    @ alt: the altitude to be plotted    
    """
    localSolarTimes = np.linspace(0, 24, 24) # 24 
    latitudes = np.linspace(-87.5, 87.5, 20) # 20
    altitudes = np.linspace(100, 800, 36)    # 36

    altitude_index = get_altitude_index(altitudes, alt)

    rho = rho_date[:, :, altitude_index]
    rho = np.power(10, rho)

    # Plot the data for the selected hour and altitude
    plt.figure(figsize=(10, 6))
    plt.contourf(localSolarTimes, latitudes, rho.T, 100, cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel('Local Solar Time (Longitude)')
    plt.ylabel('Latitude')
    plt.title(f'Density at Altitude {altitudes[altitude_index]:.0f} km on January 1st, 2017')
    plt.show()


if __name__ == "__main__":
    rho_2017 = prep_data_range([2017])
    rho_2017_01_01 = get_rho_at_date(rho_2017, 1, 1, 2017)
    print(rho_2017_01_01.shape)
    plot_rho_24(rho_2017_01_01, 500)

    # for alt in range(100, 800, 50):
    #     plot_rho_24(rho_2017_01_01, alt)


