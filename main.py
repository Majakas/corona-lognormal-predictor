import csv
import math
from os import path

import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters


def model_deaths(coeffs, data):
    """
    Fits alpha given values for mu and sigma. Returns the predicted number of cases.
    """
    mu, sigma = coeffs["mu"], coeffs["sigma"]
    distribution = lognorm(sigma, 0, math.exp(mu))
    distribution_cdf = [distribution.cdf(tau) for tau in range(len(data.cases) + 1)]
    dps = np.diff(distribution_cdf)

    theta = np.zeros_like(data.deaths)
    for j in range(len(data.cases)):
        for tau in range(j + 1):
            theta[j] += dps[tau] * data.cases[j - tau]

    alpha = np.sum(np.multiply(np.multiply(data.deaths, theta), data.deaths**0.5))
    alpha = alpha / np.sum(np.multiply(theta**2, data.deaths**0.5))
    return alpha, alpha*theta


def lognormal_residue(coeffs, data):
    """
    Calculates the residue for the lognormal distribution fit
    """
    alpha, model = model_deaths(coeffs, data)
    return data.deaths - model


class Data:
    """
    Wrapper class for the case and death statistics.
    """
    def __init__(self, case_statistics, death_statistics):
        self.cases = case_statistics
        self.deaths = death_statistics
        if case_statistics is not None:
            self.cases = case_statistics
        if death_statistics is not None:
            self.deaths = death_statistics
        if self.cases is None:
            self.cases = np.zeros_like(self.deaths, dtype=int)
        if self.deaths is None:
            self.deaths = np.zeros_like(self.cases, dtype=int)

    def combine(self, new_case_statistics=None, new_death_statistics=None):
        """
        Adds case or death statistics to the country if they weren't defined during the object's declaration
        """
        if new_case_statistics is not None:
            self.cases = self.cases + new_case_statistics
        if new_death_statistics is not None:
            self.deaths = self.deaths + new_death_statistics


class Country:
    """
    Stores the case/death statistics for a country, enables doing mode fitting for said statistics. The model uses the
    case statistics to predict the deaths by assuming that every infected person has probability alpha of dying and
    that dying follows a log-normal distribution with two parameters mu and sigma. The values for alpha, mu, and sigma
    are fitted from the existing data for deaths.
    """
    def __init__(self, country_name, case_statistics=None, death_statistics=None):
        self.country_name = country_name
        self.recorded_days = 0
        self.data = Data(case_statistics, death_statistics)
        self.total_cases = 0
        self.total_deaths = 0
        self.total_cases = np.sum(self.data.cases)
        self.total_deaths = np.sum(self.data.deaths)

    def combine(self, new_case_statistics=None, new_death_statistics=None):
        """
        Adds case or death statistics to the country if they weren't defined during the object's declaration
        """
        self.data.combine(new_case_statistics, new_death_statistics)
        self.total_cases = np.sum(self.data.cases)
        self.total_deaths = np.sum(self.data.deaths)

    def plot_model(self, coeffs, residuals):
        """
        Plots the fitted death curve and the predicted death statistics for a country.
        """
        mu, sigma = coeffs["mu"].value, coeffs["sigma"].value
        alpha, model = model_deaths(coeffs, self.data)
        residuals_squared = np.sum(residuals**2)

        distribution = lognorm(sigma, 0, math.exp(mu))

        fig, axs = plt.subplots(2)

        x = np.linspace(0, 40, num=300)
        y = distribution.pdf(x)
        peak_x = x[np.argmax(y)]
        print(f"Model statistics: {self.country_name}")
        print(f"\tPeak death at day {peak_x:.4f}")
        print(f"\tmu = {mu:.6f}")
        print(f"\tsigma = {sigma:.6f}")
        print(f"\talpha = {alpha:.6f}")
        print(f"\tres^2 = {residuals_squared:.6f}")

        fig.suptitle(self.country_name)

        axs[0].set_title("Death distribution")
        axs[0].plot(x, y)
        axs[0].xaxis.set_major_locator(plt.MultipleLocator(5))
        axs[0].set_ylabel("Probability")
        axs[0].grid()
        axs[0].annotate(f"$\\mu$={mu:.6f}\n$\\sigma$={sigma:.6f}\n$\\alpha$={alpha:.6f}\nres$^2$={residuals_squared:.3f}", xy=(0.7, 0.5), xycoords='axes fraction')

        axs[1].plot(self.data.deaths, label="Recorded")
        axs[1].plot(model, label="Model")
        axs[1].set_xlabel("Number of days")
        axs[1].set_ylabel("Deaths")
        axs[1].grid()
        axs[1].xaxis.set_major_locator(plt.MultipleLocator(5))
        axs[1].legend()

    def fit_model_least_squares(self):
        """
        Fits the values of alpha, mu, sigma using non-linear least squares method. Since residuals are linear in alpha,
        alpha can be found analytically.
        """
        params = Parameters()
        params.add('mu', value=2., min=0.01, max=3., brute_step=0.4)
        params.add('sigma', value=0.5, min=0.01, max=4., brute_step=0.4)

        fit = minimize(lognormal_residue, params, args=(self.data,))
        coeffs = fit.params
        residuals = fit.residual

        self.plot_model(coeffs, residuals)

    def print_statistics(self):
        print(f"Country name: {self.country_name}\n\tTotal cases: {self.total_cases}\n\tTotal deaths: {self.total_deaths}")


def moving_average(x, n):
    """
    Calculates the moving average of numpy array with the window width being n. n has to be odd.
    """
    if n % 2 == 0:
        print("Moving average was passed a non-odd width n!")
        return None
    k = n//2
    ret = np.zeros_like(x)
    for i in range(len(ret)):
        for j in range(max(0, i - k), min(len(ret), i + k + 1)):
            ret[i] += x[j]/(min(len(ret), j + k + 1) - max(0, j - k))
    return ret


def sparse_csv_row(row, n):
    country_data = np.array(row[4:]).astype(np.float32)
    country_data = np.diff(country_data, prepend=0)
    country_data = moving_average(country_data, n)
    country_prefix = row[0]
    country_name = row[1]

    gets_added = False
    if country_prefix == "" or country_name == "Canada" or country_name == "China":
        if country_prefix != "Grand Princess" and country_prefix != "Diamond Princess":
            gets_added = True
    if country_prefix == "Hubei":
        country_name = "Hubei"

    return country_data, country_prefix, country_name, gets_added


def read_input(filename_confirmed_global_cases, filename_confirmed_global_deaths, countries):
    n = 3
    with open(filename_confirmed_global_cases) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        skip_header = True
        for row in csv_reader:
            if skip_header:
                skip_header = False
                continue

            country_cases, country_prefix, country_name, gets_added = sparse_csv_row(row, n)

            if gets_added:
                if country_name not in countries:
                    countries[country_name] = Country(country_name, case_statistics=country_cases)
                else:
                    countries[country_name].combine(new_case_statistics=country_cases)

    with open(filename_confirmed_global_deaths) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        skip_header = True
        for row in csv_reader:
            if skip_header:
                skip_header = False
                continue

            country_deaths, country_prefix, country_name, gets_added = sparse_csv_row(row, n)

            if gets_added:
                if country_name not in countries:
                    countries[country_name] = Country(country_name, death_statistics=country_deaths)
                else:
                    countries[country_name].combine(new_death_statistics=country_deaths)


def main():
    filename_confirmed_global_cases = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    if not path.exists(filename_confirmed_global_cases):
        filename_confirmed_global_cases = "time_series_covid19_confirmed_global.csv"
    filename_confirmed_global_deaths = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    if not path.exists(filename_confirmed_global_deaths):
        filename_confirmed_global_deaths = "time_series_covid19_deaths_global.csv"

    if not path.exists(filename_confirmed_global_cases) or not path.exists(filename_confirmed_global_deaths):
        print("Couldn't find the data csv files!")
        return -1

    countries = {}

    read_input(filename_confirmed_global_cases, filename_confirmed_global_deaths, countries)

    country = countries["Germany"]
    country.print_statistics()
    country.fit_model_least_squares()


main()
plt.show()
