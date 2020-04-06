import csv
import math
from os import path
from copy import deepcopy
import random

import numpy as np
from scipy.stats import lognorm, linregress
from scipy.stats import binom
from scipy.integrate import odeint
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
        self.cases = None
        self.deaths = None
        self.x = None
        if case_statistics is not None:
            self.cases = case_statistics
            self.x = np.arange(0, len(self.cases))
        if death_statistics is not None:
            self.deaths = death_statistics
            self.x = np.arange(0, len(self.cases))
        if self.cases is None:
            self.cases = np.zeros_like(self.deaths, dtype=np.float32)
        if self.deaths is None:
            self.deaths = np.zeros_like(self.cases, dtype=np.float32)

    def combine(self, new_case_statistics=None, new_death_statistics=None):
        """
        Adds case or death statistics to the country if they weren't defined during the object's declaration
        """
        if new_case_statistics is not None:
            self.cases = self.cases + new_case_statistics
        if new_death_statistics is not None:
            self.deaths = self.deaths + new_death_statistics

    def append(self, appended_cases=None, appended_deaths=None):
        if appended_cases is not None:
            self.cases = np.concatenate((self.cases, appended_cases))
            if appended_deaths is None:
                self.deaths = np.concatenate((self.deaths, np.zeros_like(appended_cases, dtype=np.float32)))
            self.x = np.arange(0, len(self.cases))
        if appended_deaths is not None:
            self.cases = np.concatenate((self.cases, appended_deaths))
            if appended_cases is None:
                self.deaths = np.concatenate((self.deaths, np.zeros_like(appended_deaths, dtype=np.float32)))
            self.x = np.arange(0, len(self.cases))


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
        self.total_cases = int(np.round(np.sum(self.data.cases)))
        self.total_deaths = int(np.round(np.sum(self.data.deaths)))

        self.fit = None

    def combine(self, new_case_statistics=None, new_death_statistics=None):
        """
        Adds case or death statistics to the country if they weren't defined during the object's declaration
        """
        self.data.combine(new_case_statistics, new_death_statistics)
        self.total_cases = int(np.round(np.sum(self.data.cases)))
        self.total_deaths = int(np.round(np.sum(self.data.deaths)))

    def confidence_test_on_self(self, alpha, mu, sigma, show_cumulative=False, suppress_figure=False):
        D_diff = np.zeros_like(self.data.cases)
        for i in range(1, len(self.data.cases)):
            if self.data.cases[i] > 0:
                b_d = binom(int(self.data.cases[i]), alpha)
                dead = b_d.rvs()
                if dead > 0:
                    death_distribution = lognorm(sigma, 0, math.exp(mu))
                    dead_days = i + death_distribution.rvs(dead)
                    for day in dead_days:
                        if int(day) < len(self.data.cases):
                            D_diff[int(day)] += 1

        data = Data(case_statistics=self.data.cases, death_statistics=D_diff)
        self.fit_model_least_squares(overwrite_data=data)
        mu_guess = self.fit.params["mu"]
        sigma_guess = self.fit.params["sigma"]
        alpha_guess, model = model_deaths(self.fit.params, data)

        print(f"Actual parameters:\n\talpha = {alpha:.6f}\n\tmu = {mu:.6f}\n\tsigma = {sigma:.6f}")
        print(f"Predicted parameters:\n\talpha = {alpha_guess:.6f}\n\tmu = {mu_guess.value:.6f}\n\tsigma = {sigma_guess.value:.6f}")
        print(f"Predicted parameter errors:\n\tmu = {mu_guess.stderr:.6f}\n\tsigma = {sigma_guess.stderr:.6f}")

        if not suppress_figure:
            fig, axs = plt.subplots(2)

            distribution = lognorm(sigma, 0, math.exp(mu))

            x = np.linspace(0, 40, num=300)
            y = distribution.pdf(x)

            distribution_predicted = lognorm(sigma_guess.value, 0, math.exp(mu_guess.value))
            y_predicted = distribution_predicted.pdf(x)

            axs[0].set_title("Death distribution")
            axs[0].plot(x, y, alpha=0.7, lw=2,
                        label=f"actual\n$\\alpha$={alpha:.3f}\n$\\mu$={mu:.3f}\n$\\sigma$={sigma:.3f}")
            axs[0].plot(x, y_predicted, alpha=0.7, lw=2,
                        label=f"predicted\n$\\alpha$={alpha_guess:.3f}\n$\\mu$={mu_guess.value:.3f}$\\pm${mu_guess.stderr:.2f}\n$\\sigma$={sigma_guess.value:.3f}$\\pm${sigma_guess.stderr:.2f}")
            axs[0].xaxis.set_major_locator(plt.MultipleLocator(5))
            axs[0].xaxis.set_minor_locator(plt.MultipleLocator(1))
            axs[0].set_ylabel("Probability")
            axs[0].legend()
            axs[0].grid(linestyle="--")
            for spine in ('top', 'right'):
                axs[0].spines[spine].set_visible(False)

            # self.data = Data(case_statistics=np.cumsum(I_diff[:len(model)]*alpha), death_statistics=np.cumsum(D_diff[:len(model)]))
            # slope, intercept, new_data = self.extrapolate_cases(30, 40)

            if show_cumulative:
                axs[1].plot(self.data.x, np.cumsum(self.data.cases) * alpha, 'b', alpha=0.5, lw=2,
                            label='Cases*alpha')
                axs[1].plot(self.data.x, np.cumsum(D_diff), 'r', alpha=0.5, lw=2, label='Dead')
                axs[1].plot(self.data.x, np.cumsum(model), 'g--', alpha=0.5, lw=2, label='Predicted dead')
            else:
                axs[1].plot(self.data.x, self.data.cases * alpha, 'b', alpha=0.5, lw=2, label='Cases*alpha')
                axs[1].plot(self.data.x, D_diff, 'r', alpha=0.5, lw=2, label='Dead')
                axs[1].plot(self.data.x, model, 'g--', alpha=0.5, lw=2, label='Predicted dead')
            axs[1].set_xlabel('Time /days')
            axs[1].yaxis.set_tick_params(length=0)
            axs[1].xaxis.set_tick_params(length=0)
            l1, l2 = 5, 1
            if len(model) > 100:
                l1, l2 = 10, 2
            if len(model) > 200:
                l1, l2 = 20, 4
            axs[1].xaxis.set_major_locator(plt.MultipleLocator(l1))
            axs[1].xaxis.set_minor_locator(plt.MultipleLocator(l2))

            axs[1].grid(which='major', ls='--')
            axs[1].legend()
            for spine in ('top', 'right'):
                axs[1].spines[spine].set_visible(False)
            plt.show()

        return alpha_guess, mu_guess.value, sigma_guess.value

    def confidence_test(self, N, alpha, mu, sigma, show_cumulative=False, peak_offset_cutoff=np.inf, suppress_figure=False):
        # Total population, N.
        # Initial number of infected and recovered individuals, I0 and R0.
        I0, R0 = 10, 0
        # Everyone else, S0, is susceptible to infection initially.
        S0 = N - I0 - R0
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        beta, gamma = 0.35, 1. / 14
        # A grid of time points (in days)
        t_max = 500
        t = np.arange(0, t_max + 1)
        S, I, R = np.array([S0]), np.array([I0]), np.array([R0])
        I_diff = np.zeros(t_max + 1)
        I_diff[0] = I[0]
        D_diff = np.zeros(t_max + 1)
        for i in range(1, t_max + 1):
            p1 = beta * I[-1] / N
            p2 = gamma
            b_s = binom(S[-1], p1)
            ds = b_s.rvs()
            b_i = binom(I[-1], p2)
            di = b_i.rvs()
            S = np.append(S, S[-1] - ds)
            I = np.append(I, I[-1] + ds - di)
            R = np.append(R, R[-1] + di)
            I_diff[i] = ds

            if ds > 0:
                b_d = binom(ds, alpha)
                dead = b_d.rvs()
                if dead > 0:
                    death_distribution = lognorm(sigma, 0, math.exp(mu))
                    dead_days = i + death_distribution.rvs(dead)
                    for day in dead_days:
                        if int(day) <= t_max:
                            D_diff[int(day)] += 1

        peak_day = t[np.argmax(I_diff)]
        final_day = peak_day + peak_offset_cutoff
        if final_day <= 0:
            print("Epidemic ended too early!")
            return
        if final_day >= t_max:
            final_day = t_max

        data = Data(case_statistics=I_diff[:final_day + 1], death_statistics=D_diff[:final_day + 1])
        self.fit_model_least_squares(overwrite_data=data)
        mu_guess = self.fit.params["mu"]
        sigma_guess = self.fit.params["sigma"]
        alpha_guess, model = model_deaths(self.fit.params, data)

        print(f"Actual parameters:\n\talpha = {alpha:.6f}\n\tmu = {mu:.6f}\n\tsigma = {sigma:.6f}")
        print(f"Predicted parameters:\n\talpha = {alpha_guess:.6f}\n\tmu = {mu_guess.value:.6f}\n\tsigma = {sigma_guess.value:.6f}")
        print(f"Predicted parameter errors:\n\tmu = {mu_guess.stderr:.6f}\n\tsigma = {sigma_guess.stderr:.6f}")

        if not suppress_figure:
            fig, axs = plt.subplots(2)

            distribution = lognorm(sigma, 0, math.exp(mu))

            x = np.linspace(0, 40, num=300)
            y = distribution.pdf(x)

            distribution_predicted = lognorm(sigma_guess.value, 0, math.exp(mu_guess.value))
            y_predicted = distribution_predicted.pdf(x)

            axs[0].set_title("Death distribution")
            axs[0].plot(x, y, alpha=0.7, lw=2, label=f"actual\n$\\alpha$={alpha:.3f}\n$\\mu$={mu:.3f}\n$\\sigma$={sigma:.3f}")
            axs[0].plot(x, y_predicted, alpha=0.7, lw=2, label=f"predicted\n$\\alpha$={alpha_guess:.3f}\n$\\mu$={mu_guess.value:.3f}$\\pm${mu_guess.stderr:.2f}\n$\\sigma$={sigma_guess.value:.3f}$\\pm${sigma_guess.stderr:.2f}")
            axs[0].xaxis.set_major_locator(plt.MultipleLocator(5))
            axs[0].xaxis.set_minor_locator(plt.MultipleLocator(1))
            axs[0].set_ylabel("Probability")
            axs[0].legend()
            axs[0].grid(linestyle="--")
            for spine in ('top', 'right'):
                axs[0].spines[spine].set_visible(False)

            #self.data = Data(case_statistics=np.cumsum(I_diff[:len(model)]*alpha), death_statistics=np.cumsum(D_diff[:len(model)]))
            #slope, intercept, new_data = self.extrapolate_cases(30, 40)

            if show_cumulative:
                axs[1].plot(t[:len(model)], np.cumsum(I_diff[:len(model)]*alpha), 'b', alpha=0.5, lw=2, label='Cases*alpha')
                axs[1].plot(t[:len(model)], np.cumsum(D_diff[:len(model)]), 'r', alpha=0.5, lw=2, label='Dead')
                axs[1].plot(t[:len(model)], np.cumsum(model), 'g--', alpha=0.5, lw=2, label='Predicted dead')
            else:
                axs[1].plot(t[:len(model)], I_diff[:len(model)]*alpha, 'b', alpha=0.5, lw=2, label='Cases*alpha')
                axs[1].plot(t[:len(model)], D_diff[:len(model)], 'r', alpha=0.5, lw=2, label='Dead')
                axs[1].plot(t[:len(model)], model, 'g--', alpha=0.5, lw=2, label='Predicted dead')
            axs[1].set_xlabel('Time /days')
            axs[1].yaxis.set_tick_params(length=0)
            axs[1].xaxis.set_tick_params(length=0)
            l1, l2 = 5, 1
            if len(model) > 100:
                l1, l2 = 10, 2
            if len(model) > 200:
                l1, l2 = 20, 4
            axs[1].xaxis.set_major_locator(plt.MultipleLocator(l1))
            axs[1].xaxis.set_minor_locator(plt.MultipleLocator(l2))

            axs[1].grid(which='major', ls='--')
            axs[1].legend()
            for spine in ('top', 'right'):
                axs[1].spines[spine].set_visible(False)
            plt.show()

        return alpha_guess, mu_guess.value, sigma_guess.value

    def extrapolate_cases(self, start, finish):
        """
        Extrapolates the cases (not the deaths) with an exponential fit using the last 7 days)
        """
        x = np.arange(start, finish)
        y = self.data.cases[start:finish]

        y[y == 0] = 1
        y = np.log(y)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        extrapolated_x = np.arange(start, finish)
        extrapolated_y = slope * extrapolated_x + intercept
        extrapolated_y = np.exp(extrapolated_y)
        new_data = deepcopy(self.data)
        new_data.append(appended_cases=extrapolated_y)
        return slope, intercept, new_data

    def plot_model(self, extrapolate=0, show_cumulative=False):
        """
        Plots the fitted death curve and the predicted death statistics for a country.
        """
        coeffs = self.fit.params
        residuals = self.fit.residual

        mu, sigma = coeffs["mu"], coeffs["sigma"]

        fig, axs = plt.subplots(2)

        if extrapolate >= 0:
            n = extrapolate
            slope, intercept, new_data = self.extrapolate_cases(len(self.data.cases) - n, len(self.data.cases))
            alpha, model = model_deaths(coeffs, new_data)
            print(f"Predicted cases:")
            for i in range(n):
                print(f"\tdeaths on day +{i}: {int(np.round(model[len(self.data.cases) + i]))}")

            x = np.linspace(self.data.x[-n], self.data.x[-1] + n, num=100)
            y = alpha*np.exp(slope * x + intercept)
            if show_cumulative:
                axs[1].plot(x, np.sum(self.data.cases[:-n+1])*alpha + np.cumsum(y)/50*n, "--", alpha=0.7, lw=2, color="C0", label="Extrapolated cases*alpha")
            else:
                axs[1].plot(x, y, "--", alpha=0.7, lw=2, color="C0", label="Extrapolated cases*alpha")

        else:
            alpha, model = model_deaths(coeffs, self.data)
        residuals_squared = np.sum(residuals**2)

        distribution = lognorm(sigma.value, 0, math.exp(mu.value))

        x = np.linspace(0, 40, num=300)
        y = distribution.pdf(x)
        peak_x = x[np.argmax(y)]
        print(f"Model statistics: {self.country_name}")
        print(f"\tPeak death at day {peak_x:.4f}")
        print(f"\tmu = {mu.value:.6f}")
        print(f"\tsigma = {sigma.value:.6f}")
        print(f"\talpha = {alpha:.6f}")
        print(f"\tres^2 = {residuals_squared:.6f}")

        fig.suptitle(self.country_name)

        axs[0].set_title("Death distribution")
        axs[0].plot(x, y, alpha=0.7, lw=2, label=f"$\\mu$={mu.value:.4f}$\\pm${mu.stderr:.3f}\n$\\sigma$={sigma.value:.4f}$\\pm${sigma.stderr:.3f}\n$\\alpha$={alpha:.4f}\nres$^2$={residuals_squared:.3f}")
        axs[0].xaxis.set_major_locator(plt.MultipleLocator(5))
        axs[0].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[0].set_ylabel("Probability")
        axs[0].grid(linestyle="--")
        axs[0].legend()
        for spine in ('top', 'right'):
            axs[0].spines[spine].set_visible(False)

        if show_cumulative:
            axs[1].plot(np.cumsum(self.data.cases*alpha), alpha=0.7, lw=2, label="Cases*alpha", color="C0")
            axs[1].plot(np.cumsum(self.data.deaths), alpha=0.7, lw=2, label="Recorded deaths", color="r")
            axs[1].plot(np.cumsum(model), "--", alpha=0.7, lw=2, color="black", label="Model")
        else:
            axs[1].plot(self.data.cases*alpha, alpha=0.7, lw=2, label="Cases*alpha", color="C0")
            axs[1].plot(self.data.deaths, alpha=0.7, lw=2, label="Recorded deaths", color="r")
            axs[1].plot(model, "--", alpha=0.7, lw=2, color="black", label="Model")
        axs[1].set_xlabel("Number of days")
        axs[1].set_ylabel("Deaths")
        axs[1].grid(linestyle="--")
        axs[1].xaxis.set_major_locator(plt.MultipleLocator(5))
        axs[1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[1].legend()
        for spine in ('top', 'right'):
            axs[1].spines[spine].set_visible(False)

    def fit_model_least_squares(self, mu_guess=2., sigma_guess=0.5, mu_fixed=False, sigma_fixed=False, overwrite_data=None):
        """
        Fits the values of alpha, mu, sigma using non-linear least squares method. Since residuals are linear in alpha,
        alpha can be found analytically.
        """
        params = Parameters()
        params.add('mu', value=mu_guess, min=0.01, max=4., vary=not mu_fixed)
        params.add('sigma', value=sigma_guess, min=0.01, max=4., vary=not sigma_fixed)

        data = self.data
        if overwrite_data is not None:
            data = overwrite_data
        self.fit = minimize(lognormal_residue, params, args=(data,))
        alpha, _ = model_deaths(self.fit.params, data)
        return alpha, self.fit.params["mu"].value, self.fit.params["sigma"].value

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
            country_cases = country_cases[:-1]

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
            country_deaths = country_deaths[:-1]

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

    country = countries["Spain"]
    country.print_statistics()
    alpha0, mu0, sigma0 = country.fit_model_least_squares(mu_guess=1., sigma_guess=0.3, mu_fixed=False, sigma_fixed=False)
    country.plot_model(extrapolate=7, show_cumulative=False)
    plt.show()
    #country.confidence_test(N=1000000, alpha=0.05, mu=2., sigma=0.5, show_cumulative=True, peak_offset_cutoff=50)

    # 0.3025, 2.8645, 2.3906
    #         1.2684, 0.6275

    if True:
        # Plots the mu-sigma phasespace for SIR monte-carlo modelled data set with constant cut-off offset from the cases peak
        n = 100
        mus, sigmas = np.zeros(n), np.zeros(n)
        #N0, alpha0, mu0, sigma0, peak_cutoff0 = 1000000, 0.05, 2., 0.5, 10
        for i in range(n):
            print(f"=====================\nRun #{i}")
            #alpha, mu, sigma = country.confidence_test(N=N0, alpha=alpha0, mu=mu0, sigma=sigma0, show_cumulative=False, peak_offset_cutoff=peak_cutoff0, suppress_figure=False)
            alpha, mu, sigma = country.confidence_test_on_self(alpha=alpha0, mu=mu0, sigma=sigma0, show_cumulative=False, suppress_figure=True)
            mus[i] = mu
            sigmas[i] = sigma
        plt.xlim(0, 4)
        plt.ylim(0, 4)
        plt.plot(mus, sigmas, "bx", alpha=1., label="predicted")
        plt.plot(mu0, sigma0, "rx", markersize=8, label="actual")
        plt.xlabel("$\\mu$")
        plt.ylabel("$\\sigma$")
        #plt.title(f"$\\alpha$={alpha0} $\\mu$={mu0} $\\sigma$={sigma0} $N$={N0}, peak cutoff offset={peak_cutoff0}days")
        plt.title(f"$\\alpha$={alpha0:.6f} $\\mu$={mu0:.6f} $\\sigma$={sigma0:.6f}\t{country.country_name}")
        plt.grid(linestyle="--")

        mu_mean = mus.mean()
        mu_mean_std = mus.std() / np.sqrt(len(mus))
        sigma_mean = sigmas.mean()
        sigma_mean_std = sigmas.std() / np.sqrt(len(sigmas))
        print(f"Predicted mu {mu_mean:.6f}, std {mu_mean_std:.6f}")
        print(f"Predicted sigma {sigma_mean:.6f}, std {sigma_mean_std:.6f}")
        plt.errorbar(mu_mean, sigma_mean, mu_mean_std, sigma_mean_std, alpha=0.7, color="0.3", lw=3, label=f"predicted mean\n$\\mu$={mu_mean:.4f}$\\pm${mu_mean_std:.4f}\n$\\sigma$={sigma_mean:.4f}$\\pm${sigma_mean_std:.4f}")
        plt.errorbar(mu_mean, sigma_mean, mus.std(), sigmas.std(), alpha=0.6, color="0.3", lw=1, label=f"standard deviation\n$\\Delta \\mu$={mu_mean_std:.4f}\n$\\Delta \\sigma$={sigma_mean_std:.4f}")
        plt.legend()

main()
plt.show()
