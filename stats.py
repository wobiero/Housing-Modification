
# =====================================================================
# FILE: stats.py
# Sampling and generic summary/cumulative helpers.
# =====================================================================
import numpy as np
import streamlit as st

def gamma_stats(mean, std_dev, size=1000):
    """Sample a gamma distribution parameterised by mean and std dev."""
    if mean <= 0 or std_dev <= 0:
        raise ValueError("Mean and standard deviation must be positive for gamma distribution")
    shape = (mean / std_dev) ** 2
    scale = (std_dev ** 2) / mean
    simulated_data = np.random.gamma(shape, scale, size=size)
    mean_value = np.mean(simulated_data)
    confidence_interval = np.percentile(simulated_data, [2.5, 97.5])
    return mean_value, confidence_interval, simulated_data, shape, scale

def beta_stats(mean, std_dev, num_samples=1000):
    """Sample a beta distribution; inputs are clamped so live slider values
    can't push mean/std_dev out of bounds and crash the app."""
    eps = 1e-6
    mean = min(max(mean, eps), 1 - eps)
    max_sd = np.sqrt(mean * (1 - mean))
    std_dev = min(max(std_dev, eps), max_sd - eps)
    variance = std_dev ** 2
    nu = mean * (1 - mean) / variance - 1
    alpha = mean * nu
    beta_param = (1 - mean) * nu
    return np.random.beta(alpha, beta_param, num_samples)

def summarize(samples_by_year):
    """Return (mean, lower95, upper95) lists across years."""
    mean = [np.mean(x) for x in samples_by_year]
    lower = [np.percentile(x, 2.5) for x in samples_by_year]
    upper = [np.percentile(x, 97.5) for x in samples_by_year]
    return mean, lower, upper

def cumulate(samples_by_year):
    """Cumulative sum over years (per sim), then summarize."""
    cum = [[sum(vals) for vals in zip(*samples_by_year[:i + 1])]
           for i in range(len(samples_by_year))]
    m, lo, hi = summarize(cum)
    return cum, m, lo, hi

def scale_cases(cases_by_year, sampler):
    """Multiply each sim's yearly value by an independent random draw."""
    return [[c * sampler() for c in year_vals] for year_vals in cases_by_year]

def add_metric(data, name, samples, cum_name=None):
    """Add the six standard columns (level + cumulative, mean/lower/upper)."""
    m, lo, hi = summarize(samples)
    _, cm, clo, chi = cumulate(samples)
    data[name] = m
    data[f'{name}_lower'] = lo
    data[f'{name}_upper'] = hi
    cn = cum_name or f'{name}_cum'
    data[cn] = cm
    data[f'{cn}_lower'] = clo
    data[f'{cn}_upper'] = chi

