import h5py
import numpy as np
from basic_functions import *

FNAME = 'data_seed_7.h5'
f = h5py.File(FNAME, "r")
from scipy.stats import norm, beta

n_events = 100000
chirp_mass_prior_bounds = np.array([1,1.4])
log_q_prior_bounds = np.log(np.array([0.7, 0.95]))
z_by_10_prior_pdf = beta(3,9)
z_array = np.linspace(0.0001,10.,100)
effective_distance_population = f['effective distance population'][:]
H0 = np.linspace(70.4, 70.6, 11)
log_likelihood_all_events = []
for H_0_ in H0:
    log_likelihood = 0
    for n_events_ in range(n_events):
        print (H_0_, n_events_)
        luminosity_distance_array = luminosity_distance(z_array, H_0_, 0.3)
        cosmo_true_z_luminosity_distance_list = np.array([z_array, luminosity_distance_array])
        z_samples = f['z_samples'+str(n_events_)][:]
        mc_samples = f['m_chirp_samples'+str(n_events_)][:]
        log_q_samples = f['log_q_samples'+str(n_events_)][:]
        prior_z_samples = z_by_10_prior_pdf.pdf(z_samples/10)
        mask = mc_samples < chirp_mass_prior_bounds[0]
        prior_z_samples[mask] = 0
        mask = mc_samples > chirp_mass_prior_bounds[1] 
        prior_z_samples[mask] = 0
        mask = log_q_samples >  log_q_prior_bounds[1]
        prior_z_samples[mask] = 0
        mask = log_q_samples < log_q_prior_bounds[0]
        prior_z_samples[mask] = 0
        log_D_L_mean = np.log(effective_distance_population[n_events_])
        log_D_L_sigma = np.array(f['log_effective_distance_error'+str(n_events_)])
        log_D_L_samples = np.log(np.interp(z_samples, cosmo_true_z_luminosity_distance_list[0], cosmo_true_z_luminosity_distance_list[1]))
        integrated_luminosity_distance_term = norm(log_D_L_mean, log_D_L_sigma).pdf(log_D_L_samples)
        log_likelihood += np.log(np.sum(np.exp(np.log(prior_z_samples) + np.log(integrated_luminosity_distance_term)- log_D_L_samples )))#- log_D_L_samples
    log_likelihood_all_events.append(log_likelihood)
f.close()

plt.plot(H0, np.exp(log_likelihood_all_events-np.max(log_likelihood_all_events)))  
plt.xlabel('H0')
plt.ylabel('Likelihood')
plt.axvline(70.5, color='r')
plt.savefig('100000_events_likelihood_without_noise_H0_seed_71.png')