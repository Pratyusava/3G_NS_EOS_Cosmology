from basic_functions import *
import h5py

n_events = 100000
H0_injected = 70.5
Omega_m_injected = 0.3
np.random.seed(7)

z_array = np.linspace(0.0001,10.,100)
luminosity_distance_array = luminosity_distance(z_array, H0_injected, Omega_m_injected)
cosmo_true_z_luminosity_distance_list = np.array([z_array, luminosity_distance_array])

m_chirp_population = np.random.uniform(1, 1.4, n_events)
q_population = np.random.uniform(0.7, 0.95, n_events)
lambda_tilde_population = lambda_tilde_from_m_chirp_q(m_chirp_population, q_population)
z_population = np.random.beta(3,9,n_events)*10
theta_population = np.ones(n_events)
effective_distance_population = np.interp(z_population, cosmo_true_z_luminosity_distance_list[0], cosmo_true_z_luminosity_distance_list[1])/theta_population
m_chirp_z_population = m_chirp_population * (1+z_population)

freq, psd = np.loadtxt('CE_psd.txt', unpack=True)

keys = ['m_chirp_z', 'q', 'lambda_tilde', 'effective_distance', 'z']
snr_population = np.zeros(n_events)
for n_events_ in range(n_events):
    print (n_events_)
    pars_ = dict(zip(keys, [m_chirp_z_population[n_events_], q_population[n_events_], lambda_tilde_population[n_events_], effective_distance_population[n_events_], z_population[n_events_]]))
    waveform_ = Waveform(pars_, freq)
    snr_population[n_events_]  = waveform_.snr(psd)

FNAME = 'data_seed_7.h5'
log_q_array = np.linspace(np.log(0.3),np.log(3),50)
log_lambda_tilde_array = np.linspace(4,10,50)
m_chirp_grid = np.loadtxt('m_chirp_grid_using_SLY.txt')

redshift_measured = np.zeros(n_events)
log_effective_distance_error = np.zeros(n_events)
with h5py.File(FNAME, "w") as f:
    for n_events_ in range(n_events):
        print (n_events_)
        pars_ = dict(zip(keys, [m_chirp_z_population[n_events_], q_population[n_events_], lambda_tilde_population[n_events_], effective_distance_population[n_events_], z_population[n_events_]]))
        waveform_ = Waveform(pars_, freq)
        waveform_.calculate_errors(psd)
        waveform_.make_log_q_log_lambda_tilde_samples()
        f.create_dataset('log_q_samples'+str(n_events_), data=waveform_.log_q_samples)
        f.create_dataset('log_lambda_tilde_samples'+str(n_events_), data=waveform_.log_lambda_tilde_samples)
        waveform_.make_z_m_chirp_samples(m_chirp_grid, log_q_array, log_lambda_tilde_array)
        f.create_dataset('z_samples'+str(n_events_), data=waveform_.z_samples)
        f.create_dataset('m_chirp_samples'+str(n_events_), data=waveform_.m_chirp_samples)
        redshift_measured[n_events_] = np.median(waveform_.z_samples)
        log_effective_distance_error[n_events_] = waveform_.log_effective_distance_error
        f.create_dataset('log_effective_distance_error'+str(n_events_), data=waveform_.log_effective_distance_error)
    f.create_dataset('chirp mass population', data=m_chirp_population)
    f.create_dataset('q population', data=q_population)
    f.create_dataset('lambda_tilde population', data=lambda_tilde_population)
    f.create_dataset('redshift population', data=z_population)
    f.create_dataset('effective distance population', data=effective_distance_population)
    f.create_dataset('snr population', data=snr_population)



