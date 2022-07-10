import sys
sys.path.append("../")
from basic_functions import *
from scipy.stats import norm,beta, uniform

NEVENTS = 1000000
SNR_CUT = 10

z_fiducial_pdf = uniform(loc = 0.0001, scale = 10)
z_fiducial = z_fiducial_pdf.rvs(NEVENTS)
effective_distance_fiducial_pdf = uniform(loc = 10, scale = 127)
effective_distance_fiducial = effective_distance_fiducial_pdf.rvs(NEVENTS)
m_chirp_fiducial_pdf = uniform(loc = 1, scale = 0.4)
m_chirp_fiducial = m_chirp_fiducial_pdf.rvs(NEVENTS)


freq, psd = np.loadtxt('../CE_psd.txt', unpack=True)
snr = np.zeros(NEVENTS)

z_detected = []
effective_distance_detected = []
m_chirp_detected = []

for i in range(NEVENTS):
    print (i) 
    keys = ['m_chirp_z', 'q', 'lambda_tilde', 'effective_distance', 'z']
    values = [m_chirp_fiducial[i]*(1+z_fiducial[i]), 0.8, 500, effective_distance_fiducial[i], z_fiducial[i]]
    pars = dict(zip(keys, values))
    waveform = Waveform(pars, freq)
    snr[i] = waveform.snr(psd)
    if snr[i]>SNR_CUT:
        z_detected.append(z_fiducial[i])
        effective_distance_detected.append(effective_distance_fiducial[i])
        m_chirp_detected.append(m_chirp_fiducial[i])

f = h5py.File('Samples.h5', 'w')
f.create_dataset('z_fiducial', data = z_fiducial)
f.create_dataset('z_detected', data = z_detected)
f.create_dataset('effective_distance_fiducial', data = effective_distance_fiducial)
f.create_dataset('effective_distance_detected', data = effective_distance_detected)
f.create_dataset('m_chirp_fiducial', data = m_chirp_fiducial)
f.create_dataset('m_chirp_detected', data = m_chirp_detected)

f.close()
   
