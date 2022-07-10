import numpy as np
import h5py
from numba import jit
import matplotlib.pyplot as plt
import sys
import corner
from scipy.interpolate import interp2d
from scipy.optimize import fsolve

c = 299792458
GPC_TO_METER = 3.0856775814671914e25
GM_sun_by_c_squared = 1500
c_in_km_per_sec = c/1000

#########################################################################################
######DEFINE PN WAVEFORM TO DOMINANT ORDER OF EACH TERM##################################

#Masses to be entered in solar masses
#Distances to be entered in Gpc

def Psi_PP(mc_z,q,f):
    mc_z *= GM_sun_by_c_squared
    f=f/c
    return 3./(128.*(np.pi*mc_z*f)**(5./3.)) 
def Psi_PPht1(mc_z,q,f):
    mc_z *= GM_sun_by_c_squared
    f=f/c
    return (3*3715./(128.*756.))*(1+q)**(4./5)/(q**(2/5)*mc_z)*(np.pi*f)**(-1.)
def Psi_PPht2(mc_z,q,f):
    mc_z *= GM_sun_by_c_squared
    f=f/c
    return (55./(128.*3.))*q**(3./5)/(mc_z*(1+q)**(6./5))*(np.pi*f)**(-1.)
def Psi_tidal(mc_z,q,lambdat,f):
    mc_z *= GM_sun_by_c_squared
    f=f/c
    return (-3.*39./256.)*lambdat*(np.pi*f)**(5./3.)*mc_z**(5/3.)*(1+q)**4/q**2.
    
def h(mc_z,q,lambdat,deff,tc,pc,f):
    #the h(f) functin returns in proper SI units (in seconds)
    return  -np.sqrt(5*np.pi/24)*(mc_z*GM_sun_by_c_squared) **(5./6.)*(np.pi*f/c)**(-7./6.)*np.exp(-1.0j*(Psi_PP(mc_z,q,f)+Psi_PPht1(mc_z,q,f)+Psi_PPht2(mc_z,q,f)+Psi_tidal(mc_z,q,lambdat,f)))/(c * GPC_TO_METER *deff)
###################################################################################################################


#################USEFUL FUNCTIONS############################################################################################
def m_from_m_chirp_and_q(m_chirp,q):
    return m_chirp*(q/(1+q)**2)**(-3/5.)

def m1_m2_from_m_chirp_and_q(m_chirp,q):
    m2 = m_from_m_chirp_and_q(m_chirp,q)/(1+q)
    m1 = m2*q
    return m1,m2

def q_from_m1_m2(m1,m2):
    q = m1/m2
    # mask = np.where(q>1)
    # q[mask] = 1./q[mask]
    return q

def m_chirp_from_m1_m2(m1,m2):
    return ((m1*m2)**0.6)/((m1+m2)**0.2)

def lambda_from_m(mass):
    m, l = np.loadtxt('Mass_Vs_TidalDeformability_SLY.txt',dtype=float,unpack=True)  
    return np.interp(mass,m,l)

def eta_from_m1_m2(m1, m2):
    return m1*m2/(m1+m2)**2

def lambda_tilde_from_m1_m2(m1,m2):
    #from arXiv 1402.5156
    eta = eta_from_m1_m2(m1, m2)
    L1 = lambda_from_m(m1)
    L2 = lambda_from_m(m2)
    return (8./13) * ((1+7*eta-31*eta*eta)*(L1+L2) + ((m1-m2)/(m1+m2))*(1+9*eta-11*eta*eta)*(L1-L2))

def lambda_tilde_from_m_chirp_q(m_chirp,q):
    return lambda_tilde_from_m1_m2(m1_m2_from_m_chirp_and_q(m_chirp,q)[0],m1_m2_from_m_chirp_and_q(m_chirp,q)[1])

def log_lambda_tilde_grid_from_m_chirp_q(m_chirp, q):
    m_chirp_q = np.meshgrid(m_chirp, q)
    m = m1_m2_from_m_chirp_and_q(m_chirp_q[0], m_chirp_q[1])
    log_lambda_tilde_grid = np.log(lambda_tilde_from_m1_m2(m[0],m[1]))
    return log_lambda_tilde_grid

def m_chirp_grid_from_log_q_log_lambda_tilde(log_q_array,log_lambda_tilde_array):
    log_q_array_log_lambda_tilde_array_mesh = np.meshgrid(log_q_array,log_lambda_tilde_array)
    log_q_array_mesh_flattened = log_q_array_log_lambda_tilde_array_mesh[0].flatten()
    log_lambda_tilde_array_mesh_flattened = log_q_array_log_lambda_tilde_array_mesh[1].flatten()
    mc = fsolve(lambda x: np.log(lambda_tilde_from_m_chirp_q(x, np.exp(log_q_array_mesh_flattened)))-(log_lambda_tilde_array_mesh_flattened),np.ones(len(log_q_array_mesh_flattened))*1.2)
    return mc.reshape(len(log_q_array),len(log_lambda_tilde_array))


def f_isco(m_chirp_z,q):
    return c/(6.**0.5*6.*np.pi*m_from_m_chirp_and_q(m_chirp_z,q)*1500)

###################################################################################################################


##########################Some basic cosmology functions####################################################################
######################################################################################
####### CALCULATE AN ARRAY OF LUMINOSITY DISTANCE GIVEN AN ARRAY OF REDSHIFT #########
#############THE REDSHIFT ARRAY SHOULD BE UNIFORMLY SPACED WITH SMALL dz############## 
######################################################################################
@jit(nopython=True)
def E(z, Om0, w_0, w_a):
    Ol0=1-Om0
    return ((Om0*(1+z)**3.)+Ol0*(1+z)**(3.0*(1+w_0+w_a*z/(1+z))))**0.5

@jit(nopython=True)
def integrate(z_l, z_h, Om0, w_0, w_a):
    z_lh=np.linspace(z_l,z_h,5)
    return np.trapz(1.0/E(z_lh,Om0,w_0,w_a),dx=z_lh[1]-z_lh[0])

@jit(nopython=True)
def luminosity_distance(z_array, H0, Om0 = 0.31, w_0 = -1, w_a = 0):
    lz=len(z_array)
    dLt=np.zeros(lz)
    dLarr=np.zeros(lz)
    t=integrate(0,z_array[0],Om0,w_0,w_a)
    dLt[0]=t
    dLarr[0]=(c*(1.0+z_array[0])*t)/H0
    for i in range(1,lz):
        dLt[i]=integrate(z_array[i-1],z_array[i],Om0,w_0,w_a)
        dLarr[i]=np.sum(dLt)*c_in_km_per_sec*(1+z_array[i])/H0
    return dLarr/1000 #in Gpc
######################################################################################



class Waveform:
    def __init__(self, pars, freq):
        m_chirp_z = pars['m_chirp_z']
        q = pars['q']
        lambda_tilde = pars['lambda_tilde']
        effective_distance = pars['effective_distance'] 
        z = pars['z']
        mask = freq < f_isco(m_chirp_z,q)
        self.h = h(m_chirp_z,q,lambda_tilde,effective_distance,0,0,freq[mask])
        self.f = freq[mask]
        self.m_chirp_z = m_chirp_z
        self.q = q
        self.z = z
        self.lambda_tilde = lambda_tilde
        self.effective_distance = effective_distance
        self.m_z = m_from_m_chirp_and_q(m_chirp_z,q)
        self.mask = mask
    
    def plot_waveform(self):
        plt.plot(self.f,np.abs(self.h))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (s)')
        plt.show()

    def snr(self, psd):
        freq = self.f
        psd = psd[self.mask]
        return 2*np.sqrt(np.trapz(np.real(self.h*np.conj(self.h))/psd,freq))

    def calculate_errors(self, psd):
        Mc_z = self.m_chirp_z
        q = self.q
        Lambdat = self.lambda_tilde
        f = self.f
        psd = psd[self.mask]
        dhdlogq = -1.0j * self.h * (Psi_PPht1(Mc_z,q,f)*(4*q/(5*(1+q))-2./5) + Psi_PPht2(Mc_z,q,f)*(3/5-(6*q)/(5*(1+q)))+Psi_tidal(Mc_z,q,Lambdat,f)*(4*q/(1+q)-2.) )
        dhdlogDeff = -1 * self.h
        dhdlogLam = -self.h * 1.0j * Psi_tidal(Mc_z,q,Lambdat,f)
        dhdlogq2_int = np.trapz(4*np.real(dhdlogq*np.conj(dhdlogq))/psd, f)
        dhdlogDeff2_int = np.trapz(4*np.real(dhdlogDeff*np.conj(dhdlogDeff))/psd, f)
        dhdlogLam2_int = np.trapz(4*np.real(dhdlogLam*np.conj(dhdlogLam))/psd, f)
        dhdlogqLam_int = np.trapz(4*np.real(dhdlogq*np.conj(dhdlogLam))/psd, f)
        self.fisher = np.array([[dhdlogq2_int, dhdlogqLam_int],
                                [dhdlogqLam_int, dhdlogLam2_int]])
        self.log_effective_distance_error = np.sqrt(1/dhdlogDeff2_int)
        self.log_q_log_lambda_tilde_cov_matrix = np.array([[dhdlogLam2_int, -dhdlogqLam_int],
                                                [-dhdlogqLam_int, dhdlogq2_int]])/(dhdlogq2_int*dhdlogLam2_int - dhdlogqLam_int**2)
        #self.log_q_log_lambda_tilde_cov_matrix_numerical = np.linalg.inv(self.fisher)
        return None
    
    def make_log_q_log_lambda_tilde_samples(self, n_samples = 4000):
        mean = np.log(np.array([self.q, self.lambda_tilde]))
        cov = self.log_q_log_lambda_tilde_cov_matrix
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        self.log_q_samples = samples[:,0]
        self.log_lambda_tilde_samples = samples[:,1]
        return None

    # def make_log_q_log_lambda_tilde_samples_with_noise(self, n_samples = 4000):
    #     mean = np.log(np.array([self.q, self.lambda_tilde]))
    #     cov = self.log_q_log_lambda_tilde_cov_matrix
    #     mean1 = np.random.multivariate_normal(mean, cov)
    #     samples = np.random.multivariate_normal(mean1, cov, n_samples)
    #     self.log_q_samples = samples[:,0]
    #     self.log_lambda_tilde_samples = samples[:,1]
    #     return None

    def plot_log_q_log_lambda_tilde_log_effective_distance_samples(self):
        self.log_effective_distance_samples = np.random.normal(np.log(self.effective_distance), self.log_effective_distance_error, len(self.log_q_samples))
        data=np.array([self.log_q_samples, self.log_lambda_tilde_samples, self.log_effective_distance_samples]).T 
        labels=np.array(['log q','log lambda_tilde', 'log effective distance'])             
        tr=corner.corner(data,labels=labels,smooth=1.2,quantiles=[0.1,0.5,0.9],show_titles=True,title_kwargs={"fontsize": 12},use_math_text=True)
        ndim = 3
        axes = np.array(tr.axes).reshape((ndim, ndim))
        true_value = np.log(np.array([self.q, self.lambda_tilde, self.effective_distance]))
                    
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(true_value[i], color="r")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(true_value[xi], color="r")
                ax.axhline(true_value[yi],color='r')
                ax.plot(true_value[xi], true_value[yi], "sr")
        plt.show()

        

    
    def make_z_m_chirp_samples(self, m_chirp_grid, log_q_array, log_lambda_tilde_array):
        m_chirp_grid_interp = interp2d(log_q_array,log_lambda_tilde_array,m_chirp_grid)
        log_q = np.array(self.log_q_samples[:])
        mask = np.where(log_q>0)
        log_q[mask] = - log_q[mask]
        log_q = self.log_q_samples
        log_lambda_tilde_samples_argsort = np.argsort(self.log_lambda_tilde_samples)
        log_q = log_q[log_lambda_tilde_samples_argsort]
        self.log_lambda_tilde_samples = self.log_lambda_tilde_samples[log_lambda_tilde_samples_argsort]
        log_q_argsort = np.argsort(log_q)
        self.m_chirp_samples = np.diag(m_chirp_grid_interp(log_q, self.log_lambda_tilde_samples)[log_q_argsort])
        self.z_samples = (self.m_chirp_z/self.m_chirp_samples)-1
        self.log_q_samples = self.log_q_samples[log_lambda_tilde_samples_argsort][log_q_argsort]
        self.log_lambda_tilde_samples = self.log_lambda_tilde_samples[log_q_argsort]
        return None

    

    def plot_m_chirp_log_q_z(self):
        #makes a corner plot of the q and lambda_tilde samples
        data=np.array([self.m_chirp_samples, self.log_q_samples, self.z_samples]).T 
        labels=np.array(['m_chirp', 'log q', 'z'])             
        tr=corner.corner(data,labels=labels,smooth=1.2,quantiles=[0.1,0.5,0.9],show_titles=True,title_kwargs={"fontsize": 12},use_math_text=True)
        ndim = 3
        axes = np.array(tr.axes).reshape((3, 3))
        true_value = (np.array([self.m_chirp_z/(1+self.z), np.log(self.q), self.z]))
                      
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(true_value[i], color="r")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(true_value[xi], color="r")
                ax.axhline(true_value[yi],color='r')
                ax.plot(true_value[xi], true_value[yi], "sr")
        plt.show()

if __name__ == "__main__":
    m1 = 1.2
    m2 = 1.4
    z = 3
    theta = 1

    z_array = np.linspace(0.0001,10.,100)
    luminosity_distance_array = luminosity_distance(z_array, 70.5, 0.3)
    cosmo_true_z_luminosity_distance_list = np.array([z_array, luminosity_distance_array])

    effective_distance = np.interp(z, cosmo_true_z_luminosity_distance_list[0], cosmo_true_z_luminosity_distance_list[1])/theta
    q = q_from_m1_m2(m1,m2)
    lambda_tilde = lambda_tilde_from_m1_m2(m1,m2)
    m_chirp = m_chirp_from_m1_m2(m1,m2)
    m_chirp_z = m_chirp * (1+z)

    freq, psd = np.loadtxt('CE_psd.txt', unpack=True)
        
    keys = ['m_chirp_z', 'q', 'lambda_tilde', 'effective_distance', 'z']
    values = [m_chirp_z, q, lambda_tilde, effective_distance, z]
    pars = dict(zip(keys, values))
    waveform = Waveform(pars, freq)
    waveform.plot_waveform()
    waveform.calculate_errors(psd)
    waveform.make_log_q_log_lambda_tilde_samples()
    waveform.plot_log_q_log_lambda_tilde_log_effective_distance_samples()





    