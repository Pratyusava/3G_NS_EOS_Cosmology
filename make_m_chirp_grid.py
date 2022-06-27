from basic_functions import *
l = 150
log_q_array = np.linspace(np.log(0.3),np.log(1),l)
log_lambda_tilde_array = np.linspace(4,10,l)
m_chirp_grid = m_chirp_grid_from_log_q_log_lambda_tilde(log_q_array,log_lambda_tilde_array)
np.savetxt('m_chirp_grid_using_SLY'+str(l)+'.txt', m_chirp_grid)