import gp_lens
import numpy as np
from astropy.table import Table

redshift_list = ('05', '10', '15', '20', '25')

peaks = gp_lens.LensingPSorPeaks(-0.02, 1000, 'KN', redshifts=redshift_list, observable_name='Peaks', bin_center_row=0, smoothing='2.00')
x, y = peaks.get_realizations(model_index=1, verbose=True)
peaks.fid = np.mean(y,axis=0)

powerspec = gp_lens.LensingPSorPeaks(300,5000, 'KN', redshifts=redshift_list, observable_name='PS', bin_center_row=0, binscale='log')
x, y = powerspec.get_realizations(model_index=1, verbose=True)
powerspec.fid = np.mean(y,axis=0)

class LensingPS_AND_Peaks(gp_lens.LensingPSorPeaks):
    def __init__(self, sky_coverage=1e4):
        """Initialize the parameters table and specifications."""
        # read in list of simulation cosmo parameters
        self.table = Table.read('parameters.table', format='ascii')
        
        # dictionary for getting the appropriate ng for a redshift
        self.ng_dict = {
            '05': '08.83',
            '10': '13.25',
            '15': '11.15',
            '20': '07.36',
            '25': '04.26',
            '10_ng40': '40.00'
        }
        self.sky_coverage = sky_coverage
        
        self.params = np.array([self.table['M_nu(eV)'],
                                self.table['Omega_m'],
                                self.table['10^9*A_s']])
        
    def get_realizations(self, model_index, covariance=False, verbose=False):
        x_peaks, y_peaks = peaks.get_realizations(model_index, covariance, verbose)
        x_ps, y_ps = powerspec.get_realizations(model_index, covariance, verbose)
        
        return np.hstack([x_ps, x_peaks]), np.hstack([y_ps, y_peaks])

    
# set up some constants for prior evaluation
m_nu_min = 0.06  # minimum from oscillation experiments
m_nu_max = 2*np.max(combo.table['M_nu(eV)'])
om_m_min = np.min(combo.table['Omega_m'])
om_m_max = 2*np.max(combo.table['Omega_m'])
A_s_min = np.min(combo.table['10^9*A_s'])
A_s_max = 2*np.max(combo.table['10^9*A_s'])

# define emcee function calls, prior, likelihood,
def lnprior(theta):
    """Ensure the sampler stays near computed simulations."""
    m_nu, om_m, A_s = theta
    if (m_nu_min < m_nu < m_nu_max and
            om_m_min < om_m < om_m_max and
            A_s_min < A_s < A_s_max):
        return 0.0
    return -np.inf


def lnlike(theta):
    """Compute the log likelihood based on multivariate Gaussian."""
    return combo.likelihood(theta)


def lnprob(theta):
    """Combine the likelihood and prior."""
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

    
combo = LensingPS_AND_Peaks()
x, y = combo.get_realizations(model_index=1, verbose=True)
combo.fid = np.mean(y,axis=0)

index_list = np.arange(len(peaks.params.T))
index_list = np.delete(index_list, 0)
index_list = np.delete(index_list, 0)

modified_y = [combo.get_realizations(i) for i in index_list]
modified_X = (combo.params.T[index_list]).T

combo.fit(X=modified_X, real_list=modified_y)

test_model = 1
x, y = combo.get_realizations(model_index=1, verbose=True)
y_true = np.mean(y, axis=0)


ys, sigs = combo.GP(combo.params.T[1])
invcov = combo.compute_cov(0)
cov = np.linalg.inv(invcov)




import emcee

# set up emcee
ndim, nwalkers = 3, 120
p0 = [combo.params.T[1] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# begin sampling, use incremental chain
filename = '/tigress/zequnl/gp_chains/combo12.out'
f = open(filename, "w")
f.close()

for result in sampler.sample(p0, iterations=5000, storechain=False):
    position = result[0]
    f = open(filename, "a")
    for k in range(position.shape[0]):
        out_str = "{0:4d} {1:s}\n".format(k, ' '.join(map(str, position[k])))
        f.write(out_str)
    f.close()

    
    