"""Generate parameter constraints with emcee and Gaussian Processes."""

import emcee
import stats_utils
import numpy as np
import argparse

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import sem as sem


# load in the command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-o', '--output', dest='filename', action='store',
                    default='chain.dat',
                    help='output chain path. default: chain.dat, current dir')
parser.add_argument('-d', '--data', dest='data', action='store',
                    default=None, required=True,
                    help='PS or Peaks')
parser.add_argument('--datascale', dest='datascale', action='store',
                    default=1e-9, required=True, type=float,
                    help='scale of data, for normalization')

parser.add_argument('-binmin', dest='binmin', action='store',
                    default=None, type=float, required=True,
                    help='minimum bin center')
parser.add_argument('-binmax', dest='binmax', action='store',
                    default=None, type=float, required=True,
                    help='maximum bin center')
parser.add_argument('-mn', '--mean_noise', dest='mean_noise', action='store',
                    default='KN', required=True,
                    help='Means: Noisy or not noisy, K or KN.')
parser.add_argument('-cn', '--cov_noise', dest='cov_noise', action='store',
                    default='KN', required=True,
                    help='Covariance: Noisy or not noisy, K or KN.')

parser.add_argument('-z', '--redshift', nargs='*', dest='redshift',
                    action='store', default=['05', '10', '15', '20', '25'],
                    help='redshifts, i.e. 05 10 15 20 25')
parser.add_argument('-s', '--smooth', dest='smoothing_scale', action='store',
                    default='1.00',
                    help='Smoothing Scale')
parser.add_argument('--binscale', dest='binscale', action='store',
                    default='log',
                    help='log or lin, bin scale for filename')

parser.add_argument('-N', dest='mcmc_N', action='store',
                    default=1000, type=int,
                    help='number of MCMC steps')
parser.add_argument('-w', '--walkers', dest='walkers', action='store',
                    default=120, type=int,
                    help='number of MCMC walkers')

args = parser.parse_args()
print(f'filename: {args.filename}')
print(f'data source: {args.data}, scale: {args.datascale}')
print(f'redshifts: {args.redshift}')
print(f'bin min, binmax: {args.binmin}, {args.binmax}')
print(f'mean noise: {args.mean_noise}, cov noise: {args.cov_noise}')
print(f'smoothing scale: {args.smoothing_scale}, binscale: {args.binscale}')


def build_GP(params, ell, ps_mean):
    """
    Build a function that interpolates using GP.

    Returns
    -------
        function which takes in cosmological parameters, outputs interpolation

    """
    gp_list = []
    for test_ell_bin in range(len(ell)):
        X = np.array([stats_utils.table['M_nu(eV)'],
                      stats_utils.table['Omega_m'],
                      stats_utils.table['10^9*A_s']])
        X = np.atleast_2d(X).T
        y = np.array([np.mean(ps_temp, axis=0)[test_ell_bin]
                      for ps_temp in ps_mean]) / args.datascale
        dy = np.array([sem(ps_temp, axis=0)[test_ell_bin]
                       for ps_temp in ps_mean]) / args.datascale
        kernel = C(1.0, (1e-3, 1e3)) * RBF(np.ones(3), (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=(dy)**2,
                                      n_restarts_optimizer=10,
                                      normalize_y=True)
        gp.fit(X, y)
        gp_list.append(gp)

    def get_PS_(x):
        y_pred_list = []
        sigma_list = []
        for gp in gp_list:
            y_pred, sigma = gp.predict(np.atleast_2d(x), return_std=True)
            y_pred_list.append(y_pred[0])
            sigma_list.append(sigma[0])

        return np.array(y_pred_list), np.array(sigma_list)

    return get_PS_


# load up a parameter table
params = np.array([stats_utils.table['M_nu(eV)'],
                   stats_utils.table['Omega_m'],
                   stats_utils.table['10^9*A_s']])

# stitch together the realizations from different redshifts
ell = []
ps_real = []
for redshift in args.redshift:
    ell_, ps_real_ = stats_utils.get_real_list(args.data,
                                               noisy=args.mean_noise,
                                               redshift=redshift,
                                               bin_min=args.binmin,
                                               bin_max=args.binmax,
                                               smoothing=args.smoothing_scale,
                                               binscale=args.binscale)
    ell.append(ell_)
    ps_real.append(ps_real_)

ell = np.hstack(ell)
ps_real = np.dstack(ps_real)

# load the inverse covariance, and then train the Gaussian Process
invcov = (args.datascale**2) * \
          stats_utils.get_invcov(args.data,
                                 noisy=args.cov_noise,
                                 redshifts=args.redshift,
                                 bin_min=args.binmin,
                                 bin_max=args.binmax,
                                 smoothing=args.smoothing_scale,
                                 binscale=args.binscale)
fid = np.mean(ps_real[1], axis=0) / args.datascale
get_PS = build_GP(params, ell, ps_real)

# set up some constants for prior evaluation
m_nu_min = np.min(stats_utils.table['M_nu(eV)'])
m_nu_max = np.max(stats_utils.table['M_nu(eV)'])
om_m_min = np.min(stats_utils.table['Omega_m'])
om_m_max = np.max(stats_utils.table['Omega_m'])
A_s_min = np.min(stats_utils.table['10^9*A_s'])
A_s_max = np.max(stats_utils.table['10^9*A_s'])


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
    model, sig = get_PS(theta)
    dmu = fid - model
    return -0.5 * (np.dot(dmu, np.dot(invcov, dmu)))


def lnprob(theta):
    """Combine the likelihood and prior."""
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)


# set up emcee
ndim, nwalkers = 3, args.walkers
p0 = [params.T[1] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)


# begin sampling, use incremental chain
filename = args.filename
f = open(filename, "w")
f.close()

for result in sampler.sample(p0, iterations=args.mcmc_N, storechain=False):
    position = result[0]
    f = open(filename, "a")
    for k in range(position.shape[0]):
        out_str = "{0:4d} {1:s}\n".format(k, ' '.join(map(str, position[k])))
        f.write(out_str)
    f.close()
