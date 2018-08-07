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
# parser.add_argument('--datascale', dest='datascale', action='store',
#                     default=1e-9, required=True, type=float,
#                     help='scale of data, for normalization')

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
                    default='',
                    help='empty str (default), log or lin')
parser.add_argument('--bin_center_row', type=int, dest='bin_center_row',
                    action='store', default=0,
                    help='0 or 1, 0 for PS, 0 for kappa, 1 for S/N for peaks')

parser.add_argument('-N', dest='mcmc_N', action='store',
                    default=5000, type=int,
                    help='number of MCMC steps')
parser.add_argument('-w', '--walkers', dest='walkers', action='store',
                    default=120, type=int,
                    help='number of MCMC walkers')

args = parser.parse_args()
print(f'filename: {args.filename}')
print(f'data source: {args.data}')
print(f'redshifts: {args.redshift}')
print(f'bin min, binmax: {args.binmin}, {args.binmax}')
print(f'mean noise: {args.mean_noise}, cov noise: {args.cov_noise}')
print(f'smoothing scale: {args.smoothing_scale}, binscale: {args.binscale}')
print(f'bin center row: {args.bin_center_row}')


def build_GP(params, ell, ps_mean):
    """
    Build a function that interpolates using GP.

    Returns
    -------
        function which takes in cosmological parameters, outputs interpolation

    """
    gp_list = []
    datascale = []  # normalize each bin so mean is 1
    for test_ell_bin in range(len(ell)):
        X = params
        X = np.atleast_2d(X).T
        y = np.array([np.mean(ps_temp, axis=0)[test_ell_bin]
                      for ps_temp in ps_mean])
        dy = np.array([sem(ps_temp, axis=0)[test_ell_bin]
                       for ps_temp in ps_mean])
        datascale.append(np.mean(y))
        y /= datascale[-1]
        dy /= datascale[-1]

        kernel = C(5.0, (1e-4, 1e4)) * RBF([3, 0.3, 5], (1e-4, 1e4))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=(dy)**2,
                                      n_restarts_optimizer=50,
                                      normalize_y=True)
        gp.fit(X, y)
        gp_list.append(gp)

    datascale = np.array(datascale)

    def get_PS_(x):
        y_pred_list = []
        sigma_list = []
        for gp in gp_list:
            y_pred, sigma = gp.predict(np.atleast_2d(x), return_std=True)
            y_pred_list.append(y_pred[0])
            sigma_list.append(sigma[0])

        return np.array(y_pred_list * datascale),\
            np.array(sigma_list * datascale)

    return get_PS_


# load up a parameter table
params = np.array([stats_utils.table['M_nu(eV)'],
                   stats_utils.table['Omega_m'],
                   stats_utils.table['10^9*A_s']])

# stitch together the realizations from different redshifts
ell = []
ps_real = []
for redshift in args.redshift:
    ell_, ps_real_ = stats_utils.get_real_list(
        args.data,
        noisy=args.mean_noise,
        redshift=redshift,
        bin_min=args.binmin,
        bin_max=args.binmax,
        smoothing=args.smoothing_scale,
        binscale=args.binscale,
        bin_center_row=args.bin_center_row)
    del ps_real_[0]  # DELETE FID A
    ell.append(ell_)
    ps_real.append(ps_real_)

ell = np.hstack(ell)
ps_real = np.dstack(ps_real)

#  DELETE FID A
params = np.delete(params, 0, axis=1)

# load the inverse covariance, and then train the Gaussian Process
invcov = stats_utils.get_invcov(args.data,
                                noisy=args.cov_noise,
                                redshifts=args.redshift,
                                bin_min=args.binmin,
                                bin_max=args.binmax,
                                smoothing=args.smoothing_scale,
                                binscale=args.binscale,
                                bin_center_row=args.bin_center_row)
fid = np.mean(ps_real[0], axis=0)  # FID1 is now INDEX 0
get_PS = build_GP(params, ell, ps_real)

# set up some constants for prior evaluation
m_nu_min = 0.06  # minimum from oscillation experiments
m_nu_max = 2*np.max(stats_utils.table['M_nu(eV)'])
om_m_min = np.min(stats_utils.table['Omega_m'])
om_m_max = 2*np.max(stats_utils.table['Omega_m'])
A_s_min = np.min(stats_utils.table['10^9*A_s'])
A_s_max = 2*np.max(stats_utils.table['10^9*A_s'])


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
p0 = [params.T[0] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
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
