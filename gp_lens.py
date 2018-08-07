"""Compute likelihoods using Gaussian Processes."""
import numpy as np
from scipy.stats import sem as sem
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from astropy.table import Table

import config

class Observable:
    """Base class for fitting GP and computing likelihoods."""

    def __init__(self):
        """Initialize settings for the observable like model parameters."""
        pass

    def get_realizations(self, model_index):
        """Return a list of realizations for a particular model."""
        raise NotImplementedError("You need to implement the realizations function!")

    def fit(self, X=None, real_list=None):
        """Fit a Gaussian Process using the realizations function provided."""
        gp_list = []
        datascale = []  # normalize each bin so mean is 1

        if X is None:
            X = self.params
        if real_list is None:
            real_list = [self.get_realizations(i) for i in range(len(self.params.T))]
        y_real_list = [y_ for x_, y_ in real_list]
        x_list = real_list[0][0]

        X = np.atleast_2d(X).T
        for test_ell_bin in range(len(x_list)):
            y = np.array([np.mean(ps_temp, axis=0)[test_ell_bin]
                          for ps_temp in y_real_list])
            dy = np.array([sem(ps_temp, axis=0)[test_ell_bin]
                           for ps_temp in y_real_list])
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

        def GP(x):
            y_pred_list = []
            sigma_list = []
            for gp in gp_list:
                y_pred, sigma = gp.predict(np.atleast_2d(x), return_std=True)
                y_pred_list.append(y_pred[0])
                sigma_list.append(sigma[0])

            return np.array(y_pred_list * datascale),\
                np.array(sigma_list * datascale)

        self.GP = GP
        self.gp_list = gp_list

    def compute_cov(self, model_index):
        """Compute covariance using the realizations function provided."""
        pass

    def likelihood(self, theta):
        """Compute the log likelihood based on multivariate Gaussian."""
        model, sig = self.GP(theta)
        dmu = self.fid - model
        return -0.5 * (np.dot(dmu, np.dot(self.invcov, dmu)))

    def __call__(self, parameter_input):
        return self.likelihood(parameter_input)

    
# inherit from the main GP class
class LensingPSorPeaks(Observable):
    """
    Fit GP and compute likelihoods from simulation data.
    """

    def __init__(self, kappa_min, kappa_max, noisy, redshifts, observable_name,
                 smoothing='1.00', binning='050', binscale='', bin_center_row=0,
                 sky_coverage=1e4):
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
        
        self.observable_name = observable_name
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.noisy = noisy
        self.smoothing = smoothing
        self.binning = binning
        self.binscale = binscale
        self.bin_center_row = bin_center_row
        self.redshifts = redshifts
        self.sky_coverage = sky_coverage
        
        self.params = np.array([self.table['M_nu(eV)'],
                                self.table['Omega_m'],
                                self.table['10^9*A_s']])
        
    def compute_cov(self, model_index, verbose=False):
        """Compute covariance using the realizations function provided."""
        bin_centers, realizations_stacked = self.get_realizations(model_index=0, covariance=True, verbose=verbose)
#         realizations_stacked = np.hstack(real_arr)
        print(realizations_stacked.shape)
        # now compute covariance
        cov = np.cov(realizations_stacked.T)

        nrealizations, nbins = realizations_stacked.shape
        bin_correction = (nrealizations - nbins - 2) / (nrealizations - 1)
        sky_correction = 12.25/self.sky_coverage

        if verbose:
            print('nr', nrealizations, 'nb', nbins,
                  'bin', bin_correction, 'sky', sky_correction)

        # this 12.25/2e4 is from the LSST area divided by box, from Jia's email
        invcov = bin_correction * np.linalg.inv(cov * sky_correction)
        self.invcov = invcov
        return invcov
    
    def get_realizations(self, model_index, covariance=False, verbose=False):
        xlist = []
        ylist = []
        
        for redshift in self.redshifts:
            x, y = self.get_hom_realizations(model_index=model_index, redshift=redshift, covariance=covariance,
                                             observable_name=self.observable_name, 
                                             bin_min=self.kappa_min, bin_max=self.kappa_max, verbose=verbose)
            xlist.append(x)
            ylist.append(y)
        
        return np.hstack(xlist), np.hstack(ylist) 
    
    def get_hom_realizations(self, model_index, redshift, observable_name, 
                             bin_min, bin_max, covariance=False, verbose=False):
        """get realizations from a single redshift and observable"""
        row = self.table[model_index]
        
        # load in a set of realizations for a specific model
        if row['Model'] == '1a(fiducial)':
            if covariance:
                modifier = '/box5'  # use box1 for means, use box5 for covariance
            else:
                modifier = '/box1'
        else:
            modifier = ''

        if redshift == '10_ng40':
            fname = '%s_%s_s%s_z%.2f_ng%s_b%s%s' \
                % (observable_name, self.noisy, self.smoothing, 1.0,
                   self.ng_dict[redshift], self.binning, self.binscale)
            # meandir = 'data/May_stats/%s/Maps%s/%s_Mean.npy' \
            #     % (row['filename'] + modifier, redshift, fname)
            fdir = config.data_folder + '%s/Maps%s/%s.npy' \
                % (row['filename'] + modifier, redshift, fname)
        else:
            fname = '%s_%s_s%s_z%.2f_ng%s_b%s%s' \
                % (observable_name, self.noisy, self.smoothing, float(redshift)/10,
                   self.ng_dict[redshift], self.binning, self.binscale)
            # meandir = 'data/May_stats/%s/Maps%s/%s_Mean.npy' \
            #     % (row['filename'] + modifier, redshift, fname)
            fdir = config.data_folder + '%s/Maps%s/%s.npy' \
                % (row['filename'] + modifier, redshift, fname)

        if(verbose):
            print(fdir)
        obs_array_temp = np.load(fdir)
        if observable_name == 'PS':
            bin_centers = obs_array_temp[0]
            real_arr = (obs_array_temp[1:])
        elif observable_name == 'Peaks':
            bin_centers = obs_array_temp[self.bin_center_row]
            real_arr = (obs_array_temp[2:])

        filter_for_bins = np.logical_and(bin_min < bin_centers,
                                         bin_centers < bin_max)
        bin_centers = bin_centers[filter_for_bins]
        real_arr = (real_arr.T[filter_for_bins]).T
        
        return bin_centers, real_arr