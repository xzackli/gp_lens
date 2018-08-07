# gp_lens

> How do you measure numbers from simulations?

Sometimes you have a set of simulations distributed over your parameter space, and a fiducial measurement with error bars (from data and/or simulations). How do you write down a likelihood that is applicable over all of parameter space?

The main issue is interpolation, and it turns out that **Gaussian Processes (GP)** are very effective for this job. The main advantage of GP is that it uses the noise information (which you can infer from your simulations) as part of the interpolation. This reduces the error from interpolation, which especially in high-dimensional spaces can really throw off your error bars.

This set of scripts are for a paper on lensing peak counts (Li, Liu, Zorrila, Coulton 2018).

## Interface

The generic workflow is

1. inherit from the `gp_lens.Observable` class
2. write a new constructor and `get_realizations()` function
3. fit the GP and compute covariance with the included functions, yielding a likelihood.

```python
import gp_lens

# inherit from the main GP class
class LensingPeaks(gp_lens.Observable):
    """
    Fit GP and compute likelihoods from simulation data.
    """

    def __init__(self):
        # set up parameter list
        ...

    def get_realizations(self, model_number):
        # load in a set of realizations for a specific model
        ...
        return summary_stat_x, summary_stat_y


peaks = LensingPeaks(fiducial=0)
peaks.fit()
peaks.compute_cov()

# test out by computing the likelihood at the fiducial
print(peaks.likelihood(peaks.params[1]))
```
