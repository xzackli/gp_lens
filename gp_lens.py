"""Compute likelihoods using Gaussian Processes."""


class Observable:
    """Base class for fitting GP and computing likelihoods."""

    def __init__(self):
        """Initialize settings for the observable like model parameters."""
        pass

    def get_realizations(self, model_index):
        """Return a list of realizations for a particular model."""
        raise NotImplementedError("You need to implement the realizations function!")

    def fit(self):
        pass

    def compute_cov(self):
        pass

    def likelihood(parameter_input):
        pass

    def __call__(self, parameter_input):
        return self.likelihood(parameter_input)
