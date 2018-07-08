import torch
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal
from torch.nn.functional import log_softmax

class NormalLogSoftmax(Distribution):
    """
    mu, sigma have dimensionality (D-1)
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dist = Normal(mu, sigma)

    def rsample(self):
        z = self.dist.rsample()
        z = torch.cat([z, self.mu.new_zeros(self.mu[:, -1:].size())], dim=1)
        return log_softmax(z, dim=1)

    def log_prob(self, log_pi):
        z = log_pi[:, :-1] - log_pi[:, -1:]
        log_jacobian_determinant = 1 - log_pi[:, -1].exp()
        return self.dist.log_prob(z).sum(dim=1) - log_jacobian_determinant


if __name__ == "__main__":
    for sigma in [10**(4-x) for x in range(10)]:
        print("sigma", sigma)
        dist = NormalLogSoftmax(torch.randn(1,3), torch.ones(1,3)*sigma)
        log_pi = dist.rsample()
        print("log_pi", log_pi)
        score = dist.log_prob(log_pi)
        print("score", score)
        print()
