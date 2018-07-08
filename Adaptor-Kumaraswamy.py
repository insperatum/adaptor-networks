import torch
from torch import nn
from torch import optim

from torch.nn import Parameter
from scipy.special import gammaln
import numpy as np

def sample_kumaraswamy(batch_size, a, b):
    u = torch.rand(batch_size, len(a))
    x = (1-u**(1/b))**(1/a)
    return 1e-4 + (1-2e-4)*x

def mean_kumaraswamy(a, b): #Not differentiable
    log_mean = b.log() + gammaln(1+1/a) + gammaln(b) - gammaln(1 + 1/a + b)
    mean = log_mean.exp()
    return mean

def score_kumaraswamy(a, b, x):
    return a.log() + b.log() + (a-1)*x.log() + (b-1)*(1-x**a).log()

def get_log_pi(phi):
    """
    :param phi: batch_size * T
    """
    batch_size = phi.size(0)
    log_phi = phi.log()
    log_pi = log_phi + torch.cumsum(
            torch.cat([torch.zeros(batch_size, 1), (1-phi[:, :-1]).log()], dim=1),
            dim=1)
    return log_pi

def logsumexp(t):
    m, _ = torch.max(t, dim=1)
    return (t-m[:, None]).exp().sum(dim=1)+m

class Base(nn.Module):
    def forward(self, xs):
        return torch.rand(len(xs))-10

class Adaptor(nn.Module):
    """
    Reparametrised q(phi)
    Single valued q(mu_i)
    """
    def __init__(self, T, base):
        super().__init__()
        self.base = base
        self.values = [None] * T
        self._a = Parameter(torch.zeros(T))
        self._b = Parameter(torch.zeros(T))
        self.softplus = nn.Softplus()
        self.t = Parameter(torch.ones(1))

    @property
    def a(self): return self.softplus(self._a)+1

    @property
    def b(self): return self.softplus(self._b)+1
  
    def log_E_pi(self):
        E_phi = mean_kumaraswamy(self.a.detach(), self.b.detach())
        log_E_pi = get_log_pi(E_phi[None, :])[0]
        return log_E_pi

    def forward(self, xs=None, nData=None):
        """
        :param x: list(batch) of strings
        """
        batch_size = len(xs)

        # Ensure all xs are in the mixture distribution
        log_E_pi = self.log_E_pi()

        missing = list(set(xs) - set(self.values))
        replace_idxs = np.argpartition(log_E_pi, len(missing))[:len(missing)]
        for idx, value in zip(replace_idxs, missing):
            self.values[idx] = value

        # Sample from SBP
        phi = sample_kumaraswamy(batch_size, a=self.a, b=self.b)
        log_pi = get_log_pi(torch.cat([phi, self.t.new_ones(batch_size, 1)], dim=1)) # Final component is 'remaining'

        # Calculate probabilities from mixture distribution and base distribution
        idxs = [self.values.index(x) for x in xs]
        component_probs = log_pi.gather(1, log_pi.new(idxs).long().view(-1,1))[:, 0]
        base_probs = log_pi[:, -1] + self.base(xs)

        #Total
        conditional = logsumexp(torch.cat([component_probs[:, None], base_probs[:, None]], dim=1))
        
        # Variational bound with uniform([0,1]) prior on phi
        score = (conditional - score_kumaraswamy(self.a, self.b, phi).sum(dim=1)/nData).mean()
        score += (base(self.values).sum()/nData).mean()

        print(phi.min().item(), log_pi.min().item(), component_probs.mean().item(), base_probs.mean().item(), conditional.mean().item(), score.mean().item())
        if torch.isnan(score):
            print("a", self.a)
            print("b", self.b)
            print("phi", phi)
            print("log pi", log_pi)
            print("component_probs", component_probs)
            print("base_probs", base_probs)
            raise Exception()
        return score


if __name__ == "__main__":
    base = Base()
    adaptor = Adaptor(5, base)

    optimiser = optim.Adam(adaptor.parameters(), lr=1e-2)
    for batch in range(1000):
        for iteration in range(100):
            optimiser.zero_grad()
            xs = ["foo", "bar", "hello", "hello", "world"]
            score = adaptor(xs, nData=500)
            print("score", score)
            (-score).backward()
            optimiser.step()
        print("Batch", batch, "Score", score.item())
        print(", ".join("%s:%3.3f" % (v, p.item()) for v,p in zip(adaptor.values, adaptor.log_E_pi().exp())))
