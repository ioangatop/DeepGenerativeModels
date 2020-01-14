import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from src.utils import args


#####################
###     HELPERS   ###
#####################
def reparameterize(z_mean, z_log_var):
    epsilon = torch.randn_like(z_mean)
    return z_mean + torch.exp(0.5*z_log_var)*epsilon


def logsumexp(x, dim=None):
    """
    Args:
        x: A pytorch tensor (any dimension will do)
        dim: int or None, over which to perform the summation. `None`, the
             default, performs over all axes.
    Returns: The result of the log(sum(exp(...))) operation.
    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + torch.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))


def mse_loss(x, x_mean, reduction='none'):
    loss = F.mse_loss(x, x_mean, reduction='none')
    return - loss.view(loss.shape[0], -1).sum(1)


###################################
###     NORMAL DISTRIBUTIONS    ###
###################################

def log_normal_diag(x, mean, log_var, reduction='sum', dim=None):
	log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
	if dim is None:
		return getattr(torch, reduction)(log_normal)
	else:
		return getattr(torch, reduction)(log_normal, dim)


def log_normal_std(x, dim=None):
    if dim is None:
        return torch.sum(-0.5*torch.pow(x,2))
    else:
        return torch.sum(-0.5*torch.pow(x,2), dim)


###################################
###     Discretized logistic    ###
###################################

def discretized_logistic_loss(x, x_logit, reduction='none'):
    """ Discretized logistic loss; or cat for categorical
        Computes the cross entropy loss function while 
        summing over batch dimension.
    """
    num_classes = 256
    target = (x * (num_classes - 1)).long()	 # make integer class labels
    x_logit = x_logit.view(x_logit.shape[0], num_classes, -1, x_logit.shape[-2], x_logit.shape[-1])
    loss = F.cross_entropy(x_logit, target, reduction=reduction)
    return -loss.view(loss.size(0), -1).sum(1)


def sample_from_discretized_logistic_loss(logits, nc, random_sample=False):
    """ 
        type: 'max' or 'random'
    """
    num_classes = 256
    logits = logits.view(logits.shape[0], num_classes, -1, logits.shape[-2], logits.shape[-1])

    if random_sample:
        tmp = logits.max(dim=1)[1]  # TODO put categorical distribution
    else:
        tmp = logits.max(dim=1)[1]

    x_sample = tmp.float() / (num_classes - 1.)
    return x_sample


########################################
###   Mix of Discretized logistic   ####
########################################

def discretized_mix_logistic_loss(x, output, nc=3, nmix=10):
    """
    Discretized mix of logistic distributions loss for color images.
    HACK to work for grey images also, no changes for color images.

	Note that it is assumed that input is scaled to [-1, 1]
    """
    # batch_size_size = x.shape[0]
    nsampels   = args.z_dim
    bin_size   = 1. / 255.
    lower      = 1. / 255. - 1.0
    upper      = 1.0 - 1. / 255.
    eps        = 1e-12

    ################################################################
    logit_probs = output[:, :nmix]
    batch_size, nmix, H, W = logit_probs.size()

    # [BATCH, nmix, nc, H, W]
    means     = output[:, nmix:(nc + 1) * nmix].view(batch_size, nmix, nc, H, W)
    logscales = output[:, (nc + 1) * nmix:(nc * 2 + 1) * nmix].view(batch_size, nmix, nc, H, W)
    coeffs    = output[:, (nc * 2 + 1) * nmix:(nc * 2 + 4) * nmix].view(batch_size, nmix, nc, H, W)

    logscales   = logscales.clamp(min=-7.)
    logit_probs = F.log_softmax(logit_probs, dim=1)
    coeffs      = coeffs.tanh()

    ################################################################
    x = x.unsqueeze(1)

    means       = means.view(batch_size, *means.size()[1:])
    logscales   = logscales.view(batch_size, *logscales.size()[1:])
    coeffs      = coeffs.view(batch_size, *coeffs.size()[1:])
    logit_probs = logit_probs.view(batch_size, *logit_probs.size()[1:])

    ################################################################

    if nc==3:
        mean0 = means[:, :, 0]
        mean1 = means[:, :, 1] + coeffs[:, :, 0] * x[:, :, 0]
        mean2 = means[:, :, 2] + coeffs[:, :, 1] * x[:, :, 0] + coeffs[:, :, 2] * x[:, :, 1]
        means = torch.stack([mean0, mean1, mean2], dim=2)
    elif nc==1:
        means = means[:, :, 0].unsqueeze(2) * coeffs[:, :, 0].unsqueeze(2)

    centered_x = x - means
    inv_stdv = torch.exp(-logscales)

    # [batch_size, nmix, nc, H, W]
    min_in = inv_stdv * (centered_x - bin_size)
    plus_in = inv_stdv * (centered_x + bin_size)
    x_in = inv_stdv * centered_x

    # [batch_size, nsamples, nmix, nc, H, W]
    cdf_min = torch.sigmoid(min_in)
    cdf_plus = torch.sigmoid(plus_in)

    # lower < x < upper
    cdf_delta = cdf_plus - cdf_min
    log_cdf_mid = torch.log(cdf_delta.clamp(min=eps))
    log_cdf_approx = x_in - logscales - 2. * F.softplus(x_in) + np.log(2 * bin_size)

    # x < lower
    log_cdf_low = plus_in - F.softplus(plus_in)

    # x > upper
    log_cdf_up = -F.softplus(min_in)

    mask_delta = cdf_delta.gt(1e-5).float()
    log_cdf = log_cdf_mid * mask_delta + log_cdf_approx * (1.0 - mask_delta)
    mask_lower = x.ge(lower).float()
    mask_upper = x.le(upper).float()
    log_cdf = log_cdf_low * (1.0 - mask_lower) + log_cdf * mask_lower
    log_cdf = log_cdf_up * (1.0 - mask_upper) + log_cdf * mask_upper

    loss = logsumexp(log_cdf.sum(dim=2) + logit_probs, dim=1)
    return loss.view(loss.shape[0], -1).sum(1)


def sample_from_discretized_mix_logistic(x_mean, nc=3, nmix=10, random_sample=True):
    """
    Args:
        means: [batch_size, nmix, nc, H, W]
        logscales: [batch_size, nmix, nc, H, W]
        coeffs: [batch_size, nmix, nc, H, W]
        logit_probs:, [batch_size, nmix, H, W]
        random_sample: boolean
    Returns:
        samples [batch_size, nc, H, W]
    """

    logit_probs = x_mean[:, :nmix]
    batch_size, nmix, H, W = logit_probs.size()

    # [BATCH, nmix, nc, H, W]
    means     = x_mean[:, nmix:(nc + 1) * nmix].view(batch_size, nmix, nc, H, W)
    logscales = x_mean[:, (nc + 1) * nmix:(nc * 2 + 1) * nmix].view(batch_size, nmix, nc, H, W)
    coeffs    = x_mean[:, (nc * 2 + 1) * nmix:(nc * 2 + 4) * nmix].view(batch_size, nmix, nc, H, W)

    logscales   = logscales.clamp(min=-7.)
    logit_probs = F.log_softmax(logit_probs, dim=1)
    coeffs      = coeffs.tanh()

    # [batch_size, 1, H, W] -> [batch_size, nc, H, W]
    index = logit_probs.argmax(dim=1, keepdim=True) + logit_probs.new_zeros(means.size(0), *means.size()[2:]).long()
    # [batch_size, nc, H, W] -> [batch_size, 1, nc, H, W]
    index = index.unsqueeze(1)
    one_hot = means.new_zeros(means.size()).scatter_(1, index, 1)
    # [batch_size, nc, H, W]
    means = (means * one_hot).sum(dim=1)
    logscales = (logscales * one_hot).sum(dim=1)
    coeffs = (coeffs * one_hot).sum(dim=1)

    x = means
    if random_sample:
        u = means.new_zeros(means.size()).uniform_(1e-5, 1 - 1e-5)
        x = x + logscales.exp() * (torch.log(u) - torch.log(1.0 - u))
    # [batch_size, H, W]
    if nc==3:
        x0 = x[:, 0].clamp(min=-1., max=1.)
        x1 = (x[:, 1] + coeffs[:, 0] * x0).clamp(min=-1., max=1.)
        x2 = (x[:, 2] + coeffs[:, 1] * x0 + coeffs[:, 2] * x1).clamp(min=-1., max=1.)
        x = torch.stack([x0, x1, x2], dim=1)
    elif nc==1:
        x = (x * coeffs).clamp(min=-1., max=1.)

    return x


if __name__ == "__main__":
    pass
