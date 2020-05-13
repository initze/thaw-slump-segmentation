import torch
import torch.nn.functional as F

def gamma_pdf(alpha, beta, x):
    """
    PDF for a Gamma(alpha, beta) distributed RV.
    """
    return torch.pow(beta, alpha) / torch.exp(torch.lgamma(alpha)) \
         * torch.pow(x, alpha - 1) * torch.exp(-beta * x)


def gamma_nll(alpha, beta, target):
    """
    NLL for a Gamma(alpha, beta) distributed RV.
    """
    # Likelihood     = β^α / Γ(α) x^(α-1) e^(-βx)
    # Log-Likelihood = α log(β) - log(Γ(α)) + (α-1) log(x) - βx\
    ll = alpha * torch.log(beta) \
        - torch.lgamma(alpha) \
        + (alpha - 1) * torch.log(target) \
        - beta * target
    return -torch.mean(ll)


def lp_gamma_pdf(log_alpha, log_beta, x):
    """
    PDF of a log-parametrized gamma distribution
    """
    alpha = torch.exp(log_alpha)
    beta = torch.exp(log_beta)

    return torch.pow(beta, alpha) / torch.exp(torch.lgamma(alpha)) \
         * torch.pow(x, alpha - 1) * torch.exp(-beta * x)


def lp_gamma_nll(log_alpha, log_beta, target):
    """
    Negative log-likelihood of a log-parametrized gamma distribution
    """
    # Log-Likelihood = α log(β) - log(Γ(α)) + (α-1) log(x) - βx\
    alpha = torch.exp(log_alpha)
    beta = torch.exp(log_beta)
    ll = alpha * log_beta \
        - torch.lgamma(alpha) \
        + (alpha - 1) * torch.log(target + 1e-3) \
        - beta * target
    return -torch.mean(ll)


def focal_loss_with_logits(y_hat_log, y, gamma=2):
    log0 = F.logsigmoid(-y_hat_log)
    log1 = F.logsigmoid(y_hat_log)

    gamma0 = torch.pow(torch.abs(1 - y - torch.exp(log0)), gamma)
    gamma1 = torch.pow(torch.abs(y - torch.exp(log1)), gamma)

    return torch.mean(-(1 - y) * gamma0 * log0 - y * gamma1 * log1)
