"""LogSigma model: encoder -> [CLS] vector -> linear layer with 2 outputs (V, A).

Loss with homoscedastic uncertainty (Kendall, Gal & Cipolla, 2018):
    loss = 0.5 * exp(-s_V) * MSE_V + 0.5 * s_V  +  0.5 * exp(-s_A) * MSE_A + 0.5 * s_A
where s = log(sigma^2) is learned. A task with larger sigma^2 gets a smaller weight 0.5 / sigma^2,
so the model balances V and A automatically instead of using hand-tuned weights.
With use_uncertainty = False the loss is plain MSE_V + MSE_A (ablation).
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class LogSigmaRegressor(nn.Module):
    def __init__(self, encoder_name, dropout, log_var_init_range, use_uncertainty):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, 2)
        lo, hi = log_var_init_range
        # sigma^2 ~ U(lo, hi) at init. The parameters are created even without uncertainty
        # so that the random number stream is identical across runs with the same seed.
        self.log_var_v = nn.Parameter(torch.log(torch.empty(1).uniform_(lo, hi)), requires_grad=use_uncertainty)
        self.log_var_a = nn.Parameter(torch.log(torch.empty(1).uniform_(lo, hi)), requires_grad=use_uncertainty)
        self.use_uncertainty = use_uncertainty

    def forward(self, input_ids, attention_mask, labels=None):
        """Returns (predictions [B, 2], loss or None, (MSE_V, MSE_A) or None)."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        y = self.regressor(self.dropout(h)).float()
        if labels is None:
            return y, None, None
        mse_v = ((y[:, 0] - labels[:, 0]) ** 2).mean()
        mse_a = ((y[:, 1] - labels[:, 1]) ** 2).mean()
        if not self.use_uncertainty:
            return y, mse_v + mse_a, (mse_v, mse_a)
        loss = (0.5 * torch.exp(-self.log_var_v) * mse_v + 0.5 * self.log_var_v
                + 0.5 * torch.exp(-self.log_var_a) * mse_a + 0.5 * self.log_var_a)
        return y, loss, (mse_v, mse_a)

    def sigma2(self):
        """Current (sigma^2_V, sigma^2_A)."""
        return torch.exp(self.log_var_v).item(), torch.exp(self.log_var_a).item()


def build_model(mcfg, use_uncertainty):
    return LogSigmaRegressor(mcfg["encoder"], mcfg["dropout"], mcfg["log_var_init_range"], use_uncertainty)
