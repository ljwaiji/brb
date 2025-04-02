import torch


def triangle_antc_match_GPU(data, num, reference_num, reference):
    match = torch.zeros((num, reference_num), device=data.device, requires_grad=False)
    for i in range(num):
        for j in range(reference_num - 1):
            if reference[i, j + 1] >= data[i] >= reference[i, j]:
                match[i, j + 1] = (reference[i, j] - data[i]) / (reference[i, j] - reference[i, j + 1])
                match[i, j] = 1 - match[i, j + 1]
    return match


class Rule_Base:
    def __init__(self, T, N, L, antecedent_reference_value, consequent_reference, reference_num, device="cuda"):
        self.device = device
        self.T = T
        self.N = N
        self.L = L
        self.antecedent_reference_value = antecedent_reference_value.to(self.device)
        self.consequent_reference = consequent_reference.to(self.device) if consequent_reference is not None else None
        self.reference_num = reference_num


def transform_input(X, brb, M):
    brb_input = torch.zeros((M, brb.T, brb.reference_num), device=X.device, requires_grad=False)
    for m in range(M):
        brb_input[m] = triangle_antc_match_GPU(X[m], brb.T, brb.reference_num, brb.antecedent_reference_value)
    return brb_input


class Inference:
    def __init__(self, M, L, N, T, antecedent_mask_r, brb_input, theta, delta, beta, antecedent_reference_value,
                 consequent_reference=None, sigma=1e-8, a=torch.tensor(1), b=torch.tensor(0.5), c=torch.tensor(0), device="cuda", task=''):
        self.device = device
        self.Y = None
        self.sigma = sigma
        self.M = M
        self.L = L
        self.T = T
        self.N = N
        self.brb_input = brb_input.to(device)
        self.antecedent_mask_r = antecedent_mask_r.to(device)
        self.delta = delta.to(device)
        self.theta = theta.to(device)
        self.beta = beta.T.to(device)
        self.antecedent_reference_value = antecedent_reference_value.to(device)
        self.consequent_reference = consequent_reference.to(device) if consequent_reference is not None else None
        self.max_shape = antecedent_mask_r.shape[-1]
        self.a = a
        self.b = b
        self.c = c
        self.task = task

    def compute_alpha(self):
        mask = self.brb_input.unsqueeze(1) * self.antecedent_mask_r  # shape = M,L,T_k,max
        sum_mask = torch.sum(mask, dim=3)  # shape = M,L,T_k
        p = torch.pow(sum_mask, self.delta)
        alpha = torch.prod(p, dim=2)  # shape = M,L
        return alpha.to(self.device)

    def compute_W(self, alpha):
        alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)
        sum_alpha = torch.sum(alpha * self.theta, dim=-1, keepdim=True)
        return (self.theta * alpha) / sum_alpha

    def ER_forward(self, w):
        sigma_temp = torch.sum(self.beta, dim=0)
        diff = self.beta - sigma_temp
        w_expanded = w.unsqueeze(1)
        p = w_expanded * diff
        prod = torch.prod(p + 1, dim=2)
        sigma1 = torch.sum(prod, dim=1)
        sigma = torch.sum(self.beta, dim=0)
        prod_for_mu = (self.N - 1) * torch.prod(1 - w_expanded * sigma, dim=2)
        mu = 1 / (sigma1 - prod_for_mu.squeeze(-1))
        up = mu.unsqueeze(-1) * (prod - torch.prod(1 - w_expanded * sigma, dim=2))
        down = 1 - mu * torch.prod(1 - w_expanded, dim=2).squeeze(-1)
        Beta = up / down.unsqueeze(-1)
        return Beta

    def execute(self):
        alpha = self.compute_alpha()
        w = self.compute_W(alpha)
        # numpy_w = w.detach().cpu().numpy()
        assert torch.any(w >= 0)
        Beta = self.ER_forward(w)
        assert torch.any(Beta >= 0)
        if self.task == 'classification':
            self.Y = Beta
        else:
            self.Y = torch.matmul(Beta, self.consequent_reference)
        return self.Y
