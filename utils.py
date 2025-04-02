from ebrb_torch import Rule_Base
import torch

def create_brb(X, y, reference_num, N):
    T = X.shape[1]
    L = X.shape[0]
    antecedent_reference_value = torch.zeros((T, reference_num))
    for t in range(T):
        antecedent_reference_value[t] = torch.linspace(torch.min(X[:, t]), torch.max(X[:, t]), reference_num)
    consequent_reference = torch.linspace(torch.min(y), torch.max(y), N)
    brb = Rule_Base(T, N, L, antecedent_reference_value, consequent_reference, reference_num)
    return brb