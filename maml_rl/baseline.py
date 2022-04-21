import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class LinearFeatureBaseline(nn.Module):
    """Linear baseline based on handcrafted features, as described in [1] 
    (Supplementary Material 2).

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff

        self.weight = nn.Parameter(torch.Tensor(self.feature_size,),
                                   requires_grad=False).cuda()
        self.weight.data.zero_()
        self._eye = torch.eye(self.feature_size,
                              dtype=torch.float32,
                              device=self.weight.device)
        #print(self.weight.device)

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2).cuda()
        #print(ones.is_cuda)
        observations = episodes.observations.cuda()
        #print(observations.is_cuda)
        time_step = torch.arange(len(episodes)).view(-1, 1, 1).cuda() * ones / len(episodes)
        #print(time_step.is_cuda)
        #print((time_step ** 2))
        return torch.cat([
            observations,
            observations ** 2,
            time_step,
            time_step ** 2,
            time_step ** 3,
            ones
        ], dim=2)

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size
        featmat = self._feature(episodes).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        #print(featmat.size(), featmat[:100, :])
        returns = episodes.returns.view(-1, 1)
        #print(returns.size())
        # Remove blank (all-zero) episodes that only exist because episode lengths vary
        flat_mask = episodes.mask.flatten()
        flat_mask_nnz = torch.nonzero(flat_mask)
        featmat = featmat[flat_mask_nnz].view(-1, self.feature_size)
        returns = returns[flat_mask_nnz].view(-1, 1)
        reg_coeff = self._reg_coeff
        XT_y = torch.matmul(featmat.t(), returns)
        XT_X = torch.matmul(featmat.t(), featmat)
        #print(XT_X.is_cuda)
        #print(XT_y.is_cuda)
        #print(self._eye.is_cuda)
        #sol = torch.linalg.lstsq(XT_y, XT_X + reg_coeff * self._eye)
        #print(sol)
        for _ in range(5):
            try:
                coeffs = torch.linalg.lstsq(XT_y, XT_X + reg_coeff * self._eye).solution
                #print(coeffs)
                # An extra round of increasing regularization eliminated
                # inf or nan in the least-squares solution most of the time
                if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                    #print(coeffs)
                    raise RuntimeError

                break
            except RuntimeError:
                reg_coeff *= 10
        else:
            
            
            raise RuntimeError('Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.weight.copy_(coeffs.flatten())

    def forward(self, episodes):
        features = self._feature(episodes)
        values = torch.mv(features.view(-1, self.feature_size), self.weight)
        return values.view(features.shape[:2])
