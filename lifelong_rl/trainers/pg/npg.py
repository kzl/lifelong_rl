import numpy as np
import torch

from lifelong_rl.trainers.pg.pg import PGTrainer
import lifelong_rl.torch.pytorch_util as ptu


def cg_solve(f_Ax, b, x_0=None, cg_iters=10, residual_tol=1e-10):
    x = np.zeros_like(b) #if x_0 is None else x_0
    r = b.copy() #if x_0 is None else b-f_Ax(x_0)
    p = r.copy()
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x


class NPGTrainer(PGTrainer):

    """
    Natural Policy Gradient with conjugate gradient to estimate the Hessian.
    Policy gradient algorithm with normalized update step.
    See https://github.com/aravindr93/mjrl/blob/master/mjrl/algos/npg_cg.py.
    """

    def __init__(
            self,
            normalized_step_size=0.01,
            FIM_invert_args=None,
            hvp_sample_frac=1,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.normalized_step_size = normalized_step_size
        self.FIM_invert_args = FIM_invert_args if FIM_invert_args is not None else {'iters': 10, 'damping': 1e-4}
        self.hvp_sample_frac = hvp_sample_frac

    def CPI_surrogate(self, obs, actions, advantages, old_policy):
        adv_var = torch.autograd.Variable(advantages, requires_grad=False)
        log_probs = torch.squeeze(self.policy.get_log_probs(obs, actions), dim=-1)
        log_probs_old = torch.squeeze(old_policy.get_log_probs(obs, actions), dim=-1)
        LR = torch.exp(log_probs - log_probs_old)
        surr = torch.mean(LR * adv_var)
        return surr

    def flat_vpg(self, obs, actions, advantages, old_policy):
        cpi_surr = self.CPI_surrogate(obs, actions, advantages, old_policy)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).cpu().data.numpy() for g in vpg_grad])
        return vpg_grad, cpi_surr

    def HVP(self, observations, actions, old_policy, vector, regu_coef=None):
        regu_coef = self.FIM_invert_args['damping'] if regu_coef is None else regu_coef
        vec = torch.autograd.Variable(ptu.from_numpy(vector).float(), requires_grad=False)
        if self.hvp_sample_frac is not None and self.hvp_sample_frac < 0.99:
            num_samples = observations.shape[0]
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_sample_frac*num_samples))
            obs = observations[rand_idx]
            act = actions[rand_idx]
        else:
            obs = observations
            act = actions
        log_probs = torch.squeeze(self.policy.get_log_probs(obs, act), dim=-1)
        log_probs_old = torch.squeeze(old_policy.get_log_probs(obs, act), dim=-1)
        mean_kl = (log_probs_old - log_probs).mean()
        grad_fo = torch.autograd.grad(mean_kl, self.policy.trainable_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        h = torch.sum(flat_grad*vec)
        hvp = torch.autograd.grad(h, self.policy.trainable_params)
        hvp_flat = np.concatenate([g.contiguous().view(-1).cpu().data.numpy() for g in hvp])
        return hvp_flat + regu_coef * vector

    def build_Hvp_eval(self, inputs, regu_coef=None):
        def eval(v):
            full_inp = inputs + [v] + [regu_coef]
            Hvp = self.HVP(*full_inp)
            return Hvp
        return eval

    def train_policy(self, batch, old_policy):
        obs = ptu.from_numpy(batch['observations'])
        actions = ptu.from_numpy(batch['actions'])
        advantages = ptu.from_numpy(batch['advantages'])

        log_probs = torch.squeeze(self.policy.get_log_probs(obs, actions), dim=-1)
        log_probs_old = torch.squeeze(old_policy.get_log_probs(obs, actions), dim=-1)
        kl = (log_probs_old - log_probs).mean()

        vpg_grad, cpi_surr = self.flat_vpg(obs, actions, advantages, old_policy)
        hvp = self.build_Hvp_eval([obs, actions, old_policy], regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(), cg_iters=self.FIM_invert_args['iters'])

        alpha = np.sqrt(np.abs(self.normalized_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))

        cur_params = self.policy.get_param_values()
        new_params = cur_params + alpha * npg_grad
        self.policy.set_param_values(new_params)

        return -cpi_surr, kl
