import numpy as np
import torch

import lifelong_rl.torch.pytorch_util as ptu


def calculate_contrastive_empowerment(
        discriminator,
        obs,
        next_obs,
        latents,
        num_prior_samples=512,
        distribution_type='uniform',
        split_group=4096*32,
        obs_mean=None,
        obs_std=None,
        return_diagnostics=False,
        prior=None,
):
    """
    Described in Sharma et al 2019.
    Approximate variational lower bound using estimate of s' from s, z.
    Uses contrastive negatives to approximate denominator.
    """

    discriminator.eval()

    if obs_mean is not None:
        obs = (obs - obs_mean) / (obs_std + 1e-6)
        next_obs = (next_obs - obs_mean) / (obs_std + 1e-6)

    obs_deltas = ptu.from_numpy(next_obs - obs)
    obs_altz = np.concatenate([obs] * num_prior_samples, axis=0)

    with torch.no_grad():
        logp = discriminator.get_log_prob(
            ptu.from_numpy(obs),
            ptu.from_numpy(latents),
            obs_deltas,
        )
        logp = ptu.get_numpy(logp)

    if distribution_type == 'uniform':
        latent_altz = np.random.uniform(low=-1, high=1, size=(obs_altz.shape[0], latents.shape[1]))
    elif distribution_type == 'prior':
        if prior is None:
            raise AssertionError('prior specified but not passed in')
        obs_t = ptu.from_numpy(obs_altz)
        latent_altz, *_ = prior.get_action(obs_t, deterministic=False)
    else:
        raise NotImplementedError('distribution_type not found')

    # keep track of next obs/delta
    next_obs_altz = np.concatenate([next_obs - obs] * num_prior_samples, axis=0)

    with torch.no_grad():
        if obs_altz.shape[0] <= split_group:
            logp_altz = ptu.get_numpy(discriminator.get_log_prob(
                ptu.from_numpy(obs_altz),
                ptu.from_numpy(latent_altz),
                ptu.from_numpy(next_obs_altz),
            ))
        else:
            logp_altz = []
            for split_idx in range(obs_altz.shape[0] // split_group):
                start_split = split_idx * split_group
                end_split = (split_idx + 1) * split_group
                logp_altz.append(
                    ptu.get_numpy(discriminator.get_log_prob(
                        ptu.from_numpy(obs_altz[start_split:end_split]),
                        ptu.from_numpy(latent_altz[start_split:end_split]),
                        ptu.from_numpy(next_obs_altz[start_split:end_split]),
                    )))
            if obs_altz.shape[0] % split_group:
                start_split = obs_altz.shape[0] % split_group
                logp_altz.append(
                    ptu.get_numpy(discriminator.get_log_prob(
                        ptu.from_numpy(obs_altz[-start_split:]),
                        ptu.from_numpy(latent_altz[-start_split:]),
                        ptu.from_numpy(next_obs_altz[-start_split:]),
                    )))
            logp_altz = np.concatenate(logp_altz)
    logp_altz = np.array(np.array_split(logp_altz, num_prior_samples))

    if return_diagnostics:
        diagnostics = dict()
        orig_rep = np.repeat(np.expand_dims(logp, axis=0), axis=0, repeats=num_prior_samples)
        diagnostics['Pct Random Skills > Original'] = (orig_rep < logp_altz).mean()

    # final DADS reward
    intrinsic_reward = np.log(num_prior_samples + 1) - np.log(1 + np.exp(
        np.clip(logp_altz - logp.reshape(1, -1), -50, 50)).sum(axis=0))

    if not return_diagnostics:
        return intrinsic_reward, (logp, logp_altz, logp - intrinsic_reward)
    else:
        return intrinsic_reward, (logp, logp_altz, logp - intrinsic_reward), diagnostics
