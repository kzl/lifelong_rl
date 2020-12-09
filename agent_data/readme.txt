The agent_data folder stores files of transitions (to be loaded in as offline data/demos for an agent).

The files should be .pkl files stored in bytes as numpy arrays with shape:
    (N, obs_dim + action_dim + 1 (reward) + 1 (terminal) + obs_dim)
i.e. a concatenated array of N transitions: (obs, action, reward, terminal, next_obs)

Snapshots typically contain this in the "replay_buffer/transitions" key of the .pt dict.
