import pickle


def add_transitions(replay_buffer, data_file, obs_dim, action_dim, max_transitions=int(1e8)):
    with open('agent_data/%s.pkl' % data_file, 'rb') as f:
        transitions = pickle.load(f)

    # method signature: add_sample(obs, act, r, d, next_obs, info)
    n_transitions = min(len(transitions), max_transitions)

    # in form (s, a, r, d, s')
    for t in range(n_transitions):
        replay_buffer.add_sample(
            transitions[t, :obs_dim],
            transitions[t, obs_dim:obs_dim + action_dim],
            transitions[t, obs_dim + action_dim:obs_dim + action_dim + 1],
            transitions[t, obs_dim + action_dim + 1:obs_dim + action_dim + 2],
            transitions[t, -obs_dim:],
            env_info={},
        )
