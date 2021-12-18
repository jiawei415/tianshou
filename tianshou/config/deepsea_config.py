alg_config = dict(
    target_update_freq=4,
    hyper_reg_coef=0.,
    based_weight_decay=0.,
    hyper_weight_decay=0.,
    prior_scale=10.,
    hidden_sizes=[64,64],
    init_type='trunc_normal',
    step_per_collect=1,
)

'''successful for HyperC51
    alg_config = dict(
        size=10,
        num_atoms=51,
        v_max=10,
        noise_dim=2,
        buffer_size=50000,
        based_weight_decay=0.0003125,
        hyper_weight_decay=0,
        hyper_reg_coef=0.01,
        noise_dim=2,
        prior_std=2,
        prior_scale=1.,
        init_type=None,
        batch_size=128,
        target_update_freq=100,
        step_per_collect=2,
        hidden_sizes=[512,512],
    )
'''

'''successful for HyperDQN
    alg_config = dict(
        size=10,
        num_atoms=1,
        noise_dim=2,
        buffer_size=50000,
        based_weight_decay=0.0003125,
        hyper_weight_decay=0,
        hyper_reg_coef=0.01,
        noise_dim=2,
        prior_std=2,
        prior_scale=1.,
        init_type=None,
        batch_size=128,
        target_update_freq=100,
        step_per_collect=2,
        hidden_sizes=[512,512],
    )
'''