alg_config = dict(
    num_atoms=51,
    v_max=100,
    noise_dim=32,
    buffer_size=int(2e5),
    hyper_reg_coef=0.01,
    prior_scale=1.,
    init_type=None,
    batch_size=320,
    target_update_freq=100,
    step_per_collect=10,
)

'''successful for HyperC51
    alg_config = dict(
        num_atoms=51,
        v_max=100,
        noise_dim=2,
        buffer_size=50000,
        based_weight_decay=0.0003125,
        hyper_weight_decay=0.0003125,
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
