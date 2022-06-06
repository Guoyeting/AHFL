from config import *
from util.rdp_accountant import *

# global privacy adjustment
def compute_new_sampling_ratio(total_time_recomputed, it_each_local, tau_config, it_each_global, rdp_global, orders_global, sample_ratio):
    expected_round = int(
        (max_time - total_time_recomputed) / (it_each_local * tau_config + it_each_global))
    while True:
        rdp_global_tmp = rdp_global + compute_rdp(q=sample_ratio, noise_multiplier=sigma_global,
                                                  steps=4 * expected_round,
                                                  orders=orders_global)
        eps, _, opt_order = get_privacy_spent(orders_global, rdp_global_tmp,
                                              target_delta=delta_global)
        if (eps <= epsilon_global or sample_ratio <= 0.75):
            break
        else:
            sample_ratio -= 0.03

    while True:
        rdp_global_tmp = rdp_global + compute_rdp(q=sample_ratio, noise_multiplier=sigma_global,
                                                  steps=4 * expected_round,
                                                  orders=orders_global)
        eps, _, opt_order = get_privacy_spent(orders_global, rdp_global_tmp,
                                              target_delta=delta_global)
        if (eps >= epsilon_global or sample_ratio >= 0.97):
            break
        else:
            sample_ratio += 0.03

    if sample_ratio < 0.97:
        sample_ratio += 0.03

    rdp_global += compute_rdp(q=sample_ratio, noise_multiplier=sigma_global, steps=4,
                              orders=orders_global)
    epsilon_current, _, opt_order = get_privacy_spent(orders_global, rdp_global,
                                                      target_delta=delta_global)
    return sample_ratio, epsilon_current, rdp_global