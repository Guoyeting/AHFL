from config import *
from util.rdp_accountant import *

# local privacy adjustment
def compute_new_sigma_local(tau_config):
    sigma_local = 0.05
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))

    sample_ratio_local = batch_size / (total_data / n_nodes)

    rdp = compute_rdp(q=sample_ratio_local, noise_multiplier=sigma_local,
                      steps=int(tau_config), orders=orders)
    epsilon_tmp = get_privacy_spent(orders, rdp, target_delta=delta_local)[0]
    while (epsilon_tmp > epsilon_local):
        sigma_local += 0.01
        rdp = compute_rdp(q=sample_ratio_local, noise_multiplier=sigma_local,
                          steps=int(tau_config), orders=orders)
        epsilon_tmp = get_privacy_spent(orders, rdp, target_delta=delta_local)[0]
        if sigma_local > 0.25:
            break
    if sigma_local > 0.25:
        sigma_local = 0
        split_point = 4
        time_gen_tmp = time_gen[-1]
    else:
        split_point = 1
        time_gen_tmp = time_gen[0]

    return sigma_local, split_point, time_gen_tmp