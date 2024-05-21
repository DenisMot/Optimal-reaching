from basic_fns_v12_delay_simplified_nodict import *


N_NEURONS = 20
N_SAMPLES = 10000
RANDOM_SEED = 163767

T_SPAN = 1.0

BETA0 = 0  # penalty on ||vel_hand||^2 / reach_len_sq (path length)
BETA1 = 1e1  # penalty on ||pos_error||^2 / reach_len_sq
BETA3 = 4e-2  # penalty on ||x||^2  (neurons)
BETA4 = 8e-2  # penalty on ||m||^2  (muscles)
BETA5 = 8e-4  # penalty on ||A||_L1 for v12 and above (for 20 nrns)
BETA6 = 1e1  # penalty on ||q_error||^2 / q_len_sq
BETA7 = 1e-4  # penalty on ||D1||_L1 for v12 and above (for 20 nrns)

Q_DELAY = 0.0 # feedback delay (in sec)


dir_str = str(Q_DELAY) + "_feedback_delay_" + "v12"

if not os.path.exists("optimization_output/" + dir_str):
    os.makedirs("optimization_output/" + dir_str)

save_dir = "optimization_output/" + dir_str + "/"

RANDOM_P_INIT = True

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# Set params and dims for velocity feedback model

# dimensions
dim_neurons = N_NEURONS  # number of neurons
dim_muscles = 6 # number of muscles
dim_target1 = 3 # dimension of target (output of u1)
dim_feedback1 = 3 # dimension of feedback (output of v1)
dim_feedback3 = dim_muscles # dimension of output of v3
dim_arm = 2*2 # dimension of arm state variables (q, dq)

lenA = dim_neurons*dim_neurons
lenB0 = dim_neurons
lenB1 = dim_neurons*dim_target1 # u1
lenC1 = dim_neurons*dim_feedback1 # v1
lenC3 = dim_neurons*dim_feedback3 # v3
lenD0 = dim_muscles
lenD1 = dim_muscles*dim_neurons # muscles

lenB2 = dim_neurons*dim_target1  # start pos input

dim_target2 = 2 # output of u3
dim_feedback4 = 2 # q feedback, dq feedback
lenB3 = dim_neurons*dim_target2  # q input
lenB4 = dim_neurons*dim_target2  # q start input
lenC4 = dim_neurons*dim_feedback4  # q feedback


dim_p = lenA + lenB0 + lenB1 + lenB2 + lenB3 + lenB4 + lenC1 + lenC3 + lenC4 + lenD0 + lenD1
dim_z = dim_neurons + dim_muscles + dim_arm

# order is same as search_and_replace_dict_basic_fns_v6 (new stuff added at end)!
dims = (dim_neurons, dim_muscles, dim_target1, dim_feedback1, dim_feedback3, dim_arm,
        lenA, lenB0, lenB1, lenC1, lenC3, lenD0, lenD1, dim_p, dim_z, lenB2,
        dim_target2, dim_feedback4, lenB3, lenB4, lenC4)



# params
mA = 0.294 # mass of body A (kg) (upper arm)
mB = 0.194 # (forearm)
LA = 0.144 # length of body A (meters)
LB = 0.154
rhoA = 0.5*LA # distance to center of mass of body A (must be < LA)
rhoB = 0.44*LB
IA = 3.24e-4 # moment of inertia of body A w/ respect to center of mass (kg*m^2)
IB = 2.71e-4
g = 0.0
# Max muscle force
m1_max = 60.0  # shoulder in
m2_max = 60.0  # shoulder out
m3_max = 60.0  # bicep
m4_max = 60.0  # tricep
m5_max = 60.0  # bijoint flexor
m6_max = 60.0  # bijoint extensor

# muscle time constant (in sec)
muscle_tc = 0.02 # (sec) (activation is 10 ms, deactivation is 40 ms, so do something in between)

# objective fn params
beta1 = BETA1  # penalty on ||pos_error||^2 / reach_len_sq
beta3 = BETA3  # penalty on ||x||^2  (neurons)
beta4 = BETA4  # penalty on ||m||^2  (muscles)
beta5 = BETA5  # penalty on ||A||_L1 for v12 and above (for 20 nrns)
beta0 = BETA0  # penalty on ||vel_hand||^2 / reach_len_sq (path length)
beta6 = BETA6  # penalty on ||q_error||^2 / q_len_sq
beta7 = BETA7  # penalty on ||D1||_L1 for v12 and above (for 20 nrns)

q_delay = Q_DELAY # feedback delay (in sec)

params = (mA, mB, LA, LB, rhoA, rhoB, IA, IB, g, m1_max, m2_max, m3_max, m4_max,
          muscle_tc, beta1, beta3, beta4, beta5, beta0, beta6, m5_max, m6_max, beta7, q_delay)


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # dask optimization
    client = Client(threads_per_worker=1, n_workers=8)  # using dask

    random.seed(RANDOM_SEED)
    if RANDOM_P_INIT == True:
        p_init = 2*(random.rand(dim_p)-0.5)*sqrt(2.0 / dim_p)
    else:
        p_init = load(P_INIT_FILE)

    n_samples = N_SAMPLES
    if T_SPAN == 'random':
        t_span_list = [(0.0, 1.5*random.rand() + 0.25) for i in range(n_samples)]
    else:
        t_span_list = [(0, T_SPAN)] * n_samples

    # draw pts uniformly from reachable area
    reach_area = (-0.298, 0.298, -0.298, 0.298)
    (start_list,
     target_list,
     q_start_list,
     q_target_list,
     init_cond_list) = make_targets_init_cond_uniformly_from_reach_area(n_samples, reach_area, params, dims)

    n_cycles = 100
    batch_size = 200
    iter_per_batch = 10
    # save params, dims, and any other identifying data
    params_data = {'params': params, 'dims': dims, 'p_init': p_init,
                   't_span_list': t_span_list, 'start_list': start_list, 'target_list': target_list,
                   'q_start_list': q_start_list, 'q_target_list': q_target_list, 'init_cond_list': init_cond_list}
    with open(save_dir + "params_data.pickle", 'wb') as file:
        pickle.dump(params_data, file)

    result = batch_minimize_LBFGS_dask(calc_obj_fn_gradient2_list_dask, p_init, n_cycles, batch_size,
                                       iter_per_batch, save_dir,
                                       start_list, target_list, q_start_list, q_target_list,
                                       t_span_list, init_cond_list,
                                       client, params, dims)

    # save final result and other info
    with open(save_dir + "final_result.pickle", 'wb') as file:
        pickle.dump(result, file)

    client.shutdown()
