# Delay added to joint angle feedback. 

from numpy import *
# for using np.sum and np.abs (since standard library has these as well)
import numpy as np
from scipy.linalg import solve
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numba
# for making and displaying animation
# from celluloid import Camera
from IPython.display import HTML
from sklearn.neighbors import KDTree

from datetime import datetime
from dask.distributed import Client, worker_client, as_completed, progress
from dask.dataframe import from_delayed
from dask_jobqueue import SLURMCluster

# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import time
import os
import pickle
import sys

SCALE_OBJ_FN = True  # Scales objective fn by reach length squared


# q = q1,q2,dq1,dq2 (dq1 = d/dt q1)

@numba.jit(nopython=True)
def make_M_inv(q, params):
    # params is an instance of the parameters class
    # M2 is the mass matrix M in block diagonal with the identity

    mA = params[0]
    mB = params[1]
    rhoA = params[4]
    rhoB = params[5]
    LA = params[2]
    LB = params[3]
    IA = params[6]
    IB = params[7]
    g = params[8]

    c2 = cos(q[1])

    M = array([[mA * (rhoA ** 2) + mB * (LA ** 2) + 2 * mB * LA * rhoB * c2 + mB * (rhoB ** 2) + IA + IB,
                mB * LA * rhoB * c2 + mB * (rhoB ** 2) + IB],
               [c2 * mB * rhoB * LA + (rhoB ** 2) * mB + IB, (rhoB ** 2) * mB + IB]])

    M_inv = (1.0 / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])) * array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])

    return M_inv


@numba.jit(nopython=True)
def get_muscle_moment_arrays():
    # torque muscle 1 exerts on each body (moment arm * unit torque vector)
    # Unlike 3D case we just do everything in N_hat ref frame here bc the
    # a3_hat, b3_hat, and n3_hat vectors are always aligned
    m1A = 0.02 * array([0., 0., 1.])
    m1B = array([0., 0., 0.])

    # muscle 2
    m2A = 0.02 * array([0., 0., -1.])
    m2B = array([0., 0., 0.])

    # muscle 3
    m3B = 0.02 * array([0., 0., 1.])
    m3A = -m3B

    # muscle 4
    m4B = 0.02 * array([0., 0., -1.])
    m4A = -m4B

    # muscle 5 (bijoint flexor: shoulder in + elbow flex)
    m5A = 0.01 * array([0., 0., 1.])
    m5B = 0.01 * array([0., 0., 1.])

    # muscle 6 (bijoint extensor)
    m6A = 0.01 * array([0., 0., -1.])
    m6B = 0.01 * array([0., 0., -1.])

    moment_arrayA = column_stack((m1A, m2A, m3A, m4A, m5A, m6A))
    moment_arrayB = column_stack((m1B, m2B, m3B, m4B, m5B, m6B))

    return moment_arrayA, moment_arrayB



@numba.jit(nopython=True)
def calc_torques(muscle_act, q, params):
    # returns tauA and tauB
    # muscle_act is vector of muscle activations

    moment_arrayA, moment_arrayB = get_muscle_moment_arrays()

    max_muscle_force = diag(array([params[9], params[10],
                              params[11], params[12],
                              params[20], params[21]]))

    tauA_c, tauB_c = calc_constraint_moments(q, params)

    # The constraint torque on B produces an equal neg torque on A. The constraint torque on A at the shoulder
    # tauA_c, produces an equal neg torque on N, which we ignore since N is fixed
    tauA = dot(moment_arrayA, dot(max_muscle_force, muscle_act)) + tauA_c - tauB_c  # neg torque due to constraint!
    tauB = dot(moment_arrayB, dot(max_muscle_force, muscle_act)) + tauB_c

    return tauA, tauB


@numba.jit(nopython=True)
def calc_constraint_moments(q, params):
    theta1_q1 = -pi / 4
    theta2_q1 = 3*pi / 4
    tauA_c = array([0, 0, 0.1 * exp(-5 * (q[0] - theta1_q1)) - 0.1 * exp(-5 * (theta2_q1 - q[0])) - 0.05 * q[2]])

    theta1_q2 = 0.0
    theta2_q2 = 3 * pi / 4
    tauB_c = array([0, 0, 0.1 * exp(-5 * (q[1] - theta1_q2)) - 0.1 * exp(-5 * (theta2_q2 - q[1])) - 0.05 * q[3]])

    return tauA_c, tauB_c


@numba.jit(nopython=True)
def make_rhs_arm(t, q, m, ext_force, params):
    # m is a vector of muscle activations, not a fn
    q1, q2, dq1, dq2 = q

    mA = params[0]
    mB = params[1]
    rhoA = params[4]
    rhoB = params[5]
    LA = params[2]
    LB = params[3]
    IA = params[6]
    IB = params[7]
    g = params[8]

    c1 = cos(q1)
    c2 = cos(q2)
    s1 = sin(q1)
    s2 = sin(q2)

    f1, f2 = ext_force  # external force

    tauA, tauB = calc_torques(m, q, params)

    F1 = -rhoA * mA * g * c1 + -mB * g * LA * c1 - mB * g * rhoB * (-s1 * s2 + c1 * c2) + \
         f1 * (-LA * s1 + LB * (-c1 * s2 - s1 * c2)) + f2 * (LA * c1 + LB * (-s1 * s2 + c1 * c2)) + \
         dot(tauA, array([0, 0, 1.0])) + dot(tauB, array([0, 0, 1.0]))

    F2 = -rhoB * (-s1 * s2 + c1 * c2) * mB * g + f1 * LB * (-c1 * s2 - s1 * c2) + f2 * LB * (-s1 * s2 + c1 * c2) + \
         dot(tauB, array([0, 0, 1.0]))

    velocity_terms = array([mB * LA * rhoB * ((dq1 + dq2) ** 2) * s2 - mB * rhoB * LA * (dq1 ** 2) * s2,
                            -s2 * rhoB * mB * LA * (dq1 ** 2)])

    M_inv = make_M_inv(q, params)
    temp_rhs = array([F1 + velocity_terms[0], F2 + velocity_terms[1]])  # rhs before inverting the mass matrix
    rhs = hstack((array([dq1, dq2]), dot(M_inv, temp_rhs)))

    return rhs


@numba.jit(nopython=True)
def make_rhs_muscles(t, m, x, D0, D1, params):

    rhs = (1.0 / params[13]) * (-m + sigma(D0 + dot(D1, x)))

    return rhs


@numba.jit(nopython=True)
def make_rhs_brain(t, x, u1, u2, u3, u4, v1, v3, v4, A, B0, B1, B2, B3, B4, C1, C3, C4, params):
    # x: "firing rate" of neurons
    # p: parameters for A, B1, B2, B3, C as a vector [pA, pB1, pB2, pB3, pC]
    # u1: vector giving the target position at time t (u1 evaluated at time t)
    # u2: vector giving start position for current reach
    # v1, v2, v3: vector of feedback information (v1, v2 evaluated at q) (v1 = pos hand, v2 = vel hand)

    rhs = (dot(A, x) + B0 + dot(B1, u1) + dot(B2, u2) + dot(B3, u3) + dot(B4, u4)
           + dot(C1, v1) + dot(C3, v3) + dot(C4, v4))

    return rhs


@numba.jit(nopython=True)
def make_rhs_forward(t, z, start, target, q_start, q_target,
                     A, B0, B1, B2, B3, B4, C1, C3, C4, D0, D1, params, dims):
    # z is the concatenated variable of x (neuron states), m (muscle activation), and q (arm states)
    # u1_fn is a fn of t, giving target position
    # v1_fn is a fn of q, giving feedback (i.e. hand position)

    x, m, q = z_to_xmq(z, dims)

    u1 = u1_fn(t, target)
    u2 = u2_fn(t, start)
    u3 = u3_fn(t, q_target)
    u4 = u4_fn(t, q_start)
    ext_force = ext_force_fn(t)
    v1 = v1_fn(q, params)
    v3 = v3_fn(m)
    v4 = v4_fn(q, params)

    rhs1 = make_rhs_brain(t, x, u1, u2, u3, u4, v1, v3, v4,
                          A, B0, B1, B2, B3, B4, C1, C3, C4, params)

    rhs2 = make_rhs_muscles(t, m, x, D0, D1, params)
    rhs3 = make_rhs_arm(t, q, m, ext_force, params)
    rhs_forward = hstack((rhs1, rhs2, rhs3))

    return rhs_forward


@numba.jit(nopython=True)
def p_to_matrices_2D(p, dims):
    # 2D version of reshape p into matrices A, B0, B1, B2, C1, C3, D0, D1, C2

    # start and stop points on p
    Ass = (0, dims[6])
    B0ss = (Ass[1], Ass[1] + dims[7])
    B1ss = (B0ss[1], B0ss[1] + dims[8])

    C1ss = (B1ss[1], B1ss[1] + dims[9])
    C3ss = (C1ss[1], C1ss[1] + dims[10])

    D0ss = (C3ss[1], C3ss[1] + dims[11])
    D1ss = (D0ss[1], D0ss[1] + dims[12])

    # new stuff at the end
    B2ss = (D1ss[1], D1ss[1] + dims[15])

    B3ss = (B2ss[1], B2ss[1] + dims[18])
    B4ss = (B3ss[1], B3ss[1] + dims[19])
    C4ss = (B4ss[1], B4ss[1] + dims[20])


    A = reshape(p[Ass[0]:Ass[1]], (dims[0], dims[0]))  # reshape fills in row 1, then row 2, etc
    B0 = reshape(p[B0ss[0]:B0ss[1]], (dims[0],))
    B1 = reshape(p[B1ss[0]:B1ss[1]], (dims[0], dims[2]))

    C1 = reshape(p[C1ss[0]:C1ss[1]], (dims[0], dims[3]))
    C3 = reshape(p[C3ss[0]:C3ss[1]], (dims[0], dims[4]))

    D0 = reshape(p[D0ss[0]:D0ss[1]], (dims[1],))
    D1 = reshape(p[D1ss[0]:D1ss[1]], (dims[1], dims[0]))

    B2 = reshape(p[B2ss[0]:B2ss[1]], (dims[0], dims[2]))

    B3 = reshape(p[B3ss[0]:B3ss[1]], (dims[0], dims[16]))
    B4 = reshape(p[B4ss[0]:B4ss[1]], (dims[0], dims[16]))
    C4 = reshape(p[C4ss[0]:C4ss[1]], (dims[0], dims[17]))

    # have to always return everything, bc a numba fn can't return different numbers of outputs depending on
    # the input (i.e. can't have p_to_matrices(p, dims, ('A', 'B')) that returns only A and B.
    return A, B0, B1, B2, B3, B4, C1, C3, C4, D0, D1



@numba.jit(nopython=True)
def z_to_xmq(z, dims):
    # start stop locations
    xss = (0, dims[0])
    mss = (xss[1], xss[1] + dims[1])
    qss = (mss[1], mss[1] + dims[5])

    x = z[xss[0]:xss[1]]
    m = z[mss[0]:mss[1]]
    q = z[qss[0]:qss[1]]

    return x, m, q


@numba.jit(nopython=True)
def pos_a0(q, params):
    return array([0., 0, 0])


@numba.jit(nopython=True)
def pos_b0(q, params):
    # Gives pos of endpoint of a1 in n_hat coords
    q1, q2 = q[0:2]
    LA = params[2]

    # pos_a0 + LA*a1_hat
    a1_hat = array([cos(q1), sin(q1), 0])  # in n_hat coords
    pos = pos_a0(q, params) + LA * a1_hat
    return pos


@numba.jit(nopython=True)
def pos_c0(q, params):
    # Gives pos of endpoint of B in n_hat coords
    # q can be either the full q or just q1, q2.
    q1, q2 = q[0:2]
    LB = params[3]

    c1 = cos(q1)
    c2 = cos(q2)
    s1 = sin(q1)
    s2 = sin(q2)

    # pos_b0 + LB*b1_hat
    b1_hat = array([c1 * c2 - s1 * s2, s1 * c2 + c1 * s2, 0])  # in n_hat coords
    pos = pos_b0(q, params) + LB * b1_hat
    return pos


@numba.jit(nopython=True)
def vel_c0(q, params):
    # velocity of c0 in N
    q1, q2, dq1, dq2 = q
    LA = params[2]
    LB = params[3]

    a2hat = array([-sin(q1), cos(q1), 0])
    b2hat = array([-sin(q1 + q2), cos(q1 + q2), 0])

    vNC0 = (LA * a2hat + LB * b2hat) * dq1 + (LB * b2hat) * dq2
    return vNC0


@numba.jit(nopython=True)
def ext_force_fn(t):
    return array([0.0, 0.0])


# @numba.jit(nopython=True)
# def sigma(x):
#     # x is a vector
#     out = 1.0 / (1.0 + exp(-x))
#     return out
#
# @numba.jit(nopython=True)
# def dsigma_dx(x):
#     # x is a vector
#     out = exp(-x)/((1.0 + exp(-x))**2)
#     return diag(out)


@numba.jit(nopython=True)
def sigma(x):
    # new version that's 0 at -1 and 1 at 1, with 0 derivative at both ends
    # x is a vector
    x1 = 0.5 * (x + 1.0)
    out = -2.0 * (x1 ** 3) + 3.0 * (x1 ** 2)
    out[x < -1.0] = 0.0
    out[x > 1.0] = 1.0
    return out

@numba.jit(nopython=True)
def dsigma_dx(x):
    # x is a vector
    x1 = 0.5 * (x + 1.0)
    out = 0.5 * (-6.0 * (x1 ** 2) + 6.0 * x1)
    out[x < -1.0] = 0.0
    out[x > 1.0] = 0.0
    return diag(out)


@numba.jit(nopython=True)
def u1_fn(t, target):
    # constant u1; for each sample, use lambda t: u1(t, target[i]) as u1
    # target should be 3D vector, since feedback and everything else is 3D vector
    return target


@numba.jit(nopython=True)
def u2_fn(t, start):
    # starting position 3D vector
    return start


@numba.jit(nopython=True)
def u3_fn(t, q_target):
    # q_target is 2D [q1,q2] target
    return q_target


@numba.jit(nopython=True)
def u4_fn(t, q_start):
    # q_start is 2D [q1,q2] start
    return q_start


@numba.jit(nopython=True)
def v1_fn(q, params):
    # feedback function 1
    # Give position of hand in N_hat coords
    q1, q2, dq1, dq2 = q
    q_delay = params[23]
    
    delayed_q = array([q1 - dq1*q_delay, q2 - dq2*q_delay])
    out = pos_c0(delayed_q, params)
    return out

@numba.jit(nopython=True)
def v2_fn(q, params):
    # feedback function 2
    # Give velocity of hand in N_hat coords
    return vel_c0(q, params)


@numba.jit(nopython=True)
def v3_fn(m):
    # feedback function 3
    # muscle activity
    return m

@numba.jit(nopython=True)
def v4_fn(q, params):
    # feedback function 4
    # hand pos in joint angle coords (q1, q2)
    q1, q2, dq1, dq2 = q
    q_delay = params[23]
    
    delayed_q = array([q1 - dq1*q_delay, q2 - dq2*q_delay])
    return delayed_q



def make_init_cond_list(n_samples, dims):
    init_cond_list = []
    x0 = zeros(dims[0])
    m0 = zeros(dims[1])
    for i in range(n_samples):
        q1 = (pi * random.rand()) - (pi/4)
        q2 = (3 * pi / 4) * random.rand()
        dq1 = 0
        dq2 = 0

        q0 = array([q1, q2, dq1, dq2])
        init_cond_list += [hstack((x0, m0, q0))]
    return init_cond_list


def is_reachable(x, ex_kdtree):
    # determine if x is within arm's reachable area by seeing if x is w/in
    # 5 mm of a point in the kdtree
    dist, indx = ex_kdtree.query(reshape(x, (1, 3)), k=1, return_distance=True)
    dist = dist[0, 0]
    indx = indx[0, 0]

    if dist < 0.005:  # in meters
        return True
    else:
        return False


def allowable_point(x, ex_kdtree, reach_area=None):
    # determines if point x is within allowed reach area
    # ex_kdtree is a set of example points w/in the reachable area (see is_reachable())
    # reach_area: if None, any point w/in the arm's reachable area is okay
    # if (x_min, x_max, y_min, y_max), specifies a box within which reaches must lie
    if is_reachable(x, ex_kdtree):
        if reach_area == None:
            return True
        else:
            # has form (x_min, x_max, y_min, y_max)
            (x_min, x_max, y_min, y_max) = reach_area
            if x[0] > x_min and x[0] < x_max and x[1] > y_min and x[1] < y_max:
                return True
            else:
                return False
    else:
        return False


def draw_pts_from_reach_area(n_samples, ex_kdtree, reach_area=None):
    # draw points uniformly from the specified reach area. If reach_area=None, draw
    # points from the whole reachable space

    if reach_area == None:
        (x_min, x_max, y_min, y_max) = (-0.298, 0.298, -0.298, 0.298)
    else:
        (x_min, x_max, y_min, y_max) = reach_area

    x_len = x_max - x_min
    y_len = y_max - y_min
    scaler = array([x_len, y_len, 0.0])
    shift = array([x_min, y_min, 0.0])

    pts = []
    for i in range(n_samples):
        is_bad = True  # point is reachable or not
        while is_bad:
            # draw point from given reach box
            pt = random.rand(3) * scaler + shift

            if is_reachable(pt, ex_kdtree):
                pts += [pt]
                is_bad = False
    return pts


def q_pos_obj_fn(q, pos, params):
    # pos is 3D vector
    # q = (q1,q2) here
    return linalg.norm(pos_c0(q, params) - pos)


def get_q_for_pos(pos, ex_q, ex_pos, params, dims):
    # get q1,q2 corresponding to a given pos in x,y,z coords
    # pos: desired starting pos (x,y,z)
    # ex_q: array of example q, each row a q1, q2
    # ex_pos: array of example positions corresponding to the ex_q

    # closest starting pos among samples
    indx = argmin(linalg.norm(ex_pos - pos, axis=1))
    # do a quick optimization to find best associated init cond
    q = minimize(q_pos_obj_fn, ex_q[indx, :],
                   args=(pos, params), method='L-BFGS-B', bounds=((-pi/4, 3*pi/4), (0, 3*pi/4)))
    return q.x


def make_targets_init_conds_from_distribution(n_samples, dist_sampler, reach_area,
                                              ex_kdtree, params, dims):
    # make list of targets and starting positions (and corresponding list of initial conditions)
    # dist_sampler: dist_sampler() returns a samples from some distribution on reach lengths (in meters)
    # reach_area: if None, any point w/in the arm's reachable area is okay
    # if (x_min, x_max, y_min, y_max), specifies a box within which reaches must lie
    # Targets are drawn uniformly from the reach_area

    # ex_kdtree: a kdtree of example points in the reachable space, for determining when a new point is reachable or not

    # make targets (drawn uniformly from the reach_area)
    target_list = draw_pts_from_reach_area(n_samples, ex_kdtree, reach_area)

    start_pos_list = []
    for i in range(n_samples):
        reach_len = dist_sampler()
        is_bad = True
        n_tries = 0
        while is_bad:
            # get point on circle of radius reach_len around target
            theta = random.rand() * 2 * pi
            pt = reach_len * array([cos(theta), sin(theta), 0.0]) + target_list[i]
            if allowable_point(pt, ex_kdtree, reach_area):
                start_pos_list += [pt]
                is_bad = False
            else:
                n_tries += 1

            if n_tries > 100:
                # get a new reach_len, since the current one may be impossible from the given target
                reach_len = dist_sampler()

    # make some example q and corresponding positions
    ex_q = column_stack(((pi * random.rand(200)) - pi/4, (3*pi/4)*random.rand(200)))
    ex_pos = vstack([pos_c0(ex_q[i,:], params) for i in range(len(ex_q))])

    # find initial conds to match start_pos_list, and q for target_list
    x0 = zeros(dims[0])
    m0 = zeros(dims[1])
    init_cond_list = []
    q_target_list = []
    for i in range(n_samples):
        q_init = get_q_for_pos(start_pos_list[i], ex_q, ex_pos, params, dims)
        init_cond_list += [hstack((x0, m0, q_init, zeros(dims[5]//2)))]
        q_target = get_q_for_pos(target_list[i], ex_q, ex_pos, params, dims)
        q_target_list += [q_target]

    return target_list, init_cond_list, start_pos_list, q_target_list


class unif_dist_sampler:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.width = self.high - self.low

    def __call__(self, n=None):
        if n == None:
            return self.width * random.rand() + self.low
        else:
            return self.width * random.rand(n) + self.low


class trunc_exp_sampler:
    def __init__(self, scale, limit):
        self.scale = scale  # mean (if limit=inf; roughly still true if limit >= 4*scale)
        self.limit = limit  # where to truncate exponential distribution

    def __call__(self):
        while True:
            out = random.exponential(self.scale)
            if out < self.limit:
                return out


def make_target_list(n_samples, params):
    target_list = []
    for i in range(n_samples):
        q1 = (pi * random.rand()) - (pi / 4)
        q2 = (3 * pi / 4) * random.rand()
        dq1 = 0
        dq2 = 0

        q = array([q1, q2, dq1, dq2])
        c0 = pos_c0(q, params)
        target_list += [c0]
    return target_list


def make_targets_init_cond_uniformly_from_reach_area(n_samples, reach_area, params, dims):
    # draw start and stop points uniformly from the x,y,z reachable space rather than q1,q2 angle space
    # reach_area = (x_min, x_max, y_min, y_max)

    ex_pts = vstack(make_target_list(int(1e5), params))
    ex_kdtree = KDTree(ex_pts)

    # Draw uniformly pts from reach area
    start_list = draw_pts_from_reach_area(n_samples, ex_kdtree, reach_area)
    target_list = draw_pts_from_reach_area(n_samples, ex_kdtree, reach_area)

    # make some example q and corresponding positions
    ex_q = column_stack(((pi * random.rand(200)) - pi/4, (3 * pi / 4) * random.rand(200)))
    ex_pos = vstack([pos_c0(ex_q[i, :], params) for i in range(len(ex_q))])

    x0 = zeros(dims[0])
    m0 = zeros(dims[1])
    init_cond_list = [hstack((x0, m0, get_q_for_pos(start_list[i], ex_q, ex_pos, params, dims), zeros(dims[5]//2)))
                      for i in range(n_samples)]
    q_start_list = [init_cond_list[i][-4:-2] for i in range(n_samples)]
    q_target_list = [get_q_for_pos(target_list[i], ex_q, ex_pos, params, dims) for i in range(n_samples)]

    return start_list, target_list, q_start_list, q_target_list, init_cond_list


# For plotting
def make_plot(q, params):
    q1, q2, dq1, dq2 = q

    a0 = pos_a0(q, params)
    b0 = pos_b0(q, params)
    c0 = pos_c0(q, params)
    x = array([a0[0], b0[0], c0[0]])
    y = array([a0[1], b0[1], c0[1]])
    return x, y


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# Stuff related to adjoint method calculations

@numba.jit(nopython=True)
def block_diag(*blocks):
    # my version of block_diag to replace scipy's, since it doesn't
    # compile with numba. Requires 2D arrays, so 1D row vectors need to be turned
    # into 2D arrays. This is a numba limitation, since it can't even iterate over the
    # *blocks arg if it's not all the same dimension.
    n_blocks = len(blocks)
    block_dims = zeros((n_blocks, 2), dtype=int64)
    for i, block in enumerate(blocks):
        block_dims[i, :] = shape(block)

    out = zeros((sum(block_dims[:, 0]), sum(block_dims[:, 1])))
    tl = array([0, 0])  # top left indices of current block (row, col)
    for i, block in enumerate(blocks):
        br = tl + block_dims[i, :]
        out[tl[0]:br[0], tl[1]:br[1]] = block
        tl = br

    return out


@numba.jit(nopython=True)
def dv1_dr(q, params):
    # v1(r) is pos_hand(r)
    q1, q2, dq1, dq2 = q

    mA = params[0]
    mB = params[1]
    rhoA = params[4]
    rhoB = params[5]
    LA = params[2]
    LB = params[3]
    IA = params[6]
    IB = params[7]
    g = params[8]
    q_delay = params[23]
    
    c1 = cos(q1 - dq1*q_delay)
    c2 = cos(q2 - dq2*q_delay)
    s1 = sin(q1 - dq1*q_delay)
    s2 = sin(q2 - dq2*q_delay)
    
    df1_dq1 = (-1 * LA * s1 + LB * (-1 * c2 * s1 + -1 * c1 * s2))
    df1_dq2 = LB * (-1 * c2 * s1 + -1 * c1 * s2)
    df2_dq1 = (c1 * LA + LB * (c1 * c2 + -1 * s1 * s2))
    df2_dq2 = LB * (c1 * c2 + -1 * s1 * s2)

    out = array(
        [[df1_dq1, df1_dq2, -df1_dq1*q_delay, -df1_dq2*q_delay, ],
         [df2_dq1, df2_dq2, -df2_dq1*q_delay, -df2_dq2*q_delay, ],
         [0.e-323, 0.e-323, 0.e-323, 0.e-323, ], ]
    )

    return out


@numba.jit(nopython=True)
def dv2_dr(q, params):
    # v2(r) is the velocity of the hand
    q1, q2, dq1, dq2 = q

    c1 = cos(q1)
    s1 = sin(q1)

    LA = params[2]
    LB = params[3]

    out = array(
        [[(-1 * dq2 * LB * cos((q1 + q2)) + dq1 * (-1 * c1 * LA + -1 * LB * cos((q1 + q2)))),
          (-1 * dq1 * LB * cos((q1 + q2)) + -1 * dq2 * LB * cos((q1 + q2))), (-1 * LA * s1 + -1 * LB * sin((q1 + q2))),
          -1 * LB * sin((q1 + q2)), ],
         [(-1 * dq2 * LB * sin((q1 + q2)) + dq1 * (-1 * LA * s1 + -1 * LB * sin((q1 + q2)))),
          (-1 * dq1 * LB * sin((q1 + q2)) + -1 * dq2 * LB * sin((q1 + q2))), (c1 * LA + LB * cos((q1 + q2))),
          LB * cos((q1 + q2)), ], [0., 0., 0., 0., ], ]
    )

    return out


@numba.jit(nopython=True)
def dpos_error_dr(q, params):
    # For objective fn, since pos_error() isn't the same as v1_fn anymore
    q1, q2, dq1, dq2 = q

    c1 = cos(q1)
    c2 = cos(q2)
    s1 = sin(q1)
    s2 = sin(q2)

    mA = params[0]
    mB = params[1]
    rhoA = params[4]
    rhoB = params[5]
    LA = params[2]
    LB = params[3]
    IA = params[6]
    IB = params[7]
    g = params[8]

    out = array(
        [[(-1 * LA * s1 + LB * (-1 * c2 * s1 + -1 * c1 * s2)), LB * (-1 * c2 * s1 + -1 * c1 * s2), 0.e-323, 0.e-323, ],
         [(c1 * LA + LB * (c1 * c2 + -1 * s1 * s2)), LB * (c1 * c2 + -1 * s1 * s2), 0.e-323, 0.e-323, ],
         [0.e-323, 0.e-323, 0.e-323, 0.e-323, ], ]
    )

    return out


@numba.jit(nopython=True)
def dhr_dr(t, q, m, ext_force, params):
    # ext_force is a vector
    q1, q2, dq1, dq2 = q

    c1 = cos(q1)
    c2 = cos(q2)
    s1 = sin(q1)
    s2 = sin(q2)

    mA = params[0]
    mB = params[1]
    rhoA = params[4]
    rhoB = params[5]
    LA = params[2]
    LB = params[3]
    IA = params[6]
    IB = params[7]
    g = params[8]

    f1, f2 = ext_force  # components of external force

    rhs = make_rhs_arm(t, q, m, ext_force, params)
    ddq1, ddq2 = rhs[2:]

    out = array(
        [[0.e-323, 0.e-323, (-0.1e1), 0.e-323, ], [0.e-323, 0.e-323, 0.e-323, (-0.1e1), ], [((
                                                                                                 -0.1e1) * g * LA * mB * s1 + (
                                                                                                         (
                                                                                                             -0.1e1) * g * mA * rhoA * s1 + (
                                                                                                                     g * mB * rhoB * (
                                                                                                                         (
                                                                                                                             -0.1e1) * c2 * s1 + (
                                                                                                                             -0.1e1) * c1 * s2) + (
                                                                                                                                 (
                                                                                                                                     -0.1e1) * f2 * (
                                                                                                                                             (
                                                                                                                                                 -0.1e1) * LA * s1 + LB * (
                                                                                                                                                         (
                                                                                                                                                             -0.1e1) * c2 * s1 + (
                                                                                                                                                             -0.1e1) * c1 * s2)) + (
                                                                                                                                             (
                                                                                                                                                 -0.1e1) * f1 * (
                                                                                                                                                         (
                                                                                                                                                             -0.1e1) * c1 * LA + LB * (
                                                                                                                                                                     (
                                                                                                                                                                         -0.1e1) * c1 * c2 + s1 * s2)) + (
                                                                                                                                                         0.5e0 * exp(
                                                                                                                                                     -5 * (
                                                                                                                                                                 3 / 4 * pi + -1 * q1)) + 0.5e0 * exp(
                                                                                                                                                     -5 * (
                                                                                                                                                                 1 / 4 * pi + q1)))))))),
                                                                                            (0.e-323 + (c2 * (dq1) ** (
                                                                                                2) * LA * mB * rhoB + ((
                                                                                                                           -0.1e1) * c2 * (
                                                                                                                       (
                                                                                                                                   dq1 + dq2)) ** (
                                                                                                                           2) * LA * mB * rhoB + (
                                                                                                                                   -2 * ddq1 * LA * mB * rhoB * s2 + (
                                                                                                                                       (
                                                                                                                                           -0.1e1) * ddq2 * LA * mB * rhoB * s2 + (
                                                                                                                                                   (
                                                                                                                                                       -0.1e1) * f2 * LB * (
                                                                                                                                                               (
                                                                                                                                                                   -0.1e1) * c2 * s1 + (
                                                                                                                                                                   -0.1e1) * c1 * s2) + (
                                                                                                                                                               g * mB * rhoB * (
                                                                                                                                                                   (
                                                                                                                                                                       -0.1e1) * c2 * s1 + (
                                                                                                                                                                       -0.1e1) * c1 * s2) + (
                                                                                                                                                                   -0.1e1) * f1 * LB * (
                                                                                                                                                                           (
                                                                                                                                                                               -0.1e1) * c1 * c2 + s1 * s2)))))))),
                                                                                            (0.5e-1 + (
                                                                                                        2 * dq1 * LA * mB * rhoB * s2 + -2 * (
                                                                                                            dq1 + dq2) * LA * mB * rhoB * s2)),
                                                                                            (0.e-323 + -2 * (
                                                                                                        dq1 + dq2) * LA * mB * rhoB * s2), ],
         [((-0.1e1) * f2 * LB * ((-0.1e1) * c2 * s1 + (-0.1e1) * c1 * s2) + (
                     g * mB * rhoB * ((-0.1e1) * c2 * s1 + (-0.1e1) * c1 * s2) + (-0.1e1) * f1 * LB * (
                         (-0.1e1) * c1 * c2 + s1 * s2))), (c2 * (dq1) ** (2) * LA * mB * rhoB + (
                     (-0.1e1) * ddq1 * LA * mB * rhoB * s2 + (
                         (-0.1e1) * f2 * LB * ((-0.1e1) * c2 * s1 + (-0.1e1) * c1 * s2) + (
                             g * mB * rhoB * ((-0.1e1) * c2 * s1 + (-0.1e1) * c1 * s2) + (
                                 (-0.1e1) * f1 * LB * ((-0.1e1) * c1 * c2 + s1 * s2) + (
                                     0.5e0 * exp(-5 * (3 / 4 * pi + -1 * q2)) + 0.5e0 * exp(-5 * (0.e-323 + q2)))))))),
          (0.e-323 + 2 * dq1 * LA * mB * rhoB * s2), 0.5e-1, ], ]
    )

    return out


@numba.jit(nopython=True)
def dh_dr(t, q, m, ext_force, C1, C4, params, dims):
    q_delay = params[23]
    
    dv4_dr = hstack((eye(2), -q_delay * eye(2)))

    dhx_dr = -(dot(C1, dv1_dr(q, params)) + dot(C4, dv4_dr))
    dhm_dr = zeros((dims[1], dims[5]))

    out = vstack((dhx_dr, dhm_dr, dhr_dr(t, q, m, ext_force, params)))

    return out


@numba.jit(nopython=True)
def dR_dm(q, params, dims):
    moment_arrayA, moment_arrayB = get_muscle_moment_arrays()

    max_muscle_force = diag(array([params[9], params[10],
                                   params[11], params[12],
                                   params[20], params[21]]))

    dR_dtauA = zeros((dims[5], 3), dtype=float64)
    dR_dtauA[2, 2] = 1.0
    dR_dtauB = zeros((dims[5], 3), dtype=float64)
    dR_dtauB[2, 2] = 1.0
    dR_dtauB[3, 2] = 1.0

    dtauA_dm = dot(moment_arrayA, max_muscle_force)
    dtauB_dm = dot(moment_arrayB, max_muscle_force)

    out = dot(dR_dtauA, dtauA_dm) + dot(dR_dtauB, dtauB_dm)
    return out


@numba.jit(nopython=True)
def dh_dm(q, C3, params, dims):

    dhx_dm = -C3
    dhm_dm = (1.0 / params[13]) * eye(dims[1])
    dhr_dm = -dR_dm(q, params, dims)

    out = vstack((dhx_dm, dhm_dm, dhr_dm))
    return out


@numba.jit(nopython=True)
def dh_dx(x, A, D0, D1, params, dims):

    sigma_prime = dsigma_dx(D0 + dot(D1, x))

    dhx_dx = -A
    dhm_dx = -(1.0 / params[13]) * dot(sigma_prime, D1)
    dhr_dx = zeros((dims[5], dims[0]), dtype=float64)

    out = vstack((dhx_dx, dhm_dx, dhr_dx))
    return out


@numba.jit(nopython=True)
def dh_dz(t, x, m, q, ext_force, A, C1, C3, C4, D0, D1, params, dims):
    out1 = dh_dx(x, A, D0, D1, params, dims)
    out2 = dh_dm(q, C3, params, dims)
    out3 = dh_dr(t, q, m, ext_force, C1, C4, params, dims)

    return hstack((out1, out2, out3))


@numba.jit(nopython=True)
def dh_dzdot(q, params, dims):
    q1, q2, dq1, dq2 = q

    c1 = cos(q1)
    c2 = cos(q2)
    s1 = sin(q1)
    s2 = sin(q2)

    mA = params[0]
    mB = params[1]
    rhoA = params[4]
    rhoB = params[5]
    LA = params[2]
    LB = params[3]
    IA = params[6]
    IB = params[7]
    g = params[8]

    dh_dxdot = vstack((eye(dims[0]),
                       zeros((dims[1], dims[0]), dtype=float64),
                       zeros((dims[5], dims[0]), dtype=float64)))

    dh_dmdot = vstack((zeros((dims[0], dims[1]), dtype=float64),
                       eye(dims[1]),
                       zeros((dims[5], dims[1]), dtype=float64)))

    M_temp = array([[mA * (rhoA ** 2) + mB * (LA ** 2) + 2 * mB * LA * rhoB * c2 + mB * (rhoB ** 2) + IA + IB,
                     mB * LA * rhoB * c2 + mB * (rhoB ** 2) + IB],
                    [c2 * mB * rhoB * LA + (rhoB ** 2) * mB + IB, (rhoB ** 2) * mB + IB]], dtype=float64)
    M = block_diag(eye(2, dtype=float64), M_temp)

    dh_drdot = vstack((zeros((dims[0], dims[5]), dtype=float64),
                       zeros((dims[1], dims[5]), dtype=float64),
                       M))

    out = hstack((dh_dxdot, dh_dmdot, dh_drdot))
    return out


@numba.jit(nopython=True)
def d_dt_dh_dzdot(q, params, dims):
    q1, q2, dq1, dq2 = q
    s2 = sin(q2)

    mB = params[1]
    rhoB = params[5]
    LA = params[2]

    # derivative of mass matrix
    dM_dt = array(
        [[-2 * dq2 * LA * mB * rhoB * s2,-1 * dq2 * LA * mB * rhoB * s2,],[-1 * dq2 * LA * mB * rhoB * s2,0,],]
    )

    dMtilde_dt = block_diag(zeros((2, 2)), dM_dt)
    d_dt_dh_dxdot = zeros((dims[14], dims[0]))
    d_dt_dh_dmdot = zeros((dims[14], dims[1]))
    d_dt_dh_drdot = vstack((zeros((dims[0] + dims[1], dims[5])), dMtilde_dt))

    out = hstack((d_dt_dh_dxdot, d_dt_dh_dmdot, d_dt_dh_drdot))
    return out



@numba.jit(nopython=True)
def d_dpK(K, x):
    nrows = len(K)
    N = len(x)
    ncols = nrows * N
    out = zeros((nrows, ncols))
    indx = 0
    for i in range(nrows):
        out[i, indx:indx + N] = x
        indx += N
    return out


@numba.jit(nopython=True)
def dh_dp(x, u1, u2, u3, u4, v1, v3, v4, A, B1, B2, B3, B4, C1, C3, C4, D0, D1, params, dims):

    sigma_prime = dsigma_dx(D0 + dot(D1, x))

    dh_dpA = vstack((-d_dpK(A, x), zeros((dims[1], dims[6])), zeros((dims[5], dims[6]))))
    dh_dpB0 = vstack(
        (-eye(dims[7]), zeros((dims[1], dims[7])), zeros((dims[5], dims[7]))))
    dh_dpB1 = vstack(
        (-d_dpK(B1, u1), zeros((dims[1], dims[8])), zeros((dims[5], dims[8]))))
    dh_dpC1 = vstack(
        (-d_dpK(C1, v1), zeros((dims[1], dims[9])), zeros((dims[5], dims[9]))))

    dh_dpC3 = vstack(
        (-d_dpK(C3, v3), zeros((dims[1], dims[10])), zeros((dims[5], dims[10]))))

    dh_dpD0 = vstack((zeros((dims[0], dims[11])),
                      -(1.0 / params[13]) * sigma_prime,
                      zeros((dims[5], dims[11]))))

    dh_dpD1 = vstack((zeros((dims[0], dims[12])),
                      -(1.0 / params[13]) * dot(sigma_prime, d_dpK(D1, x)),
                      zeros((dims[5], dims[12]))))

    # new stuff at end
    dh_dpB2 = vstack(
        (-d_dpK(B2, u2), zeros((dims[1], dims[15])), zeros((dims[5], dims[15]))))

    dh_dpB3 = vstack(
        (-d_dpK(B3, u3), zeros((dims[1], dims[18])), zeros((dims[5], dims[18]))))
    dh_dpB4 = vstack(
        (-d_dpK(B4, u4), zeros((dims[1], dims[19])), zeros((dims[5], dims[19]))))
    dh_dpC4 = vstack(
        (-d_dpK(C4, v4), zeros((dims[1], dims[20])), zeros((dims[5], dims[20]))))

    # new stuff (i.e. C4 here) gets stuck on at the end b/c that's where it is in p (see p_to_matrices())
    out = hstack((dh_dpA, dh_dpB0, dh_dpB1, dh_dpC1, dh_dpC3, dh_dpD0, dh_dpD1, dh_dpB2,
                  dh_dpB3, dh_dpB4, dh_dpC4))
    return out



if SCALE_OBJ_FN == True:
    # version scaled by length of reach
    @numba.jit(nopython=True)
    def df_dz(t, x, m, q, start, q_start, u1, u3, v2, params):

        pos_error = pos_c0(q, params) - u1
        q_error = q[0:2] - u3
        reach_len_sq = dot(start - u1, start - u1)
        q_len_sq = dot(q_start - u3, q_start - u3)

        dq_dr = hstack((eye(2), zeros((2, 2))))

        df_dx = 2 * params[15] * x
        df_dm = 2 * params[16] * m
        df_dr = (2 * params[18] * dot(v2, dv2_dr(q, params)) / reach_len_sq +
                 2 * params[14] * dot(pos_error, dpos_error_dr(q, params)) / reach_len_sq +
                 2 * params[19] * dot(q_error, dq_dr) / q_len_sq)

        out = hstack((df_dx, df_dm, df_dr))
        return out


    @numba.jit(nopython=True)
    def obj_fn_integrand(x, m, q, start, q_start, u1, u3, v2, A, D1, params):
        # u1, v1, v3 are vectors, not fns

        pos_error = pos_c0(q, params) - u1
        q_error = q[0:2] - u3
        reach_len_sq = dot(start - u1, start - u1)
        q_len_sq = dot(q_start - u3, q_start - u3)

        pA = reshape(A, (-1,))
        pD1 = reshape(D1, (-1,))

        out = ((params[18] * dot(v2, v2) / reach_len_sq) +
               (params[14] * dot(pos_error, pos_error) / reach_len_sq) +
               params[15] * dot(x, x) +
               params[16] * dot(m, m) +
               params[17] * np.sum(np.abs(pA)) +
               (params[19] * dot(q_error, q_error) / q_len_sq) +
               params[22] * np.sum(np.abs(pD1)))

        return out

else:
    # original versions but with q0, v2 added to keep args the same as scaled version
    @numba.jit(nopython=True)
    def df_dz(t, x, m, q, start, q_start, u1, u3, v2, params):

        pos_error = pos_c0(q, params) - u1
        q_error = q[0:2] - u3

        dq_dr = hstack((eye(2), zeros((2, 2))))

        df_dx = 2 * params[15] * x
        df_dm = 2 * params[16] * m
        df_dr = (2 * params[18] * dot(v2, dv2_dr(q, params)) +
                 2 * params[14] * dot(pos_error, dpos_error_dr(q, params)) +
                 2 * params[19] * dot(q_error, dq_dr))

        out = hstack((df_dx, df_dm, df_dr))
        return out


    @numba.jit(nopython=True)
    def obj_fn_integrand(x, m, q, start, q_start, u1, u3, v2, A, D1, params):
        # u1, v1, v3 are vectors, not fns

        pos_error = pos_c0(q, params) - u1
        q_error = q[0:2] - u3

        pA = reshape(A, (-1,))
        pD1 = reshape(D1, (-1,))

        out = (params[18] * dot(v2, v2) +
               params[14] * dot(pos_error, pos_error) +
               params[15] * dot(x, x) +
               params[16] * dot(m, m) +
               params[17] * np.sum(np.abs(pA)) +
               params[19] * dot(q_error, q_error) +
               params[22] * np.sum(np.abs(pD1)))

        return out


@numba.jit(nopython=True)
def df_dp(A, D1, params, dims):
    # order of things in p is (see p_to_matrices()): A, B0, B1, C1, C3, D0, D1, B2, B3, B4, C4

    # df_dpA
    pA = reshape(A, (-1,))
    df_dpA_temp = zeros(dims[6])
    df_dpA_temp[pA > 1e-8] = 1.0
    df_dpA_temp[pA < -1e-8] = -1.0
    df_dpA = params[17] * df_dpA_temp

    # df stuff bw A and D1
    df_dpB0_B1_C1_C3_D0 = zeros(dims[7] + dims[8] + dims[9] + dims[10] + dims[11])

    # df_dpD1
    pD1 = reshape(D1, (-1,))
    df_dpD1_temp = zeros(dims[12])
    df_dpD1_temp[pD1 > 1e-8] = 1.0
    df_dpD1_temp[pD1 < -1e-8] = -1.0
    df_dpD1 = params[22] * df_dpD1_temp

    # df stuff after D1
    df_dpB2_B3_B4_C4 = zeros(dims[15] + dims[18] + dims[19] + dims[20])

    out = hstack((df_dpA, df_dpB0_B1_C1_C3_D0, df_dpD1, df_dpB2_B3_B4_C4))
    return out


def make_rhs_adjoint(t, y, x, m, q, start, target, q_start, q_target, u1, u3, ext_force, v2, A, C1, C3, C4, D0, D1, params, dims):
    # y is the adjoint variable

    rhs_temp = df_dz(t, x, m, q, start, q_start, u1, u3, v2, params).T + dot(
        dh_dz(t, x, m, q, ext_force, A, C1, C3, C4, D0, D1, params, dims).T -
        d_dt_dh_dzdot(q, params, dims).T, y)
    rhs = solve(dh_dzdot(q, params, dims).T, rhs_temp, overwrite_a=True, overwrite_b=True, check_finite=False)

    return rhs


@numba.jit(nopython=True)
def make_rhs_dFdp_integral(t, y, x, m, q,
                           u1, u2, u3, u4, v1, v3, v4,
                           A, B1, B2, B3, B4, C1, C3, C4, D0, D1, params, dims):
    # y is the adjoint variable

    rhs = df_dp(A, D1, params, dims) + dot(y, dh_dp(x, u1, u2, u3, u4, v1, v3, v4, A, B1, B2, B3, B4, C1, C3, C4, D0, D1, params, dims))
    return rhs


def make_rhs_backward2(t, y, z_fn, start, target, q_start, q_target,
                       A, B1, B2, B3, B4, C1, C3, C4, D0, D1, params, dims):
    # Doesn't include forward ode anymore.
    # z_fn is a function of t, not an integration variable
    # rhs for adjoint ode, gradient integral, and objective fn integral

    # z: [x,m,q]
    # y: adjoint variable (same dim as z)
    x, m, q = z_to_xmq(z_fn(t), dims)

    u1 = u1_fn(t, target)
    u2 = u2_fn(t, start)
    u3 = u3_fn(t, q_target)
    u4 = u4_fn(t, q_start)
    ext_force = ext_force_fn(t)
    v1 = v1_fn(q, params)
    v2 = v2_fn(q, params)
    v3 = v3_fn(m)
    v4 = v4_fn(q, params)

    rhs2 = make_rhs_adjoint(t, y, x, m, q, start, target, q_start, q_target, u1, u3, ext_force, v2,
                            A, C1, C3, C4, D0, D1, params, dims)
    rhs3 = -1 * make_rhs_dFdp_integral(t, y, x, m, q, u1, u2, u3, u4, v1, v3, v4,
                                       A, B1, B2, B3, B4, C1, C3, C4, D0, D1, params, dims)
    rhs4 = -1 * obj_fn_integrand(x, m, q, start, q_start, u1, u3, v2,
                                 A, D1, params)

    rhs = hstack((rhs2, rhs3, rhs4))
    return rhs


# USE VERSION 2; THIS VERSION DOESN'T WORK WELL BC INTEGRATING THE FORWARD ODE BACKWARD BLOWS UP
# def calc_obj_fn_gradient(p, u1_fn, v1_fn, v3_fn, ext_force_fn, target, t_span, init_cond, params, dims):
#     # calc both the obj fn and its gradient
#     # returns (obj fn, grad)
#     A, B0, B1, C1, C3, D0, D1 = p_to_matrices_2D(p, dims)

#     # solve ode forward
#     sol = solve_ivp(lambda t, z: make_rhs_forward(t, z, u1_fn, v1_fn, v3_fn, ext_force_fn, target,
#                                                   A, B0, B1, C1, C3, D0, D1, params, dims),
#                 t_span, init_cond, max_step=0.001, method='RK45')
#     z_end = sol.y[:,-1]

#     # solve backward ode (forward, adjoint, gradient integral, obj fn integral)

#     end_cond = hstack((z_end, zeros(dims[14]), zeros((dims[13])), zeros(1) ))
#     sol = solve_ivp(lambda t, Z: make_rhs_backward(t, Z[0:dims[14]], Z[dims[14]:2*dims[14]],
#                                                   u1_fn, v1_fn, v3_fn, ext_force_fn, target,
#                                                    A, B0, B1, C1, C3, D0, D1, params, dims),
#                     (t_span[1], t_span[0]), end_cond, max_step=0.001, method='RK45')


#     obj_fn = sol.y[-1, -1]
#     grad_p = sol.y[2*dims[14]:2*dims[14]+dims[13], -1]
#     return (obj_fn, grad_p)


def calc_obj_fn_gradient2(p, start, target, q_start, q_target, t_span, init_cond, params, dims):
    # Goes with make_rhs_backward2(); Uses dense soln of forward ode rather than recomputing backward
    # calc both the obj fn and its gradient
    # returns (obj fn, grad)
    A, B0, B1, B2, B3, B4, C1, C3, C4, D0, D1 = p_to_matrices_2D(p, dims)

    # solve ode forward
    sol = solve_ivp(lambda t, z: make_rhs_forward(t, z, start, target, q_start, q_target,
                                                  A, B0, B1, B2, B3, B4, C1, C3, C4, D0, D1, params, dims),
                    t_span, init_cond, method='RK45', dense_output=True)  # rtol=1e-8, atol=1e-8 doesn't help
    z_fn = sol.sol

    # solve backward ode (adjoint, gradient integral, obj fn integral)
    end_cond = hstack((zeros(dims[14]), zeros((dims[13])), zeros(1)))
    q0 = init_cond[-dims[5]:]
    sol2 = solve_ivp(lambda t, Y: make_rhs_backward2(t, Y[0:dims[14]], z_fn,
                                                     start, target, q_start, q_target,
                                                     A, B1, B2, B3, B4, C1, C3, C4, D0, D1, params, dims),
                     (t_span[1], t_span[0]), end_cond, method='RK45')  # rtol=1e-8, atol=1e-8 doesn't help

    obj_fn = sol2.y[-1, -1]
    grad_p = sol2.y[dims[14]:dims[14] + dims[13], -1]
    return (obj_fn, grad_p)


def calc_obj_fn_gradient2_list(p, start_list, target_list, q_start_list, q_target_list,
                               t_span_list, init_cond_list, params, dims):
    # same as calc_obj_fn_gradient2(), but with a list of targets, t_spans, and init_conds instead of just one.
    # target_list[i] is the ith target, and similarly with t_span_list and init_cond_list
    obj_fn_total = 0
    grad_p_total = zeros(dims[13])
    n_targets = len(target_list)
    for i in range(n_targets):
        obj_fn, grad_p = calc_obj_fn_gradient2(p, start_list[i], target_list[i], q_start_list[i], q_target_list[i],
                                               t_span_list[i], init_cond_list[i], params, dims)
        obj_fn_total += obj_fn
        grad_p_total += grad_p

    return (obj_fn_total / n_targets, grad_p_total / n_targets)


def calc_obj_fn_gradient2_list_dask(p, start_list, target_list, q_start_list, q_target_list, t_span_list, init_cond_list,
                                    client, params, dims):
    # similar to processparallelized version, but using dask
    futures = []
    obj_fn_total = 0
    grad_p_total = zeros(dims[13])
    # could also do this within a list comprehension
    n_targets = len(target_list)
    for i in range(n_targets):
        future = client.submit(calc_obj_fn_gradient2, p,
                               start_list[i], target_list[i], q_start_list[i], q_target_list[i],
                               t_span_list[i], init_cond_list[i], params, dims)
        futures += [future]
    for future in as_completed(futures):
        output = future.result()
        obj_fn_total += output[0]
        grad_p_total += output[1]

    return (obj_fn_total / n_targets, grad_p_total / n_targets)


def batch_minimize_LBFGS_dask(obj_fn_gradient, p_init, n_cycles, batch_size, iter_per_batch, save_dir,
                              start_list, target_list, q_start_list, q_target_list, t_span_list, init_cond_list,
                              client, params, dims):
    # n_cycles is the number of times to cycle through all the batches
    av_error = []  # average error over all batches in a cycle
    total_n_samples = len(target_list)
    ss = hstack((arange(0, total_n_samples, batch_size), total_n_samples))  # start, stop indices of each batch
    n_batches = len(ss) - 1
    p_est = copy(p_init)
    for cycle_i in range(n_cycles):
        # make batches (batches change every cycle)
        indices = random.permutation(total_n_samples)
        batch_error = zeros(n_batches)
        for batch_i in range(n_batches):
            batch_indices = indices[ss[batch_i]:ss[batch_i + 1]]

            batch_starts = [start_list[i] for i in batch_indices]
            batch_targets = [target_list[i] for i in batch_indices]
            batch_q_starts = [q_start_list[i] for i in batch_indices]
            batch_q_targets = [q_target_list[i] for i in batch_indices]
            batch_t_span = [t_span_list[i] for i in batch_indices]
            batch_init_cond = [init_cond_list[i] for i in batch_indices]

            batch_n_iter = 0
            batch_n_stops = 0
            stop_iter = False
            while not stop_iter:
                # minimize
                extra_args = (
                    batch_starts, batch_targets, batch_q_starts, batch_q_targets, batch_t_span, batch_init_cond,
                    client, params, dims)
                options = {'maxiter': iter_per_batch - batch_n_iter, 'gtol': 1e-7,
                           'ftol': 1e-15}  # optimization options

                result = minimize(obj_fn_gradient, p_est, extra_args, method='L-BFGS-B', jac=True, options=options)

                batch_n_iter += result.nit
                batch_n_stops += 1
                p_est = result.x

                # Determine whether to stop iter on this batch
                if batch_n_iter >= iter_per_batch:
                    stop_iter = True
                elif batch_n_stops >= 3:
                    stop_iter = True
                else:
                    stop_iter = False

            batch_error[batch_i] = result.fun
            print("batch " + str(batch_i) + " error: ", batch_error[batch_i])

        info_string = ("cycle_" + str(cycle_i) + ".npy")
        save(save_dir + info_string, p_est)

        av_error += [mean(batch_error)]
        print("average error over cycle " + str(cycle_i) + ": ", av_error[-1])

    return result, av_error

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
