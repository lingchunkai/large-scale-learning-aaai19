# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, profile=False, linetrace=False
cimport numpy as np
from .util cimport TransformLL, TransformL, idx_t, VVI, VI, VD, pywrap_GameStructure
import numpy as np
cimport cython
from libc.math cimport exp, log
from libcpp.vector cimport vector
import scipy.sparse
import logging

# x log x when x < EPSILON_ENTROPY is taken as approximately 0
cdef double EPSILON_ENTROPY = 10.**-20

cdef extern from "math.h":
    double INFINITY

cdef double MAX_TAU = 10.**50

LOGGER_DEBUG = 10
LOGGER_INFO = 20
LOGGER_WARNING = 30
cdef int glob_verbosity = LOGGER_DEBUG


def forwardSolve(P_,
        uv_c_I, uv_c_A, uv_p_I, uv_p_A, 
        tau, 
        lambd_=None,
        tol=10.**-10, max_iters=1000,
        Ptype='keep',
        verify=False, 
        prep_struct=None,
        return_only_seq=True,
        duality_freq=10):
    '''
    :param P_ sparse scipy or dense numpy payoff matrix
    :param uv_c_I 2-tuple of children list for infosets for each player
    :param uv_c_A 2-tuple of children list for actions for each plyer
    :param uv_p_I 2-tuple of parents for infosets for each player
    :param uv_p_A 2-tuple of parents for actions for each player
    :param tau step size, larger implies= faster convergence
    :param lambd np ndarray of rationality parameters for each infoset
    :param tol termination criterion, duality gap
    :param max_iters, maximum number of iterations before terminating
    :param Ptype \in {convert, keep}. 
        - use convert to convert P_ to sparse matrix before proceeding
        - use keep to keep P_ in whatever format it was before
    :param prep_struct
        - pywrap_GameStructure which wraps around a c structure storing
          the precomputed game structure
    :param return_only_seq
        - True if we want only to return sequence form answer.
        - False if we want to return behavioral strategy as well.
    :return
        - If return_only_seq = True, then return u, v numpy arrays of
        sequence form strategies.
        - If return_only_seq = False, then return u, v, u_beh, v_beh numpy arrays,
        where u_beh, v_beh are behavioral strategies of the appropriate
        infosets.
    '''
    # Pointers towards c++ VVI's and VI's. 
    cdef VVI* u_c_I
    cdef VVI* v_c_I
    cdef VVI* u_c_A
    cdef VVI* v_c_A
    cdef VI* u_p_I
    cdef VI* v_p_I
    cdef VI* u_p_A
    cdef VI* v_p_A

    # Sizes of action spaces
    cdef idx_t usize, vsize
    
    # Used to store game structure if it has not yet been computed
    cdef VVI u_c_I_, v_c_I_, u_c_A_, v_c_A_
    cdef VI u_p_I_, v_p_I_, u_p_A_, v_p_A_

    # Wrapper for pre-computed game structure
    cdef pywrap_GameStructure ps

    if prep_struct is None:
        u_c_I_ = TransformLL(uv_c_I[0])
        v_c_I_ = TransformLL(uv_c_I[1])
        u_c_A_ = TransformLL(uv_c_A[0])
        v_c_A_ = TransformLL(uv_c_A[1])

        u_p_I_ = TransformL(uv_p_I[0])
        v_p_I_ = TransformL(uv_p_I[1])
        u_p_A_ = TransformL(uv_p_A[0])
        v_p_A_ = TransformL(uv_p_A[1])
                     
        u_c_I = &u_c_I_
        v_c_I = &v_c_I_
        u_c_A = &u_c_A_
        v_c_A = &v_c_A_

        u_p_I = &u_p_I_
        v_p_I = &v_p_I_
        u_p_A = &u_p_A_
        v_p_A = &v_p_A_

        usize = u_p_A.size()
        vsize = v_p_A.size()
    else:
        ps = prep_struct
        u_c_I = &ps.thisptr.u_c_I
        v_c_I = &ps.thisptr.v_c_I
        u_c_A = &ps.thisptr.u_c_A
        v_c_A = &ps.thisptr.v_c_A

        u_p_I = &ps.thisptr.u_p_I
        v_p_I = &ps.thisptr.v_p_I
        u_p_A = &ps.thisptr.u_p_A
        v_p_A = &ps.thisptr.v_p_A
                     
        usize = ps.thisptr.usize
        vsize = ps.thisptr.vsize

    if lambd_ is None:
        u_lambd_ = np.ones(u_p_I[0].size())
        v_lambd_ = np.ones(v_p_I[0].size())
    else:
        u_lambd_, v_lambd_ = lambd_[0], lambd_[1]

    cdef double [:] u_lambd = u_lambd_
    cdef double [:] v_lambd = v_lambd_

    cdef VD u_J = ComputeJ(u_c_A[0], u_lambd_)
    cdef VD v_J = ComputeJ(v_c_A[0], v_lambd_)

    cdef double gap
    
    if Ptype == 'convert': 
        log_msg('Converting to sparse matrix', LOGGER_DEBUG)
        P = scipy.sparse.csr_matrix(P_)
        PT = scipy.sparse.csr_matrix(P.T)
        log_msg('Done converting', LOGGER_DEBUG)
    elif Ptype == 'keep':
        P = P_
        PT = P.T
    else:
        assert False, 'Invalid P matrix type chosen'

    #########################################################
    #
    # Special case: when P is a vector (i.e. one player game)
    #
    #########################################################

    cdef double [:] spec_beh_ret
    cdef double [:] spec_cent
    if P.shape[0] == 1: # u is the 'unimportant player' with 1 action
        # print('     1')
        spec_beh_ret = -np.atleast_1d(np.squeeze(P)).astype(np.float64)
        # print('     2')
        spec_cent = np.ones(vsize)
        # print('     3')
        BestResponseTree(v_c_I[0], v_c_A[0], v_p_I[0], v_p_A[0],
                                v_J,
                                vals = spec_beh_ret,
                                center = spec_cent,
                                tau = MAX_TAU,
                                lambd = v_lambd)
        # print('     4')
        # At this stage, spec_beh_ret contains the *BEHAVIORAL* strategies
        # We compute the sequence form representation and store it in spec_ret
        spec_ret = np.copy(spec_beh_ret)
        # print('     5')
        BehaviorToSeqForm(spec_ret, p_I = v_p_I[0], p_A = v_p_A[0])
        # print('     6')
        if return_only_seq:
            return np.array([1.]), spec_ret
        else:
            return np.array([1.]), spec_ret, np.array([1.]), spec_beh_ret

    elif P.shape[1] == 1: # v is the 'unimportant player'
        spec_beh_ret = np.atleast_1d(np.squeeze(P)).astype(np.float64)
        spec_cent = np.ones(usize)
        BestResponseTree(u_c_I[0], u_c_A[0], u_p_I[0], u_p_A[0],
                                u_J,
                                vals = spec_beh_ret,
                                center = spec_cent,
                                tau = MAX_TAU,
                                lambd = u_lambd)

        # At this stage, spec_beh_ret contains the *BEHAVIORAL* strategies
        # We compute the sequence form representation and store it in spec_ret
        spec_ret = np.copy(spec_beh_ret)
        BehaviorToSeqForm(spec_ret, p_I = u_p_I[0], p_A = u_p_A[0])
        if return_only_seq:
            return spec_ret, np.array([1.])
        else:
            return spec_ret, np.array([1.]), spec_beh_ret, np.array([1.])

    duality_violated_tol = np.sqrt(tol)

    #########################################################
    #
    # Initialization using best response with
    # no centering, no P (i.e. only regularization)
    #
    #########################################################
    u_init = np.zeros(usize)
    u_cent = np.ones(usize)
    BestResponseTree(u_c_I[0], u_c_A[0], u_p_I[0], u_p_A[0],
                            u_J,
                            vals = u_init,
                            center = u_cent,
                            tau = MAX_TAU,
                            lambd = u_lambd) # We want the sequence form here
    BehaviorToSeqForm(u_init, p_I = u_p_I[0], p_A = u_p_A[0])


    v_init = np.zeros(vsize)
    v_cent = np.ones(vsize)
    BestResponseTree(v_c_I[0], v_c_A[0], v_p_I[0], v_p_A[0],
                            v_J,
                            vals = v_init,
                            center = v_cent,
                            tau = MAX_TAU,
                            lambd = v_lambd) # We want the sequence form here
    BehaviorToSeqForm(v_init, p_I = v_p_I[0], p_A = v_p_A[0])

    #########################################################
    #
    # Solve using smoothed iterations (Chambolle and Pock, 2014)
    #
    #########################################################

    u, v = u_init, v_init

    for it in range(max_iters):
        # print(it)
        # PD part 1 (update x^{n+1})
        u_beh = P.dot(v)
        u_cent = np.copy(u)
        BestResponseTree(u_c_I[0], u_c_A[0], u_p_I[0], u_p_A[0],
                                u_J,
                                vals = u_beh,
                                center = u_cent,
                                tau=tau,
                                lambd = u_lambd)
        # At this stage, u_beh contains the *BEHAVIORAL* strategies
        # We compute the sequence form representation and store it in u_vals
        u_vals = np.copy(u_beh)
        BehaviorToSeqForm(u_vals, p_I = u_p_I[0], p_A = u_p_A[0])

        # PD part 2 (update y^{n+1})
        v_beh = -PT.dot(2. * u_vals - u)
        BestResponseTree(v_c_I[0], v_c_A[0], v_p_I[0], v_p_A[0],
                                v_J,
                                vals = v_beh,
                                center = v,
                                tau=tau,
                                lambd = v_lambd)
        # At this stage, v_beh contains the *BEHAVIORAL* strategies
        # We compute the sequence form representation and store it in u_vals
        v_vals = np.copy(v_beh)
        BehaviorToSeqForm(v_vals, p_I = v_p_I[0], p_A = v_p_A[0])

        u, v = u_vals, v_vals

        if it % duality_freq == 0:
            gap = DualityGap(u, v, u_beh, v_beh, P,
                    u_c_I[0], v_c_I[0], 
                    u_p_I[0], v_p_I[0],
                    u_c_A[0], v_c_A[0], 
                    u_p_A[0], v_p_A[0],
                    u_J, v_J,
                    u_lambd, v_lambd,
                    Pv=None, PTu=None)
            # log_msg('At iteration %d Forward Duality gap: %s' % (it, gap), LOGGER_DEBUG)
            if np.abs(gap) < tol:
                # print('Gap < tol. At iteration %d Forward Duality gap: %s' % (it, gap))
                break

    if verify != 'none' and verify is not None and verify is not False:
        if VerifySolution(P, u, v,
                    u_c_I[0], v_c_I[0], 
                    u_p_I[0], v_p_I[0],
                    u_c_A[0], v_c_A[0], 
                    u_p_A[0], v_p_A[0],
                    u_lambd, v_lambd) != True:
            log_msg('Warning: verification of kkt conditions failed (forward)!', LOGGER_WARNING)

    if gap > tol or gap < -duality_violated_tol: 
        log_msg('Termination criterion not met (forward)! Gap: %s' % gap, LOGGER_WARNING)

    if return_only_seq:
        return u, v
    else:
        return u, v, u_beh, v_beh


def WrapperDualityGap(P_,
        uv_c_I, uv_c_A, uv_p_I, uv_p_A,
        u, v, u_beh=None, v_beh=None,
        lambd_=None,
        Ptype='keep',
        prep_struct=None):
    # TODO factor out this part, repeated

    # Pointers towards c++ VVI's and VI's.
    cdef VVI* u_c_I
    cdef VVI* v_c_I
    cdef VVI* u_c_A
    cdef VVI* v_c_A
    cdef VI* u_p_I
    cdef VI* v_p_I
    cdef VI* u_p_A
    cdef VI* v_p_A

    # Sizes of action spaces
    cdef idx_t usize, vsize

    # Used to store game structure if it has not yet been computed
    cdef VVI u_c_I_, v_c_I_, u_c_A_, v_c_A_
    cdef VI u_p_I_, v_p_I_, u_p_A_, v_p_A_

    # Wrapper for pre-computed game structure
    cdef pywrap_GameStructure ps

    if prep_struct is None:
        u_c_I_ = TransformLL(uv_c_I[0])
        v_c_I_ = TransformLL(uv_c_I[1])
        u_c_A_ = TransformLL(uv_c_A[0])
        v_c_A_ = TransformLL(uv_c_A[1])

        u_p_I_ = TransformL(uv_p_I[0])
        v_p_I_ = TransformL(uv_p_I[1])
        u_p_A_ = TransformL(uv_p_A[0])
        v_p_A_ = TransformL(uv_p_A[1])

        u_c_I = &u_c_I_
        v_c_I = &v_c_I_
        u_c_A = &u_c_A_
        v_c_A = &v_c_A_

        u_p_I = &u_p_I_
        v_p_I = &v_p_I_
        u_p_A = &u_p_A_
        v_p_A = &v_p_A_

        usize = u_p_A.size()
        vsize = v_p_A.size()
    else:
        ps = prep_struct
        u_c_I = &ps.thisptr.u_c_I
        v_c_I = &ps.thisptr.v_c_I
        u_c_A = &ps.thisptr.u_c_A
        v_c_A = &ps.thisptr.v_c_A

        u_p_I = &ps.thisptr.u_p_I
        v_p_I = &ps.thisptr.v_p_I
        u_p_A = &ps.thisptr.u_p_A
        v_p_A = &ps.thisptr.v_p_A

        usize = ps.thisptr.usize
        vsize = ps.thisptr.vsize

    if lambd_ is None:
        u_lambd_ = np.ones(u_p_I[0].size())
        v_lambd_ = np.ones(v_p_I[0].size())
    else:
        u_lambd_, v_lambd_ = lambd_[0], lambd_[1]

    cdef double [:] u_lambd = u_lambd_
    cdef double [:] v_lambd = v_lambd_

    cdef VD u_J = ComputeJ(u_c_A[0], u_lambd_)
    cdef VD v_J = ComputeJ(v_c_A[0], v_lambd_)

    cdef double gap

    if Ptype == 'convert':
        log_msg('Converting to sparse matrix', LOGGER_DEBUG)
        P = scipy.sparse.csr_matrix(P_)
        PT = scipy.sparse.csr_matrix(P.T)
        log_msg('Done converting', LOGGER_DEBUG)
    elif Ptype == 'keep':
        P = P_
        PT = P.T
    else:
        assert False, 'Invalid P matrix type chosen'

    #########################################################
    # Compute u_beh, v_beh if it was not provided. Note
    # that there may be numerical problems here.
    #########################################################
    if u_beh is None:
        u_beh = SeqToBehaviorForm(u, u_p_I[0], u_p_A[0])

    if v_beh is None:
        v_beh = SeqToBehaviorForm(v, v_p_I[0], v_p_A[0])


    #########################################################
    # Finally, compute duality gap
    #########################################################
    gap = DualityGap(u, v, u_beh, v_beh, P,
                     u_c_I[0], v_c_I[0],
                     u_p_I[0], v_p_I[0],
                     u_c_A[0], v_c_A[0],
                     u_p_A[0], v_p_A[0],
                     u_J, v_J,
                     u_lambd, v_lambd,
                     Pv=None, PTu=None)

    return gap


cdef VerifySolution(P, np.ndarray u, np.ndarray v,
                    VVI& u_c_I, VVI& v_c_I, 
                    VI& u_p_I, VI& v_p_I,
                    VVI& u_c_A, VVI& v_c_A, 
                    VI& u_p_A, VI& v_p_A,
                    double [:] u_lambd, double [:] v_lambd):
    ''' Verify kkt conditions.
    TODO: verify using *behavioral* as well as sequence form strategy (to avoid u/u_parpar) computations
    '''

    cdef idx_t i_id = 0
    u_sum = P.dot(v)
    v_sum = P.T.dot(u)
    E_u = ConstructE(u_c_I, u_p_I, u_c_A, u_p_A)
    E_v = ConstructE(v_c_I, v_p_I, v_c_A, v_p_A)

    for i_id in range(u_p_I.size()):
        if u_p_I[i_id] < 0:
            u_parpar = 1.
        else: 
            u_para = u_p_I[i_id] 
            u_parpar = u[u_para]
            u_sum[u_para] -= u_lambd[i_id]

        for u_idx in range(u_c_I[i_id].size()):
            u_id = u_c_I[i_id][u_idx]
            u_sum[u_id] += u_lambd[i_id] * (1. + np.log(u[u_id]/u_parpar))
    res_u = scipy.optimize.lsq_linear(E_u.T, u_sum)

    for i_id in range(v_p_I.size()):
        if v_p_I[i_id] < 0:
            v_parpar = 1.
        else: 
            v_para = v_p_I[i_id] 
            v_parpar = v[v_para]
            v_sum[v_para] += v_lambd[i_id]

        for v_idx in range(v_c_I[i_id].size()):
            v_id = v_c_I[i_id][v_idx]
            v_sum[v_id] -= v_lambd[i_id] * (1. + np.log(v[v_id]/v_parpar))
    res_v = scipy.optimize.lsq_linear(E_v.T, v_sum)
    # print(res_u)
    # print(res_v)
    if np.abs(res_u.cost) < 10.**-10 and np.abs(res_v.cost) < 10.**-10:
        return True
    else:
        return False

"""
cdef VerifyOnePlayer(P, np.ndarray u, 
                    VVI& c_I, VI& p_I, VVI& c_A, VVI& p_A,
                    double [:] u_lambd, double [:] v_lambd):
    no.dot(
"""

cdef void BestResponseTree(VVI& c_I, VVI& c_A, 
                    VI& p_I, VI& p_A,
                    VD& J,
                    double [:] vals,
                    double [:] center,
                    double tau,
                    double [:] lambd):
    '''
    Solve for the best response of a single-player game tree.
    :param c_I, c_A, p_I, p_A children and parents of infosets, actions
    :param J array/list of length |A|, containing number of next infosets
    :param vals_orig immediate payoff for each action (i.e. Py if min player)
    :param center centering of Bregman divergence
    :param tau step size
    :param lambd rationality parameters
    :return Behavioral form representation is contained in vals
    ! WARNING Assumes that actions and infosets are ordered in topolgical order!
    ! WARNING Modifies vals and center in-place.
    '''

    cdef idx_t nInfosets = p_I.size()

    # Storage for values of infosets
    i_vals_np = np.zeros(nInfosets)
    cdef double [:] i_vals = i_vals_np
    cdef double center_sum = 0.
    cdef double future_rewards = 0.
       
    for i_id in reversed(range(nInfosets)):
        center_sum = 0.
        ci = c_I[i_id]
        for a in range(c_I[i_id].size()):
            a_id = c_I[i_id][a]
            # get value of this action

            future_rewards = 0.
            for k in range(c_A[a_id].size()):
                future_rewards += i_vals[c_A[a_id][k]]
            vals[a_id] += future_rewards
            center_sum += center[a_id]

        for a in range(c_I[i_id].size()):
            a_id = c_I[i_id][a]
            center[a_id] /= center_sum

        i_vals[i_id] = SingleStageBestResponse(
                vals = vals, 
                center = center, 
                tau = tau,
                lambd = lambd,
                i_id = i_id,
                J = J,
                c_I = c_I)

    # Removed to return behavioral strategy instead for numerical stability
    # if get_behavioral == False:
    #    BehaviorToSeqForm(vals, p_I = p_I, p_A = p_A)

cpdef inline double EvaluateObjective(P,
                    u_, v_,
                    u_beh_, v_beh_,
                    VVI& u_c_I, VVI& v_c_I, 
                    VI& u_p_I, VI& v_p_I,
                    double [:] u_lambd, double [:] v_lambd,
                    Pv=None, PTu=None):
    ''' Evaluate min-max objective (i.e. without the 
        smoothing factors, i.e. tau=infinity).
    :param P scipy sparse or numpy dense payoff.
    :param u_, v_ numpy arrays, sequence form strategies to be evaluated.
    :param u_beh_, v_beh_ numpy arrays, behavioral form representation of u_, v_
    :param u_c_I, v_c_I, u_p_I, v_p_I, infoset game structure for u,v players
    :param Pv, PTu, P \times u_ or P.T \times v_ if 
        already precomputed. Will recompute if left as None.
    :return double containing unsmoothed objective.
    '''
    cdef double uReg = 0. 
    cdef double vReg = 0.
    cdef double [:] u = u_
    cdef double [:] v = v_
    cdef double [:] u_beh = u_beh_
    cdef double [:] v_beh = v_beh_
    cdef idx_t a_id
    cdef double u_par, v_par, beh
    cdef double reg_onestage

    if Pv is not None: raw = u_.dot(Pv)
    elif PTu is not None: raw = v_.dot(PTu)
    else: raw = np.dot(u_, P.dot(v_))

    for u_i in range(u_c_I.size()):
        u_par = u[u_p_I[u_i]] if u_p_I[u_i] >= 0 else 1.
        reg_onestage = 0.
        for a in range(u_c_I[u_i].size()):
            a_id = u_c_I[u_i][a]
            beh = u_beh[a_id]
            # beh = u[a_id] / u_par
            if beh > EPSILON_ENTROPY: 
                reg_onestage += beh * log(beh)
        reg_onestage *= u_par * u_lambd[u_i]
        uReg += reg_onestage

    for v_i in range(v_c_I.size()):
        v_par = v[v_p_I[v_i]] if v_p_I[v_i] >= 0 else 1.
        reg_onestage = 0.
        for a in range(v_c_I[v_i].size()):
            a_id = v_c_I[v_i][a]
            beh = v_beh[a_id]
            # beh = v[a_id] / v_par
            if beh > EPSILON_ENTROPY: 
                reg_onestage += beh * log(beh)
        reg_onestage *= v_par * v_lambd[v_i]
        vReg += reg_onestage

    return raw + uReg - vReg


cpdef inline double DualityGap(u, v,
                    u_beh, v_beh,
                    P,
                    VVI& u_c_I, VVI& v_c_I, 
                    VI& u_p_I, VI& v_p_I,
                    VVI& u_c_A, VVI& v_c_A, 
                    VI& u_p_A, VI& v_p_A,
                    VD& u_J, VD& v_J,
                    double [:] u_lambd, double [:] v_lambd,
                    Pv=None, PTu=None):
    ''' Evaluate duality gap given a fixed u, v.
    :param u, v sequence form strategies of players 
    :param u_beh, v_beh behavioral form strategies of players 
    :param P scipy sparse of numpy dense matrices
    :param u_c_I u_c_A etc. game structure
    :param u_J, v_J. J parameters (number of child infosets per action)
    :param Pv, PTu, P \times u_ or P.T \times v_ if 
        already precomputed. Will recompute if left as None.
    :return double containing duality gap.
    '''

    if Pv is None: Pv = P.dot(v)
    if PTu is None: PTu = P.T.dot(u)
    
    cdef double upper_bound = 0.
    cdef double lower_bound = 0.
    cdef double gap = 0.

    # Best u (unsmoothed) response to v
    cdef double [:] u_vals_beh = Pv # naming is for storage purposes
    cdef double [:] u_center = np.ones(u_p_A.size())
    BestResponseTree(u_c_I, u_c_A, u_p_I, u_p_A,
                        u_J,
                        vals = u_vals_beh,
                        center = u_center,
                        tau = MAX_TAU,
                        lambd = u_lambd)
    # At this stage, u_beh contains the *BEHAVIORAL* strategies
    # We compute the sequence form representation and store it in u_vals
    u_vals = np.copy(u_vals_beh)
    BehaviorToSeqForm(u_vals, p_I = u_p_I, p_A = u_p_A)

    # Best v (unsmoothed) response to u
    cdef double [:] v_vals_beh = -np.copy(PTu) # naming is for storage purposes
    cdef double [:] v_center = np.ones(v_p_A.size())
    BestResponseTree(v_c_I, v_c_A, v_p_I, v_p_A,
                        v_J,
                        vals = v_vals_beh,
                        center = v_center,
                        tau = MAX_TAU,
                        lambd = v_lambd)
    # At this stage, v_beh contains the *BEHAVIORAL* strategies
    # We compute the sequence form representation and store it in u_vals
    v_vals = np.copy(v_vals_beh)
    BehaviorToSeqForm(v_vals, p_I = v_p_I, p_A = v_p_A)

    upper_bound = EvaluateObjective(P, u, np.array(v_vals),
                                    u_beh, np.array(v_vals_beh),
                                    u_c_I, v_c_I, u_p_I, v_p_I, u_lambd, v_lambd)
    lower_bound = EvaluateObjective(P, np.array(u_vals), v,
                                    np.array(u_vals_beh), v_beh,
                                    u_c_I, v_c_I, u_p_I, v_p_I, u_lambd, v_lambd)
    # print('%.40f, %.40f' % (upper_bound, lower_bound))

    gap = upper_bound - lower_bound

    return gap


cdef inline double SingleStageBestResponse(double [:] vals, 
                        double [:] center, 
                        double tau,
                        double [:] lambd,
                        idx_t i_id,
                        VD& J,
                        VVI& c_I):
    '''
    Best response for the normal-form 'subgame'
    from the *minimum* player's perspective.
    :param vals Pv (entire Pv vector)
    :param center entire vector containing smoothing center
    :param tau step size (larger = more ambitious)
    :param i_id, id of infoset we are evaluating.
    :param partial_J array containing no. of information sets after each action
    :param c_I full vector of vectors for children
    ! IMPT: Updates, in-place, the np array vals (originially containing Pv) 
    to contain the new values. 
    '''
    
    cdef idx_t N = c_I[i_id].size()

    # elementwise operations
    cdef double ssum = 0.
    cdef double mmax = -INFINITY
    cdef idx_t a_id

    # For numerical stability, Instead of u_k = \exp(z_k)/(\sum(\exp(z_i))),
    # we divide the numerator and denominator by mmax = \max z_i.
    # This gives \exp(z_k-mmax)/(\sum(\exp z_i - mmax)), which is unlikely
    # to give infinities and nan's from large exponents.
    for a in range(N):
        a_id = c_I[i_id][a]
        vals[a_id] = 1./lambd[i_id] * -(tau/(1.+tau)*vals[a_id] -
            1./(1.+tau) * (lambd[i_id] * (1 + log(center[a_id])) -J[a_id]))
        mmax = max(mmax, vals[a_id])

    for a in range(N):
        a_id = c_I[i_id][a] # children actions
        # print(i_id, a, vals[a_id])
        # if vals[a_id] - mmax < 10.*-10:
        #    print('Potential numerical issue - probability too low', vals[a_id]-mmax)
        vals[a_id] = exp(vals[a_id] - mmax)

        ssum += vals[a_id] # Note: sums are *after* recentering of softmax!

        # if i_id == 1120 or i_id == 2110:
        #    print('DEBUG', a_id, vals[a_id])
    
    # Normalizing to obtain probabilities 
    for a in range(N):
        a_id = c_I[i_id][a]
        if vals[a_id] == 0:
            # print(vals[a_id])
            # print(ssum)
            print('forward solver 0-warning', i_id, a, vals[a_id], ssum, lambd[i_id], mmax)
        vals[a_id] /= ssum

        # if i_id == 1120 or i_id == 2110:
        #   print('DEBUG', a_id, vals[a_id])
    
    cdef double objective = lambd[i_id] * -((mmax + log(ssum)) * (1. + tau) / tau)
    # print(i_id, objective)

    return objective


cdef void BehaviorToSeqForm(double [:] strat, VI& p_I, VI& p_A):
    '''
    Convert behavioral strategy to sequence form by traversing tree.
    :param strat long vector (for all actions) of normalized actions containing
    behavioral strategy.
    :param p_I, p_A vectors containing parent id's for infosets and actions.
    :return sequence form representation with strat
    ! WARNING Assumes that actions are indexed in topolgical order.
    ! WARNING strat will be updated in place to contain the sequence form
    '''

    cdef idx_t nActions = p_A.size()
    for a_id in range(nActions):
        if p_A[a_id] < 0: # No parent infoset
            continue
        parpara = p_I[p_A[a_id]] # parent of parent action
        if parpara < 0: # No parent action
            continue
        strat[a_id] *= strat[parpara]


cdef np.ndarray SeqToBehaviorForm(double [:] strat, VI& p_I, VI& p_A):
    '''
    Convert sequence form strategy to behavioral form by traversing tree.
    :param strat long vector (for all actions) of normalized actions containing
    sequence form strategy.
    :param p_I, p_A vectors containing parent id's for infosets and actions.
    :return behavioral form representation with strat
    '''

    ret = np.zeros(p_A.size())
    cdef idx_t nActions = p_A.size()
    for a_id in range(nActions):
        pari = p_A[a_id]
        parpara = p_I[pari] # parent of parent action
        if parpara < 0: # No parent action
            ret[a_id] = strat[a_id]
        else:
            ret[a_id] = strat[a_id] / strat[parpara]
    return ret


#################################################################################

cdef VD ComputeJ(VVI& c_A, double[:] lambd):
    cdef VD ret
    cdef double l
    cdef idx_t i_id
    for a in range(c_A.size()):
        l = 0.
        for idx in range(c_A[a].size()):
            i_id = c_A[a][idx]
            l += lambd[i_id]
        ret.push_back(l)

    return ret

cdef ConstructE(VVI& c_I, VI& p_I, VVI& c_A, VI& p_A):
    '''
    '''
    cdef idx_t a_id, i_id
    cdef idx_t nActions = c_A.size()
    cdef idx_t nInfosets = c_I.size()
    r, c, d = [], [], []
    for i_id in range(nInfosets):
        # parent
        a_id = p_I[i_id]
        if a_id >= 0: 
            r.append(i_id)
            c.append(a_id)
            d.append(-1)

        # child
        for a_idx in range(c_I[i_id].size()):
            a_id = c_I[i_id][a_idx]
            r.append(i_id)
            c.append(a_id)
            d.append(1)

    return scipy.sparse.coo_matrix((d,(r,c)), shape=(nInfosets, nActions))

cdef void SetVerbosity(int verbosity):
    glob_verbosity = verbosity
    log_msg('Set verbosity to %d' % verbosity, LOGGER_WARNING)

cdef inline void log_msg(msg, int level):
     if level >= glob_verbosity:
         logging.log(level, msg)
