#i distutils: language=c++
cimport numpy as np
from .util cimport TransformLL, TransformL, idx_t, VVI, VI, VD, pywrap_GameStructure
import numpy as np
cimport cython
from libc.math cimport exp, log
from libcpp.vector cimport vector
import scipy.sparse
import logging

cdef double MAX_TAU = 10.**50

LOGGER_DEBUG = 10
LOGGER_INFO = 20
LOGGER_WARNING = 30
cdef int glob_verbosity = LOGGER_DEBUG

def backwardSolve(P_,
        u, v,
        uv_c_I, uv_c_A, uv_p_I, uv_p_A,
        tau, 
        lossu, lossv,
        lambd_=None,
        tol=10.**-10, max_iters=1000,
        Ptype='keep', verify=False,
        prep_struct=None,
        u_beh=None, v_beh=None,
        use_shortcut=False,
        duality_freq=10):
    '''
    :param P_ sparse scipy or dense numpy payoff matrix
    :param u, v sequence form strategies obtained from forward pass
    :param uv_c_I 2-tuple of children list for infosets for each player
    :param uv_c_A 2-tuple of children list for actions for each plyer
    :param uv_p_I 2-tuple of parents for infosets for each player
    :param uv_p_A 2-tuple of parents for actions for each player
    :param tau step size, larger implies= faster convergence
    :param lossu, lossv *gradients* of u, v with respect to loss function
    :param lambd np ndarray of rationality parameters for each infoset
    :param tol termination criterion, duality gap
    :param max_iters, maximum number of iterations before terminating
    :param Ptype \in {convert, keep}. 
        - use convert to convert P_ to sparse matrix before proceeding
        - use keep to keep P_ in whatever format it was before
    :param verify Checks if solution is correct against exact solver
    :param prep_struct
        - pywrap_GameStructure which wraps around a c structure storing
          the precomputed game structure
    :param u_beh, v_beh behavioral strategies obtained from forward pass.
        - If u_beh or v_beh is None, then compute it on the fly
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
    # Compute u_beh, v_beh if it was not provided. Note
    # that there may be numerical problems here.
    #########################################################
    if u_beh is None:
        u_beh = SeqToBehaviorForm(u, u_p_I[0], u_p_A[0])

    if v_beh is None:
        v_beh = SeqToBehaviorForm(v, v_p_I[0], v_p_A[0])


    #########################################################
    #
    # Special case: when P is a vector (i.e. one player game)
    # Note: When applying this, the quadratic term xPy is the 0 vector, since
    # either x or y will be 0.
    # 
    #########################################################

    if P.shape[0] == 1: # u is the 'unimportant player' with 1 action
        ret_v = BestResponse(v_c_I[0], v_c_A[0], v_p_I[0], v_p_A[0],
                        v, v_beh,
                        vals = np.zeros(P.shape[1]), # vals = np.atleast_1d(np.squeeze(P)).astype(np.float64),
                        loss=-lossv,
                        center = np.zeros(vsize), 
                        tau = MAX_TAU,
                        lambd = v_lambd,
                        J = v_J,
                        use_shortcut=use_shortcut)
        '''
        vInverse = VerifyInverse(P, np.array([0.]), ret_v, 
                    u_c_I[0], v_c_I[0], 
                    u_p_I[0], v_p_I[0],
                    u_c_A[0], v_c_A[0], 
                    u_p_A[0], v_p_A[0],
                    u, v,
                    lossu, lossv,
                    u_lambd, v_lambd)
        print('Inverse difference:', vInverse)
        assert False, 'haha'
        '''
        return np.array([0.]), ret_v
    elif P.shape[1] == 1: # v is the 'unimportant player'
        ret_u = BestResponse(u_c_I[0], u_c_A[0], u_p_I[0], u_p_A[0], 
                        u, u_beh,
                        vals = np.zeros(P.shape[0]), # vals = np.atleast_1d(np.squeeze(P)).astype(np.float64),
                        loss= lossu, 
                        center = np.zeros(usize), 
                        tau = MAX_TAU,
                        lambd = u_lambd,
                        J = u_J,
                        use_shortcut=use_shortcut)
        return ret_u, np.array([0])

    duality_violated_tol = np.sqrt(tol)

    ########################################################
    # 
    # Initialization using best response with 
    # no centering, no P (i.e. only regularization)
    # 
    ########################################################
    x_init = BestResponse(u_c_I[0], u_c_A[0], u_p_I[0], u_p_A[0], 
                        u, u_beh,
                        vals = np.zeros(usize),
                        loss= lossu, 
                        center = np.zeros(usize), 
                        tau = MAX_TAU,
                        lambd = u_lambd,
                        J = u_J,
                        use_shortcut = use_shortcut)
    
    y_init = BestResponse(v_c_I[0], v_c_A[0], v_p_I[0], v_p_A[0], 
                        v, v_beh,
                        vals = np.zeros(vsize),
                        loss=-lossv,
                        center = np.zeros(vsize), 
                        tau = MAX_TAU,
                        lambd = v_lambd,
                        J = v_J,
                        use_shortcut = use_shortcut)

    ########################################################
    # 
    # Solve using smoothed iterations (Chambolle and Pock, 2014)
    # 
    #########################################################
    
    x, y = x_init, y_init
    # print(u)
    # print(v)
    # print(u_beh)
    # print(np.max(v_beh[:30]))
    for it in range(max_iters):
        # print(it)
        # PD part 1 (update x^{n+1})
        x_vals = P.dot(y)
        x_cent = np.copy(x)
        x_next = BestResponse(u_c_I[0], u_c_A[0], u_p_I[0], u_p_A[0],
                                u, u_beh,
                                vals = x_vals,
                                loss = lossu,
                                center = x,
                                tau = tau,
                                lambd = u_lambd,
                                J = u_J,
                                use_shortcut = use_shortcut)

        # PD part 2 (update y^{n+1})
        y_vals = -PT.dot(2. * x_next - x)
        y_next = BestResponse(v_c_I[0], v_c_A[0], v_p_I[0], v_p_A[0],
                                v, v_beh,
                                vals = y_vals,
                                loss = -lossv,
                                center = y,
                                tau = tau,
                                lambd = v_lambd,
                                J = v_J,
                                use_shortcut = use_shortcut)
        # print(x, y) 
        x, y = x_next, y_next

        if it % duality_freq == 0:
            gap = DualityGap(P, x, y,
                    u_c_I[0], v_c_I[0], 
                    u_p_I[0], v_p_I[0],
                    u_c_A[0], v_c_A[0], 
                    u_p_A[0], v_p_A[0],
                    u, u_beh,
                    v, v_beh,
                    lossu, lossv,
                    u_lambd, v_lambd,
                    u_J, v_J,
                    Py=None, PTx=None)
            log_msg('Backward Duality gap:', LOGGER_DEBUG)
            # print(it, 'gap:', gap)

            if gap < tol: break

    if gap > tol or gap < -duality_violated_tol: 
        log_msg('Termination criterion not met (backward)! Gap: %s' % gap, LOGGER_WARNING)
    if verify != 'none' and verify is not None and verify is not False:
        """
        return SolveExact(P, x, y, 
                    u_c_I, v_c_I, 
                    u_p_I, v_p_I,
                    u_c_A, v_c_A, 
                    u_p_A, v_p_A,
                    u, v,
                    lossu, lossv,
                    u_lambd, v_lambd)
        """
        vInverse = VerifyInverse(P, x, y,
                    u_c_I[0], v_c_I[0], 
                    u_p_I[0], v_p_I[0],
                    u_c_A[0], v_c_A[0], 
                    u_p_A[0], v_p_A[0],
                    u, v,
                    lossu, lossv,
                    u_lambd, v_lambd)
        # print(vInverse)

        if np.abs(vInverse) > np.sqrt(tol):
            log_msg('Termination criterion not met (backward)! MSE: %s' % np.abs(vInverse), LOGGER_WARNING)

    return x, y


cdef np.ndarray BestResponse(VVI& c_I, VVI& c_A, 
            VI& p_I, VI& p_A,
            np.ndarray u,
            np.ndarray u_beh,
            np.ndarray vals, 
            np.ndarray loss, 
            np.ndarray center, 
            double tau,
            double [:] lambd,
            VD& J,
            use_shortcut=False):
    '''
    Solve for the best response of a single-player game tree.
    :param c_I, c_A, p_I, p_A children and parents of infosets, actions
    :param u sequence form strategy of player
    :param u_beh behavioral strategy of player
    :param vals immediate payoff for each action (i.e. Py if min player)
    :param center centering of Bregman divergence
    :param tau step size
    :param lambd rationality parameters
    :return numpy array containing sequence form
    ! WARNING Assumes that actions and infosets are ordered in topolgical order!
    ! WARNING Modifies vals and center in-place.
    '''

    cdef np.ndarray c
    if use_shortcut:
        Xi_center = Xi_x(c_I, c_A, p_I, p_A, u, u_beh, center, lambd, J, use_shortcut=True)

        # cdef np.ndarray c = tau / (1.+tau) * (vals + loss) - \
        #                     (1./ ( 1. + tau )) * Xi_center
        c = tau / (1.+tau) * (vals + loss) * u - \
                        (1./ ( 1. + tau )) * Xi_center # [Note the multiplication of weighting factor u]
        # print('c', np.linalg.norm(c))

        solution = SolveQuadratic(c_I, c_A, p_I, p_A, u, u_beh, c, lambd, use_shortcut=True)

        # print(max(solution), min(solution), min(c), max(c))
    elif not use_shortcut:
        # POTENTIAL NUMERICAL ISSUE HERE if u has near zero entries!
        Xi_center = Xi_x(c_I, c_A, p_I, p_A, u, u_beh, center, lambd, J, use_shortcut=False)

        # cdef np.ndarray c = tau / (1.+tau) * (vals + loss) - \
        #                     (1./ ( 1. + tau )) * Xi_center
        c = tau / (1.+tau) * (vals + loss) - \
                            (1./ ( 1. + tau )) * Xi_center # [Note the non-multiplication of weighting factor u]
        # print('c', np.linalg.norm(c))

        solution = SolveQuadratic(c_I, c_A, p_I, p_A, u, u_beh, c, lambd, use_shortcut=False)
    return solution


cdef np.ndarray SolveQuadratic(VVI& c_I, VVI& c_A, 
            VI& p_I, VI& p_A,
            np.ndarray u,
            np.ndarray u_beh,
            np.ndarray c,
            double [:] lambd,
            use_shortcut=False):
    ''' Main helper function (oracle for optimizing quadratic). Assume min-player.
    *** Assume an extra factor of diag(1/u) *** 
    :param c_I, c_A, p_I, p_A children and parents of infosets and actions
    :param u player's sequence form strategy
    :param u_beh player's behavioral strategy
    :param c linear term in the quadratic term (constant after differentiating)
    :param use_shortcut 
        - True if we want to use behavioral strategies for numerical stability, instead of sequence form
            If so, then the parameter c *must* have been prev-multiplied by a factor of diag(u)
        - False if we want to use the sequence form, may be numerically unstable!
    '''

    if use_shortcut:
        l_mult = -np.squeeze((E_Xi_inv_c(c_I, c_A, p_I, p_A, u, c, lambd, use_shortcut=True)))/E_Xi_inv_ET(p_I, u, lambd)

        # [At this point, l_mult typically has an additional factor of diag(u), usually from a prior call
        # to Xi_x]
        coeff = -c - ET_x(c_I, p_I, u.size, l_mult) * u # Note the extra factor of [*u] at the end.

        # [Implicit in Xi_inv_x is an additional factor of diag(1/u), which typically cancels the extra factor from Xi_x.]
        solution = Xi_inv_x(c_I, c_A, p_A, u, u_beh, coeff, lambd, use_shortcut=True)

    elif not use_shortcut:
        l_mult = -np.squeeze((E_Xi_inv_c(c_I, c_A, p_I, p_A, u, c, lambd, use_shortcut=False)))/E_Xi_inv_ET(p_I, u, lambd)

        coeff = -c - ET_x(c_I, p_I, u.size, l_mult)# Note the extra factor of [*u] at the end.

        solution = Xi_inv_x(c_I, c_A, p_A, u, u_beh, coeff, lambd, use_shortcut=False)
        # test = Xi_inv_x_VERIFY(c_I, c_A, p_I, p_A, u, coeff, lambd)
        # print(test)
    '''
    #  Uncomment for debugging!
    # ONLY works for use_shortcut == False.
    print('****************** DEBUGGING L_MULT **************')
    print('====Testing E_Xi_inv_c====')
    anal = E_Xi_inv_c(c_I, c_A, p_I, p_A, u, c, lambd)
    print('Analytical:', anal)
    Xi = ConstructXi(c_I, p_I, c_A, p_A, u, lambd)
    E = ConstructE(c_I, p_I, c_A, p_A)
    Z = scipy.sparse.linalg.spsolve(Xi, c)
    Z = E.dot(Z)
    print('Should be 0:', scipy.linalg.norm(Z-anal))

    print('====Testing E_Xi_inv_ET====')
    anal = E_Xi_inv_ET(p_I, u, lambd)
    print('Analytical:', anal)
    Xi_inv_ET = scipy.sparse.linalg.spsolve(Xi, E.T)
    Z = E.dot(Xi_inv_ET)
    Z = Z.diagonal()
    print('Should be 0:', scipy.linalg.norm(Z-anal))


    # l_mult = -(E_Xi_inv_c(c_I, c_A, p_I, p_A, u, c, lambd))/E_Xi_inv_ET(p_I, u, lambd)
    print('=====Testing ET_x ========')
    anal = ET_x(c_I, p_I, u.size, l_mult)
    print('Analytical:', anal)
    Z = E.T.dot(l_mult)
    print('Should be 0:', scipy.linalg.norm(Z-anal))

    print('=====Testing Xi_inv_x =====')
    anal = solution
    print('Analytical:', anal)
    Z = scipy.sparse.linalg.spsolve(Xi, coeff)
    print('Should be 0:', scipy.linalg.norm(Z-anal))
    '''
    
    # return test
    return solution
    # return Z


cdef double DualityGap(P, np.ndarray x, np.ndarray y, 
                    VVI& u_c_I, VVI& v_c_I, 
                    VI& u_p_I, VI& v_p_I,
                    VVI& u_c_A, VVI& v_c_A, 
                    VI& u_p_A, VI& v_p_A,
                    np.ndarray u, np.ndarray u_beh,
                    np.ndarray v, np.ndarray v_beh,
                    np.ndarray lossu, np.ndarray lossv,
                    double [:] u_lambd, double [:] v_lambd,
                    VD& u_J, VD& v_J,
                    Py=None, PTx=None):
    ''' Evaluate duality gap given a fixed u, v.
    TODO: how to compute gap without explicitly computing the quadratic terms? 
    (which may be numerically unstable due to the 1/u terms)
    
    :param P scipy sparse of numpy dense matrices
    :param x, y numpy array containing answers provided by solver
    :param u_c_I u_c_A etc. game structure
        already precomputed. Will recompute if left as None.
    :param u, v numpy array containing strategies played by players 
    :param lossu, lossv *gradients* of u, v with respect to loss function
    :param Pv, PTu, P \times u_ or P.T \times v_ if 
    :return double containing duality gap.
    '''

    if Py is None: Py = P.dot(y)
    if PTx is None: PTx = P.T.dot(x)

    # Best x (unsmoothed) response to y
    x_br = BestResponse(u_c_I, u_c_A, u_p_I, u_p_A,
                        u, u_beh,
                        vals = Py,
                        loss = lossu,
                        center = x,
                        tau = MAX_TAU,
                        lambd = u_lambd,
                        J = u_J)

    # Best y (unsmoothed) response to x
    y_br = BestResponse(v_c_I, v_c_A, v_p_I, v_p_A,
                        v, v_beh,
                        vals = -PTx,
                        loss = -lossv,
                        center = y,
                        tau = MAX_TAU,
                        lambd = v_lambd,
                        J = v_J)
    """
    print('obj', EvaluateObjective(P, x, y_br,
                            u_c_I, v_c_I,
                            u_p_I, v_p_I,
                            u_c_A, v_c_A,
                            u_p_A, v_p_A,
                            u, v,
                            u_beh, v_beh,
                            lossu, lossv,
                            u_lambd, v_lambd,
                            u_J, v_J))
    print('obj', EvaluateObjective(P, x_br, y,
                            u_c_I, v_c_I,
                            u_p_I, v_p_I,
                            u_c_A, v_c_A,
                            u_p_A, v_p_A,
                            u, v,
                            u_beh, v_beh,
                            lossu, lossv,
                            u_lambd, v_lambd,
                            u_J, v_J))
    """

    z1 = x.T.dot(P.dot(y_br)) - x_br.dot(P.dot(y))
    z2 = lossu.T.dot(x) - lossu.T.dot(x_br)
    z3 = lossv.T.dot(y_br) - lossv.T.dot(y)
    z4 = (x-x_br).T.dot(Xi_x(u_c_I, u_c_A, u_p_I, u_p_A, u, u_beh, x+x_br, u_lambd, u_J))
    z5 = (y-y_br).T.dot(Xi_x(v_c_I, v_c_A, v_p_I, v_p_A, v, v_beh, y+y_br, v_lambd, v_J))

    return z1+z2+z3+0.5*z4+0.5*z5

    ############### OLD WITH NUMERICAL BADNESS #######
    gap = EvaluateObjective(P, x, y_br,
                    u_c_I, v_c_I, 
                    u_p_I, v_p_I,
                    u_c_A, v_c_A, 
                    u_p_A, v_p_A,
                    u, v,
                    u_beh, v_beh,
                    lossu, lossv,
                    u_lambd, v_lambd,
                    u_J, v_J) - \
        EvaluateObjective(P, x_br, y,
                    u_c_I, v_c_I, 
                    u_p_I, v_p_I,
                    u_c_A, v_c_A, 
                    u_p_A, v_p_A,
                    u, v,
                    u_beh, v_beh,
                    lossu, lossv,
                    u_lambd, v_lambd,
                    u_J, v_J)
    return gap


cdef double EvaluateObjective(P, np.ndarray x, np.ndarray y, 
                    VVI& u_c_I, VVI& v_c_I, 
                    VI& u_p_I, VI& v_p_I,
                    VVI& u_c_A, VVI& v_c_A, 
                    VI& u_p_A, VI& v_p_A,
                    np.ndarray u, np.ndarray v,
                    np.ndarray u_beh, np.ndarray v_beh,
                    np.ndarray lossu, np.ndarray lossv,
                    double [:] u_lambd, double [:] v_lambd,
                    VD& u_J, VD& v_J):
    ''' Evaluate min-max objective (i.e. without the 
        smoothing factors, i.e. tau=infinity).
    :param P scipy sparse or numpy dense payoff.
    :param x, y numpy arrays, strategies to be evaluated.
    :param u_c_I, v_c_I, u_p_I, v_p_I, infoset game structure for u,v players
    :param u, v numpy ndarray sequence form strategy of min, max player
    :param u_beh, v_beh numpy ndarray behavioral strategy of min, max player
    :param lossu, lossv 
    :return double containing unsmoothed objective.
    '''

    '''
    print  x.T.dot(P.dot(y)) +\
        lossu.T.dot(x) + lossv.T.dot(y) +\
        0.5 * x.T.dot(Xi_x(u_c_I, u_c_A, u_p_I, u_p_A, u, x, u_lambd, u_J)) -\
        0.5 * y.T.dot(Xi_x(v_c_I, v_c_A, v_p_I, v_p_A, v, y, v_lambd, v_J))
    print(x)
    print(np.max(y), np.min(y))
    print(np.max(u), np.min(u))
    print(np.max(v), np.min(v))
    input()
    '''
    print('1')
    print(x.T.dot(P.dot(y)))
    print('2')
    print(lossu.T.dot(x) + lossv.T.dot(y))
    print('3')
    print(0.5 * x.T.dot(Xi_x(u_c_I, u_c_A, u_p_I, u_p_A, u, u_beh, x, u_lambd, u_J)) )
    print('4')
    print(0.5 * y.T.dot(Xi_x(v_c_I, v_c_A, v_p_I, v_p_A, v, v_beh, y, v_lambd, v_J)))
    return x.T.dot(P.dot(y)) +\
        lossu.T.dot(x) + lossv.T.dot(y) +\
        0.5 * x.T.dot(Xi_x(u_c_I, u_c_A, u_p_I, u_p_A, u, u_beh, x, u_lambd, u_J)) -\
        0.5 * y.T.dot(Xi_x(v_c_I, v_c_A, v_p_I, v_p_A, v, v_beh, y, v_lambd, v_J))


cdef double VerifyInverse(P, np.ndarray x, np.ndarray y, 
                    VVI& u_c_I, VVI& v_c_I, 
                    VI& u_p_I, VI& v_p_I,
                    VVI& u_c_A, VVI& v_c_A, 
                    VI& u_p_A, VI& v_p_A,
                    np.ndarray u, np.ndarray v,
                    np.ndarray lossu, np.ndarray lossv,
                    double [:] u_lambd, double [:] v_lambd):

    x_ideal, y_ideal = SolveExact(P, x, y, 
                    u_c_I, v_c_I, 
                    u_p_I, v_p_I,
                    u_c_A, v_c_A, 
                    u_p_A, v_p_A,
                    u, v,
                    lossu, lossv,
                    u_lambd, v_lambd)

    return np.sum(x-x_ideal)**2 + np.sum(y-y_ideal)**2


cdef SolveExact(P, np.ndarray x, np.ndarray y, 
                    VVI& u_c_I, VVI& v_c_I, 
                    VI& u_p_I, VI& v_p_I,
                    VVI& u_c_A, VVI& v_c_A, 
                    VI& u_p_A, VI& v_p_A,
                    np.ndarray u, np.ndarray v,
                    np.ndarray lossu, np.ndarray lossv,
                    double [:] u_lambd, double [:] v_lambd):

    # Construct giant matrix to be inverted.
    E_u = ConstructE(u_c_I, u_p_I, u_c_A, u_p_A)
    E_v = ConstructE(v_c_I, v_p_I, v_c_A, v_p_A)
    Xi_u = ConstructXi(u_c_I, u_p_I, u_c_A, u_p_A, u, u_lambd)
    Xi_v = ConstructXi(v_c_I, v_p_I, v_c_A, v_p_A, v, v_lambd)

    mat = scipy.sparse.bmat([[Xi_u, P, E_u.T, None], 
                    [P.T, -Xi_v, None, E_v.T], 
                    [E_u, None, None, None], 
                    [None, E_v, None, None]])

    L = -np.concatenate([lossu, lossv, np.zeros(u_c_I.size()+v_c_I.size())])
    ideal = scipy.sparse.linalg.spsolve(mat, L)
    x_ideal = ideal[0:lossu.size]
    y_ideal = ideal[lossu.size:(lossu.size+lossv.size)]

    '''
    if (np.any(scipy.sparse.linalg.eigs(Xi_u)[0] < 0) or\
            np.any(scipy.sparse.linalg.eigs(Xi_v)[0] < 0)):
        log_msg('Xi not PSD! Eigenvalues not positive.', LOGGER_WARNING)
    '''

    return x_ideal, y_ideal


cdef ConstructXi(VVI& c_I, VI& p_I, 
                VVI& c_A, VI& p_A,
                np.ndarray u,
                double [:] lambd):
    '''
    Construct (potentially) sparse Xi.
    '''

    cdef idx_t a_id, i_id, 
    cdef idx_t nActions = c_A.size()
    cdef idx_t nInfosets = c_I.size()
    sI = [0.] * nActions
    r, c, d = [], [], []

    for a_id in range(nActions):
        # child
        for i_idx in range(c_A[a_id].size()):
            i_id = c_A[a_id][i_idx]
            sI[a_id] += lambd[i_id]
            for child_a_idx in range(c_I[i_id].size()):
                child_a_id = c_I[i_id][child_a_idx]
                r.append(a_id)
                c.append(child_a_id)
                d.append(-lambd[i_id]/u[a_id])
        # self
        r.append(a_id)
        c.append(a_id)
        d.append((lambd[p_A[a_id]] + sI[a_id])/u[a_id])

        # parent
        p_i_id = p_A[a_id]
        if p_I[p_i_id] < 0: continue
        para_a_id = p_I[p_i_id]
        r.append(a_id)
        c.append(para_a_id)
        d.append(-lambd[p_i_id]/u[para_a_id])

    mat = scipy.sparse.coo_matrix((d,(r,c)), shape=(nActions, nActions))

    return mat


##################################################################################
# 
# Helper functions
# 
##################################################################################


cdef np.ndarray Xi_x(VVI& c_I, VVI& c_A, 
        VI& p_I, VI& p_A,
        np.ndarray u_,
        np.ndarray u_beh_,
        np.ndarray x_,
        double [:] lambd,
        VD& J,
        use_shortcut=False):
    '''
    Compute Xi \times x (sparse-matrix multiplication-ish) ***MULTIPLIED***
    by diag(u_) [added for the purposes of numerical-stability]
    
    :param u_ behavioral strategy
    :param u_beh_ behavioral strategy
    '''
    
    cdef idx_t nActions = c_A.size()
    cdef double [:] x = x_
    cdef double [:] u = u_
    cdef double [:] u_beh = u_beh_
    ret_ = np.zeros(nActions)
    cdef double [:] ret = ret_
    cdef idx_t i_par, parpar_u_id
    cdef idx_t child_i_id, childchild_u_id

    for u_id in range(nActions):
        # check parent
        i_par = p_A[u_id]
        if p_I[i_par] >= 0:
            parpar_u_id = p_I[i_par]
            if not use_shortcut:
                ret[u_id] -= x[parpar_u_id] * lambd[i_par] / u[parpar_u_id] # Without factor of diag(u)
            elif use_shortcut:
                ret[u_id] -= x[parpar_u_id] * lambd[i_par] * u_beh[u_id]

        # check child
        for c_i_id in range(c_A[u_id].size()):
            child_i_id = c_A[u_id][c_i_id]
            for cc_i_id in range(c_I[child_i_id].size()):
                childchild_u_id = c_I[child_i_id][cc_i_id]
                if not use_shortcut:
                    # print('moo')
                    ret[u_id] -= x[childchild_u_id] * lambd[child_i_id] / u[u_id] # Without factor of diag(u)
                elif use_shortcut:
                    ret[u_id] -= x[childchild_u_id] * lambd[child_i_id]
        
        # Get diagonal
        if not use_shortcut:
            # print('foo')
            # print(u.shape)
            # print(u_id, nActions)
            # print(u[u_id])
            ret[u_id] += x[u_id] * (lambd[i_par] + J[u_id]) / u[u_id] # Without factor of diag(u)
        elif use_shortcut:
            ret[u_id] += x[u_id] * (lambd[i_par] + J[u_id])
 
    return ret_


cdef np.ndarray E_Xi_inv_c(VVI& c_I, VVI& c_A, 
        VI& p_I, VI& p_A, 
        np.ndarray u_, np.ndarray c_, 
        double [:] lambd,
        use_shortcut=False):
    '''
    Evaluate E * Xi^{-1}(u) c for some c by bottom up traversal of the tree.
    :param use_shortcut
        - If True, then we will assume c has an extra factor of diag(u) and we are
            actually interested in computing E_Xi_inv diag(1/u) c
    WARNING: Assumes that information sets are indexed in topological ordering.
    '''

    cdef double [:] u = u_
    cdef double [:] c = c_
    store_ = np.zeros(c_A.size())
    ret_ = np.zeros(c_I.size())
    cdef double [:] store = store_
    cdef double [:] ret = ret_
    cdef double total
    cdef idx_t u_id

    for i_id in reversed(range(c_I.size())):
        total = 0.
        
        # Sum up from child
        for id in range(c_I[i_id].size()):
            u_id = c_I[i_id][id]

            if not use_shortcut:
                store[u_id] += c[u_id] * u[u_id]
            elif use_shortcut:
                store[u_id] += c[u_id]
            total += store[u_id]
        
        # update parent action
        if p_I[i_id] is not None:
            store[p_I[i_id]] += total
        ret[i_id] = total / lambd[i_id]
        
    return ret_


cdef np.ndarray E_Xi_inv_ET(VI& p_I, np.ndarray u_, double [:] lambd):
    ''' Compute diag(E \times Xi^{-1} \times E^T)
    :param u_ strategy of player (either max or min)
    '''
    cdef double [:] u = u_
    ret_ = np.zeros(p_I.size())
    cdef double [:] ret = ret_

    for i_id in range(p_I.size()):
        if p_I[i_id] < 0: 
            ret[i_id] = 1. / lambd[i_id] # maybe not this!!!
        else: 
            ret[i_id] = u[p_I[i_id]] / lambd[i_id]

    return ret_


cdef np.ndarray ET_x(VVI& c_I, VI& p_I, idx_t u_len, np.ndarray x_):
    ''' Compute E^T \times lambda.
    :param c_I, p_I 
    :param u_len size of action space
    :param lambd_ dual variables
    '''
    ret_ = np.zeros(u_len)
    cdef double [:] ret = ret_
    cdef double [:] x = x_
    cdef idx_t u_id

    for i_id in range(c_I.size()): 
        for id in range(c_I[i_id].size()):
            u_id = c_I[i_id][id]
            ret[u_id] += x[i_id]
        if p_I[i_id] >= 0:
            ret[p_I[i_id]] -= x[i_id]
    return ret_

cdef np.ndarray Xi_inv_x_VERIFY(VVI& c_I, VVI& c_A, VI& p_I, VI& p_A,
        np.ndarray u_, np.ndarray x_, double [:] lambd):
    '''
    For testing purposes. 
    [Note that this does *NOT* include the correction factor of diag(1/u)
    '''
    ret_ = np.atleast_1d(np.squeeze(np.copy(x_)))
    ret_ *= np.squeeze(u_)
    cdef double [:] ret = ret_
    cdef double [:] u = u_
    cdef double [:] x = x_
    cdef idx_t i_id, u_next_id
    
    Xi = ConstructXi(c_I, p_I, c_A, p_A, u_, lambd).todense()

    # Traverse tree to get lower triangular matrix (gaussian elimination)
    # Add actions bottom up
    for u_id in reversed(range(c_A.size())):
        for id in range(c_A[u_id].size()):
            i_id = c_A[u_id][id]
            for id2 in range(c_I[i_id].size()):
                u_next_id = c_I[i_id][id2]
                # ret[u_id] += ret[u_next_id]
                ret[u_id] += ret[u_next_id] * u[u_next_id] / u[u_id]
                
                Xi[u_id, :] += Xi[u_next_id, :] * u[u_next_id] / u[u_id]
    
    '''
    print('one direction')
    print(Xi)
    Xi_upper = np.triu(Xi)
    Xi_upper -= np.diag(np.diag(Xi_upper))
    print(Xi_upper)
    print(np.sum(np.abs(Xi_upper)))
    '''

    # Eliminate to obtain identity matrix, top down
    for u_id in range(c_A.size()):
        for id in range(c_A[u_id].size()):
            i_id = c_A[u_id][id]
            for id2 in range(c_I[i_id].size()):
                u_next_id = c_I[i_id][id2]
                # print(ret[u_id], u[u_next_id], u[u_id])
                # print(ret[u_id] * (u[u_next_id] / u[u_id]))
                # ret[u_next_id] += ret[u_id] * (u[u_next_id] / u[u_id])
                ret[u_next_id] += ret[u_id] / lambd[p_A[u_id]] * lambd[i_id]
                # ret[u_next_id] += ret[u_id]
                Xi[u_next_id, :] += Xi[u_id, :] / lambd[p_A[u_id]] * lambd[i_id]

        # ret[u_id] /= lambd[p_A[u_id]]
        
        
        # PUT THIS BACK
        ret[u_id] *= u[u_id] / lambd[p_A[u_id]]
        Xi[u_id, :] *= u[u_id] / lambd[p_A[u_id]]
    
    '''
    print('BAD!')
    Xi_lower = np.tril(Xi)
    Xi_lower -= np.diag(np.diag(Xi_lower))
    print(np.sum(np.abs(Xi_lower)))
    print('Diagz', np.sum(np.abs(np.diag(Xi) - np.ones(Xi.shape[0]))))
    '''

    print(np.sum(np.abs(Xi-np.eye(Xi.shape[0]))))

    print('rrefed xi')
    print(Xi)

    return ret_

    return Xi


cdef np.ndarray Xi_inv_x(VVI& c_I, VVI& c_A, VI& p_A,
        np.ndarray u_,
        np.ndarray u_beh_,
        np.ndarray x_,
        double [:] lambd,
        use_shortcut=False):
    '''
    :param c_I, c_A
    :param u_ sequence form strategy of player
    :param u_beh_ behavioral strategy of player in the forward pass
    :param x_ numpy ndarray to be multiplied
    :param use_shortcut
        - True if x has an extra factor of diag(u) premultiplied. That is, we are computing Xi_inv diag(1/u) x
        - False if we are just interested in computing Xi_inv_x
    
    :return Xi_inv * x, multiplied by diag(1/u) 
    [The extra factor of diag(1/u) cancels out with the 
    other diag(u) factor which was added in the Xi_x call] 
    WARNING Assume that actions are indexed in topological ordering
    '''
    ret_ = np.atleast_1d(np.squeeze(np.copy(x_)))
    cdef double [:] ret = ret_
    cdef double [:] u = u_
    cdef double [:] u_beh = u_beh_
    cdef double [:] x = x_
    cdef idx_t i_id, u_next_id
    
    # Traverse tree to get lower triangular matrix (gaussian elimination)
    # Add actions bottom up
    for u_id in reversed(range(c_A.size())):
        for id in range(c_A[u_id].size()):
            i_id = c_A[u_id][id]
            for id2 in range(c_I[i_id].size()):
                u_next_id = c_I[i_id][id2]
                if not use_shortcut:
                    ret[u_id] += ret[u_next_id] * u_beh[u_next_id]
                elif use_shortcut:
                    ret[u_id] += ret[u_next_id]

    # Eliminate to obtain identity matrix, top down
    for u_id in range(c_A.size()):
        for id in range(c_A[u_id].size()):
            i_id = c_A[u_id][id]
            for id2 in range(c_I[i_id].size()):
                u_next_id = c_I[i_id][id2]
                if not use_shortcut:
                    ret[u_next_id] += ret[u_id] / lambd[p_A[u_id]] * lambd[i_id]
                elif use_shortcut:
                    ret[u_next_id] += ret[u_id] * u_beh[u_next_id]/ lambd[p_A[u_id]] * lambd[i_id]

        if not use_shortcut:
            # Without the correction of diag(1/u)
            ret[u_id] *= u[u_id] / lambd[p_A[u_id]]
        elif use_shortcut:
            # With the correction factor of diag(u)
            ret[u_id] *= 1. / lambd[p_A[u_id]]

    return ret_


def gradLambda(u_, v_,
        uv_c_I, uv_c_A, uv_p_I, uv_p_A, 
        du_, dv_,
        prep_struct=None,
        u_beh=None, v_beh=None,
        use_shortcut = False):

    cdef double[:] u = u_
    cdef double[:] v = v_
    cdef double[:] du = du_
    cdef double[:] dv = dv_

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
    
    cdef int ui_id, vi_id
    cdef double z, para_prob
    cdef int a_id
    dlambdu, dlambdv = [], []

    #########################################################
    # Compute u_beh, v_beh if it was not provided. Note
    # that there may be numerical problems here.
    #########################################################
    if u_beh is None:
        u_beh = SeqToBehaviorForm(u, u_p_I[0], u_p_A[0])

    if v_beh is None:
        v_beh = SeqToBehaviorForm(v, v_p_I[0], v_p_A[0])

    #########################################################
    # Compute the derivatives proper
    #########################################################
    for ui_id in range(u_p_I.size()):
        z = 0.
        if u_p_I[0][ui_id] < 0:
            para_prob=1.
        else:
            para_prob=u[u_p_I[0][ui_id]]
            z -= du[u_p_I[0][ui_id]]

        for q in range(u_c_I[0][ui_id].size()):
            a_id = u_c_I[0][ui_id][q]
            if not use_shortcut:
                z += du[a_id] * (1. + log(u[a_id]) - log(para_prob))
            else:
                z += du[a_id] * (1. + log(u_beh[a_id]))


        dlambdu.append(z)

    for vi_id in range(v_p_I.size()):
        z = 0.
        if v_p_I[0][vi_id] < 0:
            para_prob=1.
        else:
            para_prob=v[v_p_I[0][vi_id]]
            z -= dv[v_p_I[0][vi_id]]

        for q in range(v_c_I[0][vi_id].size()):
            a_id = v_c_I[0][vi_id][q]
            if not use_shortcut:
                z += dv[a_id] * (1. + log(v[a_id]) - log(para_prob))
            else:
                z += dv[a_id] * (1. + log(v_beh[a_id]))

        dlambdv.append(-z)

    return np.array(dlambdu), np.array(dlambdv)


    '''
    Old method of updating that does *NOT* use indexing
    # TODO: perform this compute only if gradients required
    # TODO: speed up this part
    for ui_id in range(len(self.Iu)):
        z = 0.
        if g.Par_inf_u[ui_id] is None:
            para_prob = 1.
        else:
            para_prob = u[g.Par_inf_u[ui_id]]
            z -= du[g.Par_inf_u[ui_id]]

        for q in g.Iu[ui_id]:
            z += du[q] * (1. + np.log(u[q]/para_prob))
        
        dlambdu[i, ui_id] = z

    for vi_id in range(len(self.Iv)):
        z = 0.
        if g.Par_inf_v[vi_id] is None:
            para_prob = 1.
        else:
            para_prob = v[g.Par_inf_v[vi_id]]
            z -= dv[g.Par_inf_v[vi_id]]

        for q in g.Iv[vi_id]:
            z += dv[q] * (1. + np.log(v[q]/para_prob))
        
        dlambdv[i, vi_id] = -z
    '''


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
    ''' Construct constraint matrix
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


cdef void SetVerbosity(int verbosity):
    glob_verbosity = verbosity
    log_msg('Set verbosity to %d' % verbosity, LOGGER_WARNING)

cdef inline void log_msg(msg, int level):
     if level >= glob_verbosity:
         logging.log(level, msg)
