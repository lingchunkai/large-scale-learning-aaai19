# Evaluate first and second order methods

# Add parent directory to path. Not needed if this is located at root directory.

import numpy as np
import matplotlib.pyplot as plt
from ..core.game import ZeroSumSequenceGame
from ..core.solve import QRESeqLogit
from ..core.solve import Solver
import itertools
import scipy
import copy

from ..core.fom_solve import QRESeqFOMLogit
from ..core.prox_solver import WrapperDualityGap

from ..core.prox_solver_back import backwardSolve

import timeit

import sacred
import faulthandler
faulthandler.enable()

ex = sacred.Experiment('r2_synthetic')

def GenSeqGame(gSize, depth=3, P_ranges=[-2, 2], lambda_ranges=[0.5,1.5], seed=0, num_samples=1):
    '''
    Generate games with deterministic transitions (no chance nodes)
    TODO: optimize

    :param gSize (uSize, vSize) of each component game
    :param depth of each player
    '''

    # Map infosets to infoset_id.
    # In this case, the infosets are just the history,
    # so they are 'equivalent' for each player
    infosets = [dict() for x in range(depth)]  # infosets[depth][.]
    actions = [[dict(), dict()] for x in range(depth)]  # actions[depth][id]

    r = np.random.RandomState(seed=seed)
    random_pay = lambda n: r.uniform(P_ranges[0], P_ranges[1], size=n)

    p_dict = [dict() for i in range(num_samples)]

    infosets[0][((), ())] = 0
    icount = 1
    u_acount, v_acount = 0, 0
    for d in range(0, depth):
        for i in infosets[d]:
            uact, vact = [], []
            # Add actions
            for uu in range(gSize[0]):
                action = (i[0] + (uu,), i[1])
                actions[d][0][action] = u_acount
                uact.append(u_acount)
                u_acount += 1

            for vv in range(gSize[1]):
                action = (i[0], i[1] + (vv,))
                actions[d][1][action] = v_acount
                vact.append(v_acount)
                v_acount += 1

            if d == depth - 1:
                tsize = [len(uact), len(vact), num_samples]
                gen = random_pay(tsize)
                for uact_id, vact_id in itertools.product(range(len(uact)), range(len(vact))):
                    uact_ = uact[uact_id]
                    vact_ = vact[vact_id]
                    for z in range(num_samples):
                        p_dict[z][(uact_, vact_)] = gen[uact_id, vact_id, z]

            # Add in dataset
            if d < depth - 1:
                for uu, vv in itertools.product(range(gSize[0]), range(gSize[1])):
                    infosets[d + 1][(i[0] + (uu,), i[1] + (vv,))] = icount
                    icount += 1

    # children of information set
    c_I = [[[] for i in range(icount)], [[] for i in range(icount)]]  # c_I[player id][infoset_id]
    c_A = [[[] for a in range(u_acount)], [[] for i in range(v_acount)]]
    p_I = [[None for i in range(icount)], [None for i in range(icount)]]
    p_A = [[None for a in range(u_acount)], [None for i in range(v_acount)]]

    # Get parent child relationships
    for d in range(0, depth):
        for i in infosets[d]:
            infoset_id = infosets[d][i]

            for uu in range(gSize[0]):
                action = (i[0] + (uu,), i[1])
                action_id = actions[d][0][action]

                c_I[0][infoset_id].append(action_id)
                p_A[0][action_id] = infoset_id

                # Set children infosets
                if d < depth - 1:
                    for vv in range(gSize[1]):
                        next_infoset = (action[0], action[1] + (vv,))
                        next_infoset_id = infosets[d + 1][next_infoset]
                        c_A[0][action_id].append(next_infoset_id)
                        p_I[0][next_infoset_id] = action_id

            for vv in range(gSize[1]):
                action = (i[0], i[1] + (vv,))
                action_id = actions[d][1][action]

                c_I[1][infoset_id].append(action_id)
                p_A[1][action_id] = infoset_id

                if d < depth - 1:
                    for uu in range(gSize[0]):
                        next_infoset = (action[0] + (uu,), action[1])
                        next_infoset_id = infosets[d + 1][next_infoset]
                        c_A[1][action_id].append(next_infoset_id)
                        p_I[1][next_infoset_id] = action_id

    # Converting to np array here makes for quicker indexing later
    # c_I = [np.array(x) for x in c_I]
    # c_A = [np.array(x) for x in c_A]

    c_I = [[np.array(y, dtype=np.intc) for y in x] for x in c_I]
    c_A = [[np.array(y, dtype=np.intc) for y in x] for x in c_A]
    p_I = [np.array(x) for x in p_I]
    p_A = [np.array(x) for x in p_A]

    # Generate lambdas
    lambdu_ret, lambdv_ret = [], []
    for z in range(num_samples):
        lambdu_ret.append(r.uniform(lambda_ranges[0], lambda_ranges[1], len(c_I[0])))
        lambdv_ret.append(r.uniform(lambda_ranges[0], lambda_ranges[1], len(c_I[1])))

    '''
    if p_dict == 1:
        p_dict = p_dict[0]
        lambdu_ret = lambdu_ret[0]
        lambdv_ret = lambdv_ret[1]
    '''

    return c_I, c_A, p_I, p_A, p_dict, lambdu_ret, lambdv_ret, actions, infosets


def FullP(p_dict, nAu, nAv, sp=False):
    ''' Convert payoff dictionary to full matrix
    '''
    if not sp:
        P = np.zeros([nAu, nAv])
        for k, v in p_dict.items():
            P[k[0], k[1]] = v
        return P
    else:
        items = p_dict.items()
        r = [z[0][0] for z in items]
        c = [z[0][1] for z in items]
        data = [z[1] for z in items]
        P = scipy.sparse.csr_matrix((data, (r,c)), shape=(nAu, nAv))
        print(P.shape)
        return P


def Sample(c_I, c_A, p_I, p_A, actions, infosets, u, v, batchsize, uniform=False, seed=0):
    '''

    :param c_I:
    :param c_A:
    :param p_I:
    :param p_A:
    :param actions:  dict (ref to GenSeqGame)
    :param infosets: dict (ref to GenSeqGame)
    :param u:
    :param v:
    :param args:
    :param seed:
    :return:
    '''

    r = np.random.RandomState(seed=seed)

    depth = len(infosets)
    cur_inf = ((), ())
    for d in range(0, depth):
        ##  get u player action
        infoset_id = infosets[d][cur_inf]
        uact_list = c_I[0][infoset_id]
        uprobs = u[uact_list]
        uprobs_1 = uprobs/np.sum(uprobs)
        uact = r.choice(range(uprobs.size), p=uprobs_1) # this is in {0,1,2...}
        uact_id = uact_list[uact]

        ##  get v player action
        infoset_id = infosets[d][cur_inf]
        vact_list = c_I[1][infoset_id]
        vprobs = u[vact_list]
        vprobs_1 = vprobs/np.sum(vprobs)
        vact = r.choice(range(vprobs.size), p=vprobs_1) # this is in {0,1,2...}
        vact_id = vact_list[vact]

        cur_inf = (cur_inf[0]+(uact,), cur_inf[1] + (vact,))

    return uact_id, vact_id


def Compare(c_I, c_A, p_I, p_A, P, lambdu, lambdv, args):
    g = ZeroSumSequenceGame(c_I[0], c_I[1], c_A[0], c_A[1], P, sp=args['sparse'])
    solver = QRESeqLogit(g, lambdas=[lambdu, lambdv], sp=args['sparse'])
    # u, v, dev2 = solver.Solve(tol=10 ** -32, epsilon=10 ** -16)

    gap=-1.0
    t_som=0.

    if args['no_som']==False:
        # SOM
        ret2 = dict()
        def solve_som_with_time():
            ret2['u'], ret2['v'], _ = solver.Solve(tol=10.**-3, epsilon=10. ** -20, verify=None, min_t=10.**-20)
        t_som = timeit.timeit(solve_som_with_time, number=args['num_trials_per_sample'])
        print('timing for som', t_som/args['num_trials_per_sample'])
        #print(ret2['u'])
        # print(ret2)

        # Compute gap
        gap = WrapperDualityGap(P,
                          c_I, c_A, p_I, p_A,
                          ret2['u'].flatten(), ret2['v'].flatten(), u_beh=None, v_beh=None,
                          lambd_=[lambdu, lambdv],
                          Ptype='keep',
                          prep_struct=None)

    print('gap:', gap)
    if gap <= args['min_allowed_gap']:
        gap = args['min_gap']
        print('gap < 0, reverting to min_gap', args['min_gap'])


    # FOM
    solve = QRESeqFOMLogit(g=g, lambd=None, tau=0.1, tol=gap, verify=None,duality_freq=args['fom_check_freq'])
    ret1 = dict()
    def solve_fom_with_time():
        ret1['u'], ret1['v'], _ = solve.Solve(g=None, lambd=[lambdu, lambdv],
                  tau=None, tol=None, max_it=None, Ptype=None,
                  verify=None, prep_struct=None, return_only_seq=True)
    t_fom = timeit.timeit(solve_fom_with_time, number=args['num_trials_per_sample'])
    # print(ret1['u'])
    print('timing for fom', t_fom/args['num_trials_per_sample'])
    # print(ret1)
    if args['no_som'] == False:
        print('norm diff between 2 u solns:', np.linalg.norm(ret1['u'].flatten()-ret2['u'].flatten()))

    return t_fom, t_som

    # print(ret2['u'])
    ''''
    f = lambda: Compare(c_I=c_I, c_A=c_A, p_I=p_I, p_A=p_A, P=P, lambdu=lambdu, lambdv=lambdv)
    '''


@ex.config
def config():
    args = dict()
    args['actions_size'] = [500,500]
    args['depth'] = 1
    args['experiment_path'] = './default_results/default_synth'
    args['seed'] = 0
    args['num_samples'] = 100
    args['lambda_range'] = [0.5, 1.5]
    args['P_ranges'] = [-10., 10.]
    args['fom_check_freq'] = 5
    args['num_trials_per_sample'] = 1
    args['min_gap'] = 10.**-12
    args['mode'] = 'forward'
    args['batchsize'] = 64
    args['min_allowed_gap'] = 10.**-12
    args['no_som'] = False
    args['tau_backward'] = 0.1

    args['sparse'] = False


def eval_forward(args):
    fom_list, som_list = [], []
    speedup_list = []
    for i in range(args['num_samples']):
        c_I, c_A, p_I, p_A, p_dict_arr, lambdu_arr, lambdv_arr, actions, infosets, = GenSeqGame(args['actions_size'],
                                                                            depth=args['depth'],
                                                                            lambda_ranges=args['lambda_range'],
                                                                            P_ranges=args['P_ranges'],
                                                                            seed=args['seed']+i,
                                                                            num_samples=1)
        if i == 0:
            print('Size of game:', len(c_A[0]), len(c_A[1]))

        lambdu = lambdu_arr[0]
        lambdv = lambdv_arr[0]
        P = FullP(p_dict_arr[0], len(c_A[0]), len(c_A[1]), sp=args['sparse'])

        t_fom, t_som = Compare(c_I=c_I, c_A=c_A, p_I=p_I, p_A=p_A, P=P, lambdu=lambdu, lambdv=lambdv, args=args)

        fom_list.append(t_fom)
        som_list.append(t_som)
        speedup_list.append(t_som/t_fom)

    print('fom_mean', np.mean(fom_list))
    print('som_mean', np.mean(som_list))
    print('speedup_mean', np.mean(speedup_list))
    print('fom_sd', np.std(fom_list))
    print('som_sd', np.std(som_list))
    print('speedup_std', np.std(speedup_list))
    print('fom', fom_list)
    print('som', som_list)
    print('speedup', speedup_list)


def eval_backward(args):
    fom_list, som_list = [], []
    speedup_list = []
    for i in range(args['num_samples']):
        print('test:', i)
        c_I, c_A, p_I, p_A, p_dict_arr, lambdu_arr, lambdv_arr, actions, infosets = GenSeqGame(args['actions_size'],
                                                                            depth=args['depth'],
                                                                            lambda_ranges=args['lambda_range'],
                                                                            P_ranges=args['P_ranges'],
                                                                            seed=args['seed']+i,
                                                                            num_samples=1)
        if i == 0:
            print('Size of game:', len(c_A[0]), len(c_A[1]))
        # if i < 24: continue
        # print(p_dict_arr)
        # print(lambdu_arr)
        # if i >24: continue


        lambdu = lambdu_arr[0]
        lambdv = lambdv_arr[0]
        P = FullP(p_dict_arr[0], len(c_A[0]), len(c_A[1]), sp=args['sparse'])
        if args['mode'] == 'backward_exact':
            g = ZeroSumSequenceGame(c_I[0], c_I[1], c_A[0], c_A[1], P, sp=args['sparse'])
            # use FOM for ground truth of U and V
            solve = QRESeqFOMLogit(g=g, lambd=None, tau=0.1, tol=args['min_gap'], verify=None, duality_freq=args['fom_check_freq'])
            ret1 = dict()
            ret1['u'], ret1['v'], _ = solve.Solve(g=None, lambd=[lambdu, lambdv],
                                                  tau=None, tol=None, max_it=None, Ptype=None,
                                                  verify=None, prep_struct=None, return_only_seq=True)

            u_seq = ret1['u']
            v_seq = ret1['v']

            # Lets draw a sample!
            u_act, v_act = Sample(c_I, c_A, p_I, p_A, actions, infosets, u_seq, v_seq,
                                  batchsize=args['batchsize'], uniform=False, seed=args['seed']+i)
        else:
            assert False, 'not done yet'

        t_fom, t_som = CompareBack(c_I, c_A, p_I, p_A, P, lambdu, lambdv,
                        u_seq, v_seq, u_act, v_act, args)

        fom_list.append(t_fom)
        som_list.append(t_som)
        speedup_list.append(t_som/t_fom)

    print('fom_mean', np.mean(fom_list))
    print('som_mean', np.mean(som_list))
    print('speedup_mean', np.mean(speedup_list))
    print('fom_sd', np.std(fom_list))
    print('som_sd', np.std(som_list))
    print('speedup_std', np.std(speedup_list))
    print('fom', fom_list)
    print('som', som_list)
    print('speedup', speedup_list)


def CompareBack(c_I, c_A, p_I, p_A, P, lambdu, lambdv,
                u_seq, v_seq, u_act, v_act, args):

    usize, vsize = len(c_A[0]), len(c_A[1])

    # Compute losses and whatnot
    loss = -np.log(u_seq[u_act]) - np.log(v_seq[v_act])
    dlossu, dlossv = np.zeros(u_seq.size), np.zeros(v_seq.size)
    dlossu[u_act] = -1./u_seq[u_act]
    dlossv[v_act] = -1./v_seq[v_act]

    # FOM for backward Setup backward
    ret1 = dict()
    def solve_fom_with_time():
        u_solve_back, v_solve_back = backwardSolve(P,
                                                   u_seq.squeeze(),
                                                   v_seq.squeeze(),
                                                   c_I, c_A,
                                                   p_I, p_A,
                                                   tau=args['tau_backward'],
                                                   tol=10.**-10,
                                                   lossu=dlossu, lossv=dlossv,
                                                   lambd_=(lambdu, lambdv),
                                                   max_iters=10000,
                                                   verify=False,
                                                   duality_freq=args['fom_check_freq'])
        ret1['u_solve_back'] = u_solve_back
        ret1['v_solve_back'] = v_solve_back
    t_fom = timeit.timeit(solve_fom_with_time, number=args['num_trials_per_sample'])

    if not args['no_som']:
        # Direct solution
        ret2 = dict()
        M, stk = SOMInvPrep(c_I, c_A, p_I, p_A, P, lambdu, lambdv,
                   u_seq, v_seq, dlossu, dlossv, args['sparse'])
        def solve_som_with_time():
            if args['sparse'] == False:
                s = np.linalg.solve(M, stk)
            else:
                s = scipy.sparse.linalg.spsolve(M, stk)
            u_solve_som_back = s[:usize]
            v_solve_som_back = s[usize:(usize+vsize)]
            ret2['u_solve_back'] = u_solve_som_back
            ret2['v_solve_back'] = v_solve_som_back
        t_som = timeit.timeit(solve_som_with_time, number=args['num_trials_per_sample'])
        # print(ret1['u_solve_back'], ret2['u_solve_back'])
        print('Normdiff', np.linalg.norm(ret1['u_solve_back'].squeeze() - ret2['u_solve_back'].squeeze()))
    else: t_som = 0.

    print('fom', t_fom)
    print('som', t_som)

    return t_fom, t_som

def SOMInvPrep(c_I, c_A, p_I, p_A, P, lambdu, lambdv,
                u_seq, v_seq, dlossu, dlossv, sp=False):
    Xi_u = GetXiMatrix2(u_seq, lambdu, p_I[0], p_A[0], sp=sp)
    Xi_v = GetXiMatrix2(v_seq, lambdv, p_I[1], p_A[1], sp=sp)

    usize = dlossu.size
    vsize = dlossv.size

    print('godamn game')
    g = ZeroSumSequenceGame(c_I[0], c_I[1], c_A[0], c_A[1], P, sp=sp)
    print('goddamn game over')
    stk = np.concatenate([-dlossu, -dlossv, np.zeros(g.Cu.shape[0]), np.zeros(g.Cv.shape[0])], axis=0)

    if sp == False:
        M1 = np.concatenate([Xi_u, P, g.Cu.T, np.zeros([usize, g.Cv.T.shape[1]])], axis=1)
        M2 = np.concatenate([P.T, -Xi_v, np.zeros([vsize, g.Cu.T.shape[1]]), g.Cv.T], axis=1)
        M3 = np.concatenate([g.Cu, np.zeros([g.Cu.shape[0], M2.shape[1]-g.Cu.shape[1]])], axis=1)
        M4 = np.concatenate([np.zeros([g.Cv.shape[0], g.Cu.shape[1]]), g.Cv, np.zeros([g.Cv.shape[0], g.Cu.shape[0] + g.Cv.shape[0]])], axis=1)
        M = np.concatenate([M1, M2, M3, M4], axis=0)
    else:
        CuSp, _ = g.MakeConstraintMatrices(g.Iu, g.Par_inf_u, len(g.Au), sp=True)
        CvSp, _ = g.MakeConstraintMatrices(g.Iv, g.Par_inf_v, len(g.Av), sp=True)
        print('sparse done')
        # M1 = scipy.sparse.bmat([[Xi_u, P, g.Cu.T, np.zeros([usize, g.Cv.T.shape[1]])]], format='csr')
        M1 = scipy.sparse.bmat([[Xi_u, P, CuSp.T, scipy.sparse.csr_matrix(([], ([], [])), shape=[usize, g.Cv.T.shape[1]])]], format='csr')
        M2 = scipy.sparse.bmat([[P.T, -Xi_v, scipy.sparse.csr_matrix(([], ([], [])), shape=[vsize, CuSp.T.shape[1]]), CvSp.T]], format='csr')
        # M3 = scipy.sparse.bmat([[g.Cu, np.zeros([g.Cu.shape[0], M2.shape[1]-g.Cu.shape[1]])]])
        M3 = scipy.sparse.bmat([[CuSp, scipy.sparse.csr_matrix(([], ([], [])), shape=(CuSp.shape[0], M2.shape[1]-CuSp.shape[1]))]], format='csr')
        # M4 = scipy.sparse.bmat([[np.zeros([g.Cv.shape[0], g.Cu.shape[1]]), g.Cv, np.zeros([g.Cv.shape[0], g.Cu.shape[0] + g.Cv.shape[0]])]])
        M4 = scipy.sparse.bmat([[scipy.sparse.csr_matrix(([], ([], [])), shape=([CvSp.shape[0], CuSp.shape[1]])),
                                CvSp,
                                scipy.sparse.csr_matrix(([], ([], [])), shape=(CvSp.shape[0], CuSp.shape[0] + CvSp.shape[0]))]], format='csr')
        M = scipy.sparse.bmat([[M1], [M2], [M3], [M4]], format='csr')

    print('done M')
    return M, stk

def GetXiMatrix2(uv, lambd, uv_p_I, uv_p_A, sp=False):
    ''' Same as GetXiMatrix except we accept lambdas
    '''

    if sp == False:
        ret = np.zeros([uv.size, uv.size])
        sum_lambds_c = [0] * uv.size  # previously just the number of children
        for u_id in range(uv.size):
            if uv_p_A[u_id] is None: continue
            pari = uv_p_A[u_id]
            if uv_p_I[pari] is None: continue
            parpara = uv_p_I[pari]

            ret[parpara][u_id] = -lambd[pari] / uv[parpara]
            ret[u_id][parpara] = -lambd[pari] / uv[parpara]

        for i_id in range(len(uv_p_I)):
            if uv_p_I[i_id] is not None:
                sum_lambds_c[uv_p_I[i_id]] += lambd[i_id]

        for u_id in range(uv.size):
            if uv_p_A[u_id] is None:
                s = 1.
            else:
                s = lambd[uv_p_A[u_id]]
            ret[u_id][u_id] = (s + sum_lambds_c[u_id]) / uv[u_id]

        return ret

    elif sp == True:
        ret = dict()
        sum_lambds_c = [0] * uv.size  # previously just the number of children
        for u_id in range(uv.size):
            if uv_p_A[u_id] is None: continue
            pari = uv_p_A[u_id]
            if uv_p_I[pari] is None: continue
            parpara = uv_p_I[pari]

            ret[(parpara, u_id)] = -lambd[pari] / uv[parpara]
            ret[(u_id, parpara)] = -lambd[pari] / uv[parpara]

        for i_id in range(len(uv_p_I)):
            if uv_p_I[i_id] is not None:
                sum_lambds_c[uv_p_I[i_id]] += lambd[i_id]

        for u_id in range(uv.size):
            if uv_p_A[u_id] is None:
                s = 1.
            else:
                s = lambd[uv_p_A[u_id]]
            ret[(u_id, u_id)] = (s + sum_lambds_c[u_id]) / uv[u_id]

        items = ret.items()
        r = [z[0][0] for z in items]
        c = [z[0][1] for z in items]
        data = [z[1] for z in items]
        return scipy.sparse.csr_matrix((data, (r, c)), shape=(uv.size, uv.size))


@ex.automain
def main(args):
    if args['mode'] == 'forward':
        eval_forward(args)
    elif args['mode'] == 'backward_exact' or args['mode'] == 'backward_uniform':
        eval_backward(args)
    '''
    # Solve using fom
    u_fast, v_fast = prox_solver.forwardSolve(P, c_I, c_A, p_I, p_A,
                                              lambd_=(lambdu, lambdv),
                                              tau=0.025, tol=10. ** -52, max_iters=10000, verify=True)

    # Solve using som
    g = ZeroSumSequenceGame(c_I[0], c_I[1], c_A[0], c_A[1], P)
    solver = QRESeqLogit(g)
    u, v, dev2 = solver.Solve(tol=10 ** -32, epsilon=10 ** -16)
    '''





