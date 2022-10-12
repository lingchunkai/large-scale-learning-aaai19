'''
QRE solver for *zero sum* 2 player games
'''

import numpy as np
import scipy.sparse
from . import game
from .solve import Solver
from . import prox_solver
from . import prox_solver_back
from .solve import QRESeqLogit

import logging
logger = logging.getLogger(__name__)

from . import util

class QRESeqFOMLogit(object):
    def __init__(self, g=None, lambd=None,
            tau=0.01, tol=10.**-10, max_it=10000, Ptype='keep',
            verify=None, prep_struct=None, duality_freq=10):
        # TODO: subclass of solver
        self.g = g
        self.lambd = lambd
        self.tau = tau
        self.tol = tol
        self.max_it = max_it
        self.Ptype = Ptype
        self.verify = verify
        self.prep_struct = prep_struct
        self.duality_freq=duality_freq

    def Solve(self, g=None, lambd=None, 
            tau=None, tol=None, max_it=None, Ptype=None,
            verify=None, prep_struct=None, return_only_seq=True):
        '''
        :param lambd = 2-tuple of numpy arrays (u_lambd, v_lambd), each of size
            equal to the number of information sets, containing rationality params.
        :param verify TODO: VERIFY correctness of solution
        '''
        if g is None: g = self.g
        if lambd is None: lambd = self.lambd
        if tau is None: tau = self.tau
        if tol is None: tol = self.tol
        if max_it is None: max_it = self.max_it
        if Ptype is None: Ptype = self.Ptype
        if verify is None: verify = self.verify
        if prep_struct is None: prep_struct = self.prep_struct

        assert g is not None, 'No game supplied to solver!'

        # Default value of lambda if it was not supplied either
        # by constructor or by Solve(.)
        if lambd is None:
            lambd = (np.ones(len(g.Iu)), np.ones(len(g.Iv)))

        P = g.Pz
        c_I = (g.Iu, g.Iv)
        c_A = (g.Au, g.Av)
        p_I = (g.Par_inf_u, g.Par_inf_v)
        p_A = (g.Par_act_u, g.Par_act_v)
        # prep_struct = util.MakeGameStructure(c_I, c_A, p_I, p_A)

        if return_only_seq:
            u_solve, v_solve = prox_solver.forwardSolve(P, c_I, c_A, p_I, p_A,
                    tau=tau, lambd_=lambd,
                    tol=tol, max_iters=max_it, Ptype=Ptype, verify=verify, prep_struct=prep_struct, duality_freq=self.duality_freq)
            return u_solve, v_solve, None
        else:
            u_solve, v_solve, u_beh, v_beh = prox_solver.forwardSolve(P, c_I, c_A, p_I, p_A,
                    tau=tau, lambd_=lambd,
                    tol=tol, max_iters=max_it, Ptype=Ptype, verify=verify, prep_struct=prep_struct, duality_freq=self.duality_freq,
                    return_only_seq=False)
            # print(np.max(v_beh))
            return (u_solve, u_beh), (v_solve, v_beh), None

    
    def BackwardsSolve(self, u, v, lossu, lossv, lambd=None, g=None,
                    tau=None, tol=None, max_it=None, Ptype=None,
                    verify=None, prep_struct=None,
                    u_beh=None, v_beh=None,
                    use_shortcut=False):
        '''
        :param u, v sequence form strategies found from forwrad solver.
        :param lossu, lossv nparrays containing gradient of losses.
        :param lambd = 2-tuple of numpy arrays (u_lambd, v_lambd), each of size
            equal to the number of information sets, containing rationality params.
        :param verify Verify correctness of solution
        '''
        
        if g is None: g = self.g
        if lambd is None: lambd = self.lambd
        if tau is None: tau = self.tau
        if tol is None: tol = self.tol
        if max_it is None: max_it = self.max_it
        if Ptype is None: Ptype = self.Ptype
        if verify is None: verify = self.verify
        if prep_struct is None: prep_struct = self.prep_struct

        assert g is not None, 'No game supplied to solver!'

        # Default value of lambda if it was not supplied either
        # by constructor or by Solve(.)
        if lambd is None:
            lambd = (np.ones(len(g.Iu)), np.ones(len(g.Iv)))
        
        P = g.Pz
        c_I = (g.Iu, g.Iv)
        c_A = (g.Au, g.Av)
        p_I = (g.Par_inf_u, g.Par_inf_v)
        p_A = (g.Par_act_u, g.Par_act_v)

        x, y = prox_solver_back.backwardSolve(P, u, v, c_I, c_A, p_I, p_A, 
                tau=tau, lossu=lossu, lossv=lossv, lambd_=lambd,
                tol=tol, max_iters=max_it, Ptype=Ptype,
                verify=verify, prep_struct=prep_struct,
                u_beh=u_beh, v_beh=v_beh,
                use_shortcut=use_shortcut)

        return x, y


    def BackwardSolveLU(self, P, u, v, u_lambd, v_lambd, g=None):
        # Construct giant matrix to be inverted.
        E_u = self.ConstructE(g.Iu, g.Par_inf_u, g.Au, g.Par_act_u).tocsc()
        E_v = self.ConstructE(g.Iv, g.Par_inf_v, g.Av, g.Par_act_v).tocsc()
        Xi_u = self.ConstructXi(g.Iu, g.Par_inf_u, g.Au, g.Par_act_u, u, u_lambd).tocsc()
        Xi_v = self.ConstructXi(g.Iv, g.Par_inf_v, g.Av, g.Par_act_v, v, v_lambd).tocsc()
        mat = scipy.sparse.bmat([[Xi_u, P, E_u.T, None], 
                        [P.T, -Xi_v, None, E_v.T], 
                        [E_u, None, None, None], 
                        [None, E_v, None, None]], format='csc')

        inv = scipy.sparse.linalg.splu(mat)

        return inv

    
    def ConstructE(self, c_I, p_I, c_A, p_A):
        ''' Construct constraint matrix
        '''
        nActions = len(c_A)
        nInfosets = len(c_I)
        r, c, d = [], [], []
        for i_id in range(nInfosets):
            # parent
            a_id = p_I[i_id]
            if a_id is not None: 
                r.append(i_id)
                c.append(a_id)
                d.append(-1)

            # child
            for a_id in c_I[i_id]:
                r.append(i_id)
                c.append(a_id)
                d.append(1)

        return scipy.sparse.coo_matrix((d,(r,c)), shape=(nInfosets, nActions))

    def ConstructXi(self, c_I, p_I, 
                    c_A, p_A,
                    u,
                    lambd):
        '''
        Construct (potentially) sparse Xi.
        '''

        nActions = len(c_A)
        nInfosets = len(c_I)
        sI = [0.] * nActions
        r, c, d = [], [], []

        for a_id in range(nActions):
            # child
            for i_id in c_A[a_id]:
                sI[a_id] += lambd[i_id]
                for child_a_id in c_I[i_id]:
                    r.append(a_id)
                    c.append(child_a_id)
                    d.append(-lambd[i_id]/u[a_id])
            # self
            r.append(a_id)
            c.append(a_id)
            d.append((lambd[p_A[a_id]] + sI[a_id])/u[a_id])

            # parent
            p_i_id = p_A[a_id]
            if p_I[p_i_id] is None: continue
            para_a_id = p_I[p_i_id]
            r.append(a_id)
            c.append(para_a_id)
            d.append(-lambd[p_i_id]/u[para_a_id])

        mat = scipy.sparse.coo_matrix((d,(r,c)), shape=(nActions, nActions))

        return mat

if __name__ == '__main__':
    # Log to both file and stdout
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler('solver.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    print('---- Sequence Form -----')
    zssg = game.ZeroSumSequenceGame.MatchingPennies(left=0.75, right=0.25)
    FOMsolver = QRESeqFOMLogit(zssg)
    u_fom, v_fom, _ = FOMsolver.Solve(max_it=1000, tol=10**-10, verify='both')
    solver = QRESeqLogit(zssg)
    u_gt, v_gt, _ = solver.Solve(max_it=1000, tol=10**-10, verify='both')
    print('Solver difference:', np.linalg.norm(u_fom - np.squeeze(u_gt)), 
            np.linalg.norm(v_fom-np.squeeze(v_gt)))
    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.PerfectInformationMatchingPennies(left=3., right=1.)
    FOMsolver = QRESeqFOMLogit(zssg)
    u_fom, v_fom, _ = FOMsolver.Solve(max_it=1000, tol=10**-10, verify='both')
    solver = QRESeqLogit(zssg)
    u_gt, v_gt, _ = solver.Solve(max_it=1000, tol=10**-10, verify='both')
    print('Solver difference:', np.linalg.norm(u_fom - np.squeeze(u_gt)), 
            np.linalg.norm(v_fom-np.squeeze(v_gt)))
    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.MinSearchExample()
    FOMsolver = QRESeqFOMLogit(zssg)
    u_fom, v_fom, _ = FOMsolver.Solve(max_it=1000, tol=10**-10, verify='both')
    solver = QRESeqLogit(zssg)
    u_gt, v_gt, _ = solver.Solve(max_it=1000, tol=10**-10, verify='both')

    print('Solver difference:', np.linalg.norm(u_fom - np.squeeze(u_gt)), 
            np.linalg.norm(v_fom-np.squeeze(v_gt)))
    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.MaxSearchExample()
    FOMsolver = QRESeqFOMLogit(zssg)
    u_fom, v_fom, _ = FOMsolver.Solve(max_it=1000, tol=10**-10, verify='both')
    solver = QRESeqLogit(zssg)
    u_gt, v_gt, _ = solver.Solve(max_it=1000, tol=10**-10, verify='both')
    print('Solver difference:', np.linalg.norm(u_fom - np.squeeze(u_gt)), 
            np.linalg.norm(v_fom-np.squeeze(v_gt)))
    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.OneCardPoker(4)
    FOMsolver = QRESeqFOMLogit(zssg)
    u_fom, v_fom, _ = FOMsolver.Solve(max_it=1000, tol=10**-10, verify='both')
    solver = QRESeqLogit(zssg)
    u_gt, v_gt, _ = solver.Solve(max_it=1000, tol=10**-10, verify='both')
    print(zssg.ComputePayoff(u_fom, v_fom))
    print('Solver difference:', np.linalg.norm(u_fom - np.squeeze(u_gt)), 
            np.linalg.norm(v_fom-np.squeeze(v_gt)))
    print('Any key to continue')
    input()

    zssg = game.ZeroSumSequenceGame.OneCardPoker(4, 10.0)
    FOMsolver = QRESeqFOMLogit(zssg)
    u_fom, v_fom, _ = FOMsolver.Solve(max_it=1000, tol=10**-10, verify='both')
    solver = QRESeqLogit(zssg)
    u_gt, v_gt, _ = solver.Solve(max_it=1000, tol=10**-10, verify='both')
    print(zssg.ComputePayoff(u_fom, v_fom))
    print('Solver difference:', np.linalg.norm(u_fom - np.squeeze(u_gt)), 
            np.linalg.norm(v_fom-np.squeeze(v_gt)))
