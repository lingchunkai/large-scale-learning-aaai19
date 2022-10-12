import numpy as np
import argparse
import h5py, scipy
from itertools import *

from ..core.game import ZeroSumSequenceGame 
from ..core.solve import QRESeqLogit
from ..core.fom_paynet import ZSGFOMSolver
from ..core.fom_solve import QRESeqFOMLogit

from ..experiments.datagen import DataGen, dict_to_h5
from ..experiments.seq_datagen import SeqDataGen

from .poker_eval import OneCardPokerComputeP, OneCardPokerComputeSets, OneCardPokerSeqSample

import timeit

class OneCardPokerRatGen(object):
    '''
    One card poker with WNHD distributed card draws
    '''
    def __init__(self, nCards, initial=1., raiseval=1.,
                 seed=0, data_sampling_dist='same_as_cards',
                 lambd_range=(0.001, 0.010), tau=0.5):
        '''
        :param dist normalized (e.g. via softmax) card distribution
        :param data_sampling_dist card distribution the *data* is sampled from. 
               Note this *can* be different from the card distribution assumed by the players.
               defaults to the same as dist (i.e. data distribution is the same as card distribution) 
        '''
        self.nCards = nCards
        self.nFeats = 1
        self.initial, self.raiseval = initial, raiseval
        self.Iu, self.Iv, self.Au, self.Av = OneCardPokerComputeSets(nCards)

        self.r = np.random.RandomState(seed)
        self.dist = np.array([1./nCards for i in range(nCards)])

        self.lambd_range = lambd_range

        if data_sampling_dist is not 'same_as_cards':
            if data_sampling_dist == 'uniform': # Almost uniform...
                self.data_sampling_dist = [1./nCards] * self.nCards 
                print('sampling from uniform')
            else:
                assert False, 'Unknown data sampling distribution'
        else: # same as card distribution
            self.data_sampling_dist = self.dist

        # Compute GT
        self.P, _, _ = OneCardPokerComputeP(nCards, initial=initial, raiseval=raiseval)
        print(self.P)
        self.dense_P = self.P
        self.P = scipy.sparse.csr_matrix(self.P)


        self.g = ZeroSumSequenceGame(self.Iu, self.Iv, self.Au, self.Av, self.P, sp=True)
        self.solve = QRESeqFOMLogit(tau=0.1, tol=10. ** -10,
                                    max_it=10000, Ptype='keep', verify=True)


        self.tau=tau

    
    def GenData(self, nSamples):
        # range from lambd_range_u[0] to (lambd_rnage_u[0] + lambd_range_u[1])

        lambd_min = self.lambd_range[0]
        lambd_maxgap = self.lambd_range[1]

        # parameter to be learnt
        weights = self.r.uniform(0, lambd_maxgap, size=4)

        F = self.r.uniform(0, 1, size=[nSamples, 1])
        W = weights
        Au = np.zeros(nSamples)
        Av = np.zeros(nSamples)
        U = np.zeros((nSamples, 4 * self.nCards))
        V = np.zeros((nSamples, 4 * self.nCards))
        for n in range(nSamples):

            print('Sample', n)

            ls = F[n] * weights + lambd_min

            lu = np.array([ls[0]] * self.nCards + [ls[1]] * self.nCards)
            lv = np.array([ls[2]] * self.nCards + [ls[3]] * self.nCards)

            # Solve using FOM

            print('S1')
            ugt, vgt, _ = self.solve.Solve(g=self.g, lambd=[lu, lv],
                                        tau=self.tau, tol=10.**-15, max_it=None, Ptype=None,
                                        verify='none', prep_struct=None, return_only_seq=True)
            print('-S1')

            """
            # Solve using dense Newton
            self.g_nonsparse = ZeroSumSequenceGame(self.Iu, self.Iv, self.Au, self.Av, self.dense_P, sp=False)
            brute_solver = QRESeqLogit(self.g_nonsparse, lambdas=[lu, lv])
            print('S2')
            ugt, vgt, _ = brute_solver.Solve(alpha=0.1, beta=0.5, tol=10 ** -4, epsilon=0., min_t=10 ** -25,
                                             max_it=10000, verify='none')
            print('-S2')
            """

            """
            # Solve using sparse Newton
            brute_solver = QRESeqLogit(self.g, lambdas=[lu, lv], sp=True)
            ugt, vgt, _ = brute_solver.Solve(alpha=0.1, beta=0.5, tol=10**-7, epsilon=0., min_t=10**-25, max_it=10000, verify='none')
            print(np.dot(self.dense_P, vgt.squeeze()).dot(ugt.squeeze()))
            """

            U[n, :] = ugt.squeeze()
            V[n, :] = vgt.squeeze()


            print('payoffs', np.dot(self.dense_P, vgt.squeeze()).dot(ugt.squeeze()))
            print('lambdas', ls)


            au, av = OneCardPokerSeqSample(self.nCards, ugt, vgt, r=self.r)
            Au[n] = au
            Av[n] = av

        return F, Au, Av, U, V, W



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''generate extensive form dataset (sequence form).\n
                                python -m src.fom_evaluation.poker_eval_datagen --nSamples=200 --nCards=200 ''')
    parser.add_argument('--nCards', type=int, required=True)
    parser.add_argument('--nSamples', type=int, required=True,
                        help='number of samples to generate.')
    parser.add_argument('--seed', type=int, default=0,
                        help='randomization seed.')
    parser.add_argument('--savepath', type=str, default='./data/fom_eval_poker.h5',
                        help='path to save generated dataset.')
    parser.add_argument('--lambd_min', type=float, default=0.001, help='Min lambda')
    parser.add_argument('--lambd_maxrange', type=float, default=0.01, help = 'Maximum lambda range')
    parser.add_argument('--tau', type=float, default=0.5, help = 'tau for solving')

    args = parser.parse_args()

    gen = OneCardPokerRatGen(nCards=args.nCards, seed=args.seed, lambd_range=(args.lambd_min, args.lambd_maxrange))
    F, Au, Av, U, V, W = gen.GenData(nSamples=args.nSamples)

    f = h5py.File(args.savepath, 'w')
    f['F'] = F
    f['Au'] = Au
    f['Av'] = Av
    f['U'] = U
    f['V'] = V
    f['W'] = W


