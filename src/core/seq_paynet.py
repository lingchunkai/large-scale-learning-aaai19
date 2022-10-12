import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import *

from .game import ZeroSumSequenceGame 
from .solve import QRESeqLogit


class ZSGSeqSolver(torch.autograd.Function):
    # Naive way of iterating through each item in the mini-batch
    def __init__(self, Iu, Iv, Au, Av, verify='kkt', single_feature=False,
                 fixed_P=None):
        self.Iu, self.Iv = Iu, Iv
        self.Au, self.Av = Au, Av
        self.usize, self.vsize = len(Au), len(Av)
        self.uinfosize, self.vinfosize = len(Iu), len(Iv)
        self.verify = verify
        self.single_feature = single_feature

        # This is for cases where P is known a priori, usually when we are learning lambdas
        self.fixed_P = fixed_P
        if self.fixed_P is None:
            self.use_default_P = False
        else:
            self.use_default_P = True

        super(ZSGSeqSolver, self).__init__()

    def forward(self, input, lambdu=None, lambdv=None):
        input_np = input.numpy()

        batchsize = input_np.shape[0]

        # Lambda gradients if we need them
        self.need_lambd_grad_u, self.need_lambd_grad_v = False, False
        if lambdu is not None:
            lambdu_np = lambdu.numpy().astype(np.float64)
            self.need_lambd_grad_u = True
        else:
            lambdu_np = np.ones([batchsize, len(self.Iu)])

        if lambdv is not None:
            lambdv_np = lambdv.numpy().astype(np.float64)
            self.need_lambd_grad_v = True
        else:
            lambdv_np = np.ones([batchsize, len(self.Iu)])

        U, V = np.zeros([batchsize, self.usize], dtype=np.float64), np.zeros([batchsize, self.vsize], dtype=np.float64)
        self.dev2s = []
        for i in range(batchsize):
            if not self.single_feature or i == 0:
                if not self.use_default_P:
                    p = np.squeeze(input_np[i, :, :])
                else:
                    p = self.fixed_P

                game = ZeroSumSequenceGame(self.Iu, self.Iv, self.Au, self.Av, p)
                logitSolver = QRESeqLogit(game, lambdas=[lambdu_np[i, :], lambdv_np[i, :]])
                u, v, dev2 = logitSolver.Solve(alpha=0.15, beta=0.50, tol=10**-7, epsilon=0., min_t=10**-20,
                                               max_it=1000, verify=self.verify)
                U[i, :] , V[i, :] = np.squeeze(u), np.squeeze(v)
                self.dev2s.append(dev2)
            else: # reuse old result when input features are guaranteed to be the same
                U[i, :] , V[i, :] = np.squeeze(u), np.squeeze(v)
                self.dev2s.append(dev2)

        Ut, Vt = torch.DoubleTensor(U), torch.DoubleTensor(V)
        self.input, self.U, self.V = input_np, U, V
        # IMPORTANT NOTE: save for backward does not work unless static method! Memory accumulates and leaks
        # self.save_for_backward(input, U, V)
        return Ut, Vt

    def backward(self, grad_u, grad_v):
        batchsize = grad_u.shape[0]
        # P, U, V = tuple([x.data.numpy() for x in self.saved_variables])
        P, U, V = self.input, self.U, self.V

        if self.needs_input_grad[0]:
            dP = np.zeros([batchsize, self.usize, self.vsize])
        else: dP = None

        if self.needs_input_grad[1]:
            dlambdu = np.zeros([batchsize, len(self.Iu)])
        else: dlambdu = None

        if self.needs_input_grad[2]:
            dlambdv = np.zeros([batchsize, len(self.Iv)])
        else: dlambdv = None

        for i in range(batchsize):
            u, v = U[i, :], V[i, :]
            # p = P[i, :, :]
            
            gu, gv = grad_u[i, :].numpy(), grad_v[i, :].numpy()
            solve_rhs = -np.concatenate([gu, gv, [0] * (self.uinfosize + self.vinfosize)], axis=0)
            if not self.single_feature:
            # if True:
                d = np.linalg.solve(self.dev2s[i], solve_rhs)
            elif self.single_feature and i == 0:
                lu_and_piv = scipy.linalg.lu_factor(self.dev2s[0], check_finite=False)
                d = scipy.linalg.lu_solve(lu_and_piv, solve_rhs, check_finite=False)
            else:
                d = scipy.linalg.lu_solve(lu_and_piv, solve_rhs, check_finite=False)

            du, dv = d[:self.usize], d[self.usize:(self.usize+self.vsize)]
            if self.needs_input_grad[0]:
                dp = np.outer(du, v) + np.outer(u, dv)
                dP[i, :, :] = dp
            if self.needs_input_grad[1] or self.needs_input_grad[2]:
                # g = ZeroSumSequenceGame(self.Iu, self.Iv, self.Au, self.Av, p)
                g = ZeroSumSequenceGame(self.Iu, self.Iv, self.Au, self.Av, 0)

                k = np.zeros(dlambdu.shape[1])
                pprobs = np.ones(self.uinfosize)
                nonNone = np.logical_not(np.equal(g.Par_inf_u, None))
                tmp = np.array(g.Par_inf_u)[nonNone].astype(int)
                pprobs[nonNone] = u[tmp]
                k[nonNone] -= np.atleast_1d(du.squeeze())[tmp]

                parpprobs = np.ones(self.usize)
                parActNonNone = np.logical_not(np.equal(g.ParPar_act_u, None))
                parpprobs[parActNonNone] = u[np.array(g.ParPar_act_u)[parActNonNone].astype(int)]
                beh = u/parpprobs
                k += np.bincount(g.Par_act_u, np.atleast_1d(du.squeeze()) * (1+np.log(beh)))

                dlambdu[i, :] = k

                # For V
                k = np.zeros(dlambdv.shape[1])
                pprobs = np.ones(self.vinfosize)
                nonNone = np.logical_not(np.equal(g.Par_inf_v, None))
                tmp = np.array(g.Par_inf_v)[nonNone].astype(int)
                pprobs[nonNone] = v[tmp]
                k[nonNone] -= np.atleast_1d(dv.squeeze())[tmp]

                parpprobs = np.ones(self.vsize)
                parActNonNone = np.logical_not(np.equal(g.ParPar_act_v, None))
                parpprobs[parActNonNone] = v[np.array(g.ParPar_act_v)[parActNonNone].astype(int)]
                beh = v/parpprobs
                k += np.bincount(g.Par_act_v, np.atleast_1d(dv.squeeze()) * (1+np.log(beh)))

                dlambdv[i, :] = -k

        dlambdu = torch.DoubleTensor(dlambdu)
        dlambdv = torch.DoubleTensor(dlambdv)
        if dP is not None: dp = torch.DoubleTensor(dP)
        return dP, dlambdu, dlambdv


if __name__ == '__main__':
    pass
