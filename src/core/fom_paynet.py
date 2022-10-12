import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import *

from .game import ZeroSumSequenceGame
from .fom_solve import QRESeqFOMLogit
from .solve import QRESeqLogit

from . import prox_solver_back
from . import util

class ZSGFOMSolver(torch.autograd.Function):
    # Naive way of iterating through each item in the mini-batch
    # TODO: Is it necessary to fix Iu, Iv, ... in the constructor?
    def __init__(self, Iu, Iv, Au, Av,
            solver=None,
            single_feature=False,
            use_inverse_back=False,
            fixed_P = None,
            tau_both=(0.1, 0.1)):
        
        self.Iu, self.Iv = Iu, Iv
        self.Au, self.Av = Au, Av
        self.usize, self.vsize = len(Au), len(Av)
        self.uinfosize, self.vinfosize = len(Iu), len(Iv)
        self.single_feature = single_feature

        self.forward_tau, self.backward_tau = tau_both[0], tau_both[1]

        # Compute sparse LU decomposition for fast backwards passes
        # If false, will use FOM backward solver.
        # Only use when single_feature is true.
        self.use_inverse_back = use_inverse_back

        if solver is None:
            self.solver = QRESeqFOMLogit(tau=0.1, tol=10.**-10,
                    max_it=10000, Ptype='keep', verify=None)
        else: self.solver = solver

        # Create game
        self.game = ZeroSumSequenceGame(self.Iu, self.Iv, 
                                self.Au, self.Av, np.zeros((self.usize, self.vsize)))

        c_I = (self.game.Iu, self.game.Iv)
        c_A = (self.game.Au, self.game.Av)
        p_I = (self.game.Par_inf_u, self.game.Par_inf_v)
        p_A = (self.game.Par_act_u, self.game.Par_act_v)
        self.prep_struct = util.MakeGameStructure(c_I, c_A, p_I, p_A)

        self.fixed_P = fixed_P
        if self.fixed_P is None:
            self.use_default_P = False
        else:
            self.use_default_P = True

        super(ZSGFOMSolver, self).__init__()

    def forward(self, input, lambdu=None, lambdv=None):
        # TODO we seem to be reusing storage in self.input and stored_P

        '''
        if input is not None or input.size() == 0:
            print(input)
            input_np = input.numpy()
            self.use_default_P = False
        else:
            self.use_default_P = True
        '''

        input_np = input.numpy()

        batchsize = input_np.shape[0]

        # Lambda gradiens
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

        U = np.zeros([batchsize, self.usize], dtype=np.float64)
        V = np.zeros([batchsize, self.vsize], dtype=np.float64)
        U_beh = np.zeros([batchsize, self.usize], dtype=np.float64)
        V_beh = np.zeros([batchsize, self.vsize], dtype=np.float64)

        self.stored_P, self.stored_lambds = [], [] 

        for i in range(batchsize):

            if not self.single_feature or i == 0:
                if not self.use_default_P:
                    p = input_np[i,:,:]
                else:
                    p = self.fixed_P
                # p = np.squeeze(input_np[i, :, :]) # OLD
                # game = ZeroSumSequenceGame(self.Iu, self.Iv, 
                #                       self.Au, self.Av, p)
                # self.stored_games.append(game)
                self.game.SetP(p)
                self.stored_P.append(p)
                '''
                print('shape', lambdu_np.shape, lambdv_np.shape)
                print(lambdu_np[i, :].shape)
                print(lambdv_np[i, :].shape)
                # print(np.squeeze(lambdv_np[i, :], axis=0).shape)
                '''
                # self.stored_lambds.append((np.squeeze(lambdu_np[i,:]), np.squeeze(lambdv_np[i,:])))
                self.stored_lambds.append((lambdu_np[i,:], lambdv_np[i,:]))
                # lambd_np = [np.squeeze(lambdu_np[i,:]), np.squeeze(lambdv_np[i,:])]
                lambd_np = [lambdu_np[i,:], lambdv_np[i,:]]
                u, v, _ = self.solver.Solve(g=self.game, lambd=lambd_np, tau=self.forward_tau,
                                            prep_struct=self.prep_struct, return_only_seq=False)

                u, u_beh = u
                v, v_beh = v
                U[i, :], V[i, :] = u, v
                U_beh[i, :], V_beh[i, :] = u_beh, v_beh
                # print(V_beh[i, :])
            else: 
                # reuse old result when input features 
                # are guaranteed to be the across items in minibatch
                U[i, :] , V[i, :] = u, v
                U_beh[i, :], V[i, :] = u_beh, v_beh

        Ut, Vt = torch.DoubleTensor(U), torch.DoubleTensor(V)
        self.input, self.U, self.V = input_np, U, V
        self.U_beh, self.V_beh = U_beh, V_beh

        # IMPORTANT NOTE: save for backward does not work unless 
        # static method! Memory accumulates and leaks
        # self.save_for_backward(input, U, V)

        return Ut, Vt

    def backward(self, grad_u, grad_v):
        need_P_grad = not self.use_default_P

        batchsize = grad_u.shape[0]
        P, U, V = self.input, self.U, self.V

        if need_P_grad:
            dP = np.zeros([batchsize, self.usize, self.vsize])
        else: dP = None

        if self.need_lambd_grad_u or self.need_lambd_grad_v:
            dlambdu = np.zeros([batchsize, len(self.Iu)])
            dlambdv = np.zeros([batchsize, len(self.Iv)])

        for i in range(batchsize):
            u, v = U[i, :], V[i, :]

            if self.use_default_P == False:
                p = P[i, :, :]
            else:
                p = self.fixed_P

            gu, gv = grad_u[i, :].numpy(), grad_v[i, :].numpy()

            # Either compute or retrieve backward solve solution.
            # If single_feature is true, we do not store data, so use first index
            g = self.game
            if self.single_feature:
                if self.use_inverse_back:
                    # g = self.stored_games[0]
                    # g.SetP(self.stored_P[0]) # MAYBE USE THIS
                    g.SetP(p)
                    if i == 0:
                        back_solved = self.solver.BackwardSolveLU (p, u, v,
                            self.stored_lambds[0][0],
                            self.stored_lambds[0][1], 
                            g = g)
                    L = -np.concatenate([gu, gv, np.zeros(len(self.Iu)+len(self.Iv))])
                    d = back_solved.solve(np.atleast_2d(L).T)
                    d = (d[:self.usize].reshape([-1]), d[self.usize:(self.usize+self.vsize)].reshape([-1]))
                else:
                    # g.SetP(self.stored_P[0]) # MAYBE USE THIS
                    g.SetP(p)
                    d = self.solver.BackwardsSolve(u, v, gu, gv, tau=self.backward_tau,
                        g = g, lambd=self.stored_lambds[0], prep_struct=self.prep_struct,
                        u_beh = self.U_beh[i, :], v_beh = self.V_beh[i, :],
                        use_shortcut=True)
            else:
                # g.SetP(self.stored_P[i]) # MAYBE USE THIS
                g.SetP(p)
                d = self.solver.BackwardsSolve(u, v, gu, gv,
                    g = g, lambd=self.stored_lambds[i], prep_struct=self.prep_struct,
                    u_beh = self.U_beh[i, :], v_beh = self.V_beh[i, :], tau=self.backward_tau,
                    use_shortcut=True)

            # From: https://discuss.pytorch.org/t/why-input-is-tensor-in-the-forward-function-when-extending-torch-autograd/9039
            # Returning gradients for inputs that don't require it is
            # not an error.


            # Compute gradients of P.
            du, dv = d[0], d[1]

            if self.needs_input_grad[0]:
                dp = np.outer(du, v) + np.outer(u, dv)
                dP[i, :, :] = dp

            '''
            print(np.max(dv), np.min(dv))
            assert False, 'haha'
            '''

            '''
            print(du)
            print(dv)
            print(self.U_beh.size)
            
            '''
            # Compute gradients of lambda.
            if self.need_lambd_grad_v or self.need_lambd_grad_u:
                dlambdu[i, :], dlambdv[i, :] = prox_solver_back.gradLambda(u, v,
                        None, None, None, None,
                        du, dv,
                        prep_struct=self.prep_struct,
                        u_beh=self.U_beh[i, :], v_beh= self.V_beh[i, :],
                        use_shortcut=True)
            """
            print(max(dlambdv[i, :].flatten()),
                  min(dlambdv[i, :].flatten()),
                  max(v),
                  min(v),
                  min(self.stored_lambds[i][1].squeeze()),
                  max(self.stored_lambds[i][1].squeeze()))
            """

            # print(max(dlambdv), min(dlambdv))

            '''
            # THIS HAS A BUG!!!!!!!!!!!!!!!!!!!!! PROBABLY NUMERICAL...
            # Old method of updating that does *NOT* use indexing
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
                    z += dv[g.Par_inf_v[vi_id]]

                ssum = 0.
                for q in g.Iv[vi_id]:
                    # z += dv[q] * (1. + np.log(v[q]/para_prob))
                    z += dv[q] * (1. + np.log(v[q]) - np.log(para_prob))
                    ssum += v[q]
                # print('a', ssum - para_prob, 'b', ssum)
                
                dlambdv[i, vi_id] = -z
            '''

            '''
            # New method that *does* use numpy indexing.
            # For U 
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
            '''

        # Should be 0 at convergence
        if self.need_lambd_grad_u == False:
            dlambdu = None
        else:
            dlambdu = torch.DoubleTensor(dlambdu)

        if self.need_lambd_grad_v == False:
            dlambdv = None
        else:
            dlambdv = torch.DoubleTensor(dlambdv)

        if self.needs_input_grad[0]:
            dP = torch.DoubleTensor(dP)
        else: dP = None

        return dP, dlambdu, dlambdv


"""
class OneCardPokerPaynet(nn.Module):
    def __init__(self, nCards, 
                tie_initial_raiseval=False, 
                uniform_dist=True, 
                initial_params=None, 
                fixbets=None, 
                verify='kkt', 
                single_feature=False, 
                dist_type='original'):
        '''
        :param dist_type affects what type of distribution transform we use. 
            -original for -1 version
            -WNHD for hypergeometric.
            For dist_type='WNHD', we apply a softmax after the 
            linear transform to ensure a valid distribution
        '''

        super(OneCardPokerPaynet, self).__init__()
        self.nCards = nCards
        self.tie_initial_raiseval = tie_initial_raiseval 
        self.uniform_dist = uniform_dist
        self.dist_type = dist_type 
        if tie_initial_raiseval:
            self.fc = nn.Linear(1, 1, bias=False)
        else:
            self.fc = nn.Linear(1, 2, bias=False)
        if not uniform_dist:
            self.fc_dist = nn.Linear(1, self.nCards, bias=False)

        # Initial parameters
        if initial_params is not None:
            if self.tie_initial_raiseval:
                self.fc_dist.weight = torch.nn.Parameter(
                        torch.from_numpy(np.atleast_2d(initial_params[1:]).T))
                self.fc.weight = torch.nn.Parameter(
                        torch.from_numpy(np.atleast_2d(initial_params[0:1]).T)) 
            else: # both ante and bets are to be learnt, 
                # these are parameters 0 and 1
                self.fc_dist.weight = torch.nn.Parameter(
                        torch.from_numpy(np.atleast_2d(initial_params)[2:].T))
                self.fc.weight = torch.nn.Parameter(
                        torch.from_numpy(np.atleast_2d(initial_params)[:2].T)) 

        # initialize and fix bets. Note this OVERWRITES what is in initial_params
        if fixbets is not None:
            if self.tie_initial_raiseval:
                self.fc.weight = torch.nn.Parameter(
                        torch.from_numpy(np.atleast_2d([fixbets]).T)) 
            else: # both ante and bets are to be learnt, 
                # these are parameters 0 and 1
                self.fc.weight = torch.nn.Parameter(
                        torch.from_numpy(np.atleast_2d([fixbets, fixbets]).T)) 

            for param in self.fc.parameters(): param.requires_grad = False

        self.Iu, self.Iv, self.Au, self.Av = OneCardPokerComputeSets(nCards)
        self.solver = ZSGFOMSolver(self.Iu, self.Iv, self.Au, self.Av, 
                            verify=verify, single_feature=single_feature)

        if self.dist_type == 'original':
            self.transform = OneCardPokerTransform_v2(nCards)
        elif self.dist_type == 'WNHD':
            self.transform = OneCardPokerWNHDTransform(nCards)
            self.softmax = torch.nn.Softmax()
        else: assert False, 'Invalid dist type'

    def forward(self, x):
        nbatchsize = x.size()[0]
        j = self.fc(x) # 2-vector in initial and raising values
        if self.tie_initial_raiseval:
            j = torch.cat([j, j], dim=1)

        if self.dist_type=='original':
            if not self.uniform_dist:
                # unnormalized distribution of cards
                r = 1 + torch.abs(self.fc_dist(x))             
            else:
                r = Variable(torch.DoubleTensor(np.ones([nbatchsize, self.nCards])))
        elif self.dist_type=='WNHD':
            r = self.softmax(self.fc_dist(x))

        P = self.transform(torch.cat([j, r], dim=1))
        u, v = self.solver(P)

        return u, v, P, j, r


class OneCardPokerTransform(nn.Module):
    '''
    Transform parameters into payoffs for one-card-poker
    Rules: uniform distribution with *KNOWN* ordinal ranking
    Parameters: 
        - initial: amount that players are forced to place in the pot before 
            you even get the cards 
        - raiseval: amount raised for min (1st) player in first decision 
            (and the second player needs to meet if he does not fold).
            If raiseval is constrained to be equal to initial, then this 
            is equivalent to one-card poker with learning the lambda parameter.
    '''

    def __init__(self, nCards):
        '''
        :param nCard number of unique cards
        '''
        super(OneCardPokerTransform, self).__init__()
        self.nCards = nCards
        _, self.imat, self.rmat = OneCardPokerComputeP(nCards)
        self.imat = Variable(torch.DoubleTensor(self.imat), requires_grad=False)
        self.rmat = Variable(torch.DoubleTensor(self.rmat), requires_grad=False)
    

    def forward(self, param):
        '''
        :param param[0]: initial, param[1]: raiseval
        '''
        batchsize = param.size()[0]
        initial, raiseval = param[:, 0], param[:, 1]
        
        normalizing_factor = 1./self.nCards/(self.nCards-1) # Probability (for chance nodes of each path)
        '''
        cmpmat = np.zeros([self.nCards, self.nCards])
        cmpmat[np.triu_indices(self.nCards)] += 1.
        cmpmat[np.tril_indices(self.nCards)] -= 1.
        '''

        P = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards*4, self.nCards*4])))

        # TODO: Vectorize, do not loop over each entry!
        for i in range(batchsize):
            '''
            # 1st player waits, second player waits -- compare cards
            P[np.ix_(i, range(0,self.nCards*2,2), range(0,self.nCards*2,2))] = cmpmat * initial
            # 1st player waits, second player raises, first player waits(folds) (1)
            P[np.ix_(i, range(self.nCards*2, self.nCards*4, 2), range(1, self.nCards*2, 2))] = initial
            # 1st player waits, second player raises, first follows (+-2)
            P[np.ix_(i, range(self.nCards*2+1,self.nCards*4,2),range(1,self.nCards*2,2))] = cmpmat * (initial+raiseval)
            # First player raises, 2nd player either forfeits (-1) or fights (+-2). 0's are for impossible scenarios
            P[np.ix_(i, range(1,self.nCards*2,2), range(self.nCards*2,self.nCards*4,2))] = -initial # 2nd player forfeits after first player leaves
            P[np.ix_(i, range(1,self.nCards*2,2), range(self.nCards*2+1,self.nCards*4,2))] = cmpmat * (initial+raiseval)
            '''
            P[i, :, :] = self.imat * initial[i].expand_as(self.imat) + self.rmat * (initial[i]+raiseval[i]).expand_as(self.rmat)
            # P[i, :, :] += self.rmat * (initial[i]+raiseval[i]).expand_as(self.rmat)
        
        P *= normalizing_factor
       
        return P


class OneCardPokerTransform_v2(nn.Module):
    '''
    Transform parameters into payoffs for one-card-poker
    Rules: uniform distribution with *KNOWN* ordinal ranking
    Parameters: 
        - initial: amount that players are forced to place in the pot before 
            you even get the cards 
        - raiseval: amount raised for min (1st) player in first decision 
            (and the second player needs to meet if he does not fold).
            If raiseval is constrained to be equal to initial, then this 
            is equivalent to one-card poker with learning the lambda parameter.
        - carddist: vector of unnormalized card distributions
    '''

    def __init__(self, nCards):
        '''
        :param nCard number of unique cards
        '''
        super(OneCardPokerTransform_v2, self).__init__()
        self.nCards = nCards
    
        self.cmpmat = np.zeros([nCards, nCards])
        self.cmpmat[np.triu_indices(nCards)] += 1.
        self.cmpmat[np.tril_indices(nCards)] -= 1.


    def forward(self, param):
        '''
        :param param[0]: initial
        :param param[1]: raiseval
        :param param[2:]: carddist
        '''
        batchsize = param.size()[0]
        initial, raiseval = param[:, 0], param[:, 1]
        traiseval = initial + raiseval
        initial_m = initial.unsqueeze(-1).unsqueeze(-1).expand([batchsize, self.nCards, self.nCards])
        traiseval_m = traiseval.unsqueeze(-1).unsqueeze(-1).expand([batchsize, self.nCards, self.nCards])
        carddist = param[:, 2:]
        total_cards = carddist.sum(1)
        normalizing_constant = total_cards * (total_cards - 1)

        # Unnormalized probability matrices
        outer = torch.bmm(carddist.unsqueeze(2), carddist.unsqueeze(1))
        correction = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards, self.nCards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            correction[i, :, :] = torch.diag(carddist[i, :])

        probability_matrix = (outer-correction)/normalizing_constant.expand([self.nCards, self.nCards, batchsize]).transpose(0,2)
        '''
        probability_matrix[i, :, :] -= torch.diag(carddist[i, :])
        probability_matrix /= normalizing_constant.unsqueeze(-1).expand([batchsize, self.nCards, self.nCards])
        '''

        P = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards*4, self.nCards*4])))
        
        vcmpmat = Variable(torch.DoubleTensor(self.cmpmat).unsqueeze(0).expand([batchsize, self.nCards, self.nCards]), requires_grad=False)
        P[:, slice(0,self.nCards*2,2), slice(0, self.nCards*2,2)] = vcmpmat * probability_matrix * initial_m
        P[:, slice(self.nCards*2, self.nCards*4, 2), slice(1, self.nCards*2, 2)] = probability_matrix * initial_m
        P[:, slice(1, self.nCards*2,2), slice(self.nCards*2, self.nCards*4,2)] = -probability_matrix * initial_m 
        P[:, slice(self.nCards*2+1, self.nCards*4,2),slice(1, self.nCards*2,2)] = vcmpmat * probability_matrix * traiseval_m
        P[:, slice(1, self.nCards*2,2), slice(self.nCards*2+1, self.nCards*4,2)] = vcmpmat * probability_matrix * traiseval_m

        '''
        # TODO: Vectorize, do not loop over each entry!
        for i in range(batchsize):
            # Partial payoff matrix for portions where payoff is only the initial
            initial_mat = np.zeros([nCards*4, nCards*4])
            # 1st player waits, second player waits -- compare cards
            initial_mat[np.ix_(range(0,nCards*2,2), range(0, nCards*2,2))] = cmpmat * probability_matrix
            # 1st player waits, second player raises, first player waits(folds) (1)
            initial_mat[np.ix_(range(nCards*2, nCards*4, 2), range(1, nCards*2, 2))] = np.ones([nCards, nCards]) * probability_matrix
            # First player raises, 2nd player forfeits (-1)
            initial_mat[np.ix_(range(1, nCards*2,2), range(nCards*2, nCards*4,2))] = -np.ones([nCards, nCards]) * probability_matrix             

            # Partial payoff matrix for portions where payoff is initial+raiseval
            raise_mat = np.zeros([nCards*4, nCards*4])
            # 1st player waits, second player raises, first follows (+-2)
            raise_mat[np.ix_(range(nCards*2+1, nCards*4,2),range(1, nCards*2,2))] = cmpmat * probability_matrix
            # First player raises, 2nd player fights (+-2). 
            raise_mat[np.ix_(range(1, nCards*2,2), range(nCards*2+1, nCards*4,2))] = cmpmat * probability_matrix

            P[i, :, :] = self.imat * initial[i].expand_as(self.imat) + self.rmat * (initial[i]+raiseval[i]).expand_as(self.rmat)
            # P[i, :, :] += self.rmat * (initial[i]+raiseval[i]).expand_as(self.rmat)
        
        P *= normalizing_factor
        '''

        return P


class OneCardPokerWNHDTransform(nn.Module):
    '''
    Transform parameters into payoffs for one-card-poker.
    Transformation is done similar to the Wallenius' noncentral hypergeometric distribution.
    Rules: *KNOWN* ordinal ranking
    Parameters: 
        - initial: amount that players are forced to place in the pot before 
            you even get the cards 
        - raiseval: amount raised for min (1st) player in first decision 
            (and the second player needs to meet if he does not fold).
            If raiseval is constrained to be equal to initial, then this 
            is equivalent to one-card poker with learning the lambda parameter.
        - carddist: vector of *normalized* card distributions
    '''

    def __init__(self, nCards):
        '''
        :param nCard number of unique cards
        '''
        super(OneCardPokerWNHDTransform, self).__init__()
        self.nCards = nCards
    
        self.cmpmat = np.zeros([nCards, nCards])
        self.cmpmat[np.triu_indices(nCards)] += 1.
        self.cmpmat[np.tril_indices(nCards)] -= 1.


    def forward(self, param):
        '''
        :param param[0]: initial
        :param param[1]: raiseval
        :param param[2:]: carddist
        '''
        batchsize = param.size()[0]
        initial, raiseval = param[:, 0], param[:, 1]
        traiseval = initial + raiseval
        initial_m = initial.unsqueeze(-1).unsqueeze(-1).expand([batchsize, self.nCards, self.nCards])
        traiseval_m = traiseval.unsqueeze(-1).unsqueeze(-1).expand([batchsize, self.nCards, self.nCards])
        carddist = param[:, 2:]

        # Unnormalized probability matrices
        outer = torch.bmm(carddist.unsqueeze(2), carddist.unsqueeze(1))
        correction = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards, self.nCards])))
        normalizing_constant = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards, self.nCards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            correction[i, :, :] = torch.diag(carddist[i, :]) ** 2
            normalizing_constant[i, :, :] = torch.diag(1./(1.0 - carddist[i, :]))

        probability_matrix = torch.bmm(normalizing_constant, outer-correction)
        
        P = Variable(torch.DoubleTensor(np.zeros([batchsize, self.nCards*4, self.nCards*4])))
        
        vcmpmat = Variable(torch.DoubleTensor(self.cmpmat).unsqueeze(0).expand([batchsize, self.nCards, self.nCards]), requires_grad=False)
        P[:, slice(0,self.nCards*2,2), slice(0, self.nCards*2,2)] = vcmpmat * probability_matrix * initial_m
        P[:, slice(self.nCards*2, self.nCards*4, 2), slice(1, self.nCards*2, 2)] = probability_matrix * initial_m
        P[:, slice(1, self.nCards*2,2), slice(self.nCards*2, self.nCards*4,2)] = -probability_matrix * initial_m 
        P[:, slice(self.nCards*2+1, self.nCards*4,2),slice(1, self.nCards*2,2)] = vcmpmat * probability_matrix * traiseval_m
        P[:, slice(1, self.nCards*2,2), slice(self.nCards*2+1, self.nCards*4,2)] = vcmpmat * probability_matrix * traiseval_m

        return P


def OneCardPokerComputeP(nCards, initial=1., raiseval=1.):
    normalizing_factor = 1./nCards/(nCards-1) # Probability (for chance nodes of each path)
    cmpmat = np.zeros([nCards, nCards])
    cmpmat[np.triu_indices(nCards)] += 1.
    cmpmat[np.tril_indices(nCards)] -= 1.

    # Partial payoff matrix for portions where payoff is only the initial
    initial_mat = np.zeros([nCards*4, nCards*4])
    # 1st player waits, second player waits -- compare cards
    initial_mat[np.ix_(range(0,nCards*2,2), range(0, nCards*2,2))] = cmpmat
    # 1st player waits, second player raises, first player waits(folds) (1)
    initial_mat[np.ix_(range(nCards*2, nCards*4, 2), range(1, nCards*2, 2))] = np.ones([nCards, nCards]) - np.eye(nCards)
    # First player raises, 2nd player forfeits (-1)
    initial_mat[np.ix_(range(1, nCards*2,2), range(nCards*2, nCards*4,2))] = -(np.ones([nCards, nCards]) - np.eye(nCards)) # 2nd player forfeits after first player leaves
    
    # Partial payoff matrix for portions where payoff is initial+raiseval
    raise_mat = np.zeros([nCards*4, nCards*4])
    # 1st player waits, second player raises, first follows (+-2)
    raise_mat[np.ix_(range(nCards*2+1, nCards*4,2),range(1, nCards*2,2))] = cmpmat
    # First player raises, 2nd player fights (+-2). 
    raise_mat[np.ix_(range(1, nCards*2,2), range(nCards*2+1, nCards*4,2))] = cmpmat
    
    full_mat = initial_mat * initial + raise_mat * (raiseval + initial)
    full_mat *= normalizing_factor
    return full_mat, initial_mat, raise_mat


def OneCardPokerComputeP_v2(nCards, initial=1., raiseval=1., card_distribution=None):
    '''
    :param dist array-like *UNNORMALIZED* distribution of cards. Required to be >= 1.0 for each element.
    '''
    if card_distribution is None:
        card_distribution = [1. for i in range(nCards)] # uniform by default
    dist = np.array(card_distribution)
    assert(np.all(dist >= 1.))
    total_cards = np.sum(dist)
    normalizing_constant = total_cards * (total_cards - 1)
    probability_matrix = (np.outer(dist, dist) - np.diag(dist)) / normalizing_constant

    cmpmat = np.zeros([nCards, nCards])
    cmpmat[np.triu_indices(nCards)] += 1.
    cmpmat[np.tril_indices(nCards)] -= 1.

    # Partial payoff matrix for portions where payoff is only the initial
    initial_mat = np.zeros([nCards*4, nCards*4])
    # 1st player waits, second player waits -- compare cards
    initial_mat[np.ix_(range(0,nCards*2,2), range(0, nCards*2,2))] = cmpmat * probability_matrix
    # 1st player waits, second player raises, first player waits(folds) (1)
    initial_mat[np.ix_(range(nCards*2, nCards*4, 2), range(1, nCards*2, 2))] = np.ones([nCards, nCards]) * probability_matrix
    # First player raises, 2nd player forfeits (-1)
    initial_mat[np.ix_(range(1, nCards*2,2), range(nCards*2, nCards*4,2))] = -np.ones([nCards, nCards]) * probability_matrix # 2nd player forfeits after first player leaves
    
    # Partial payoff matrix for portions where payoff is initial+raiseval
    raise_mat = np.zeros([nCards*4, nCards*4])
    # 1st player waits, second player raises, first follows (+-2)
    raise_mat[np.ix_(range(nCards*2+1, nCards*4,2),range(1, nCards*2,2))] = cmpmat * probability_matrix
    # First player raises, 2nd player fights (+-2). 
    raise_mat[np.ix_(range(1, nCards*2,2), range(nCards*2+1, nCards*4,2))] = cmpmat * probability_matrix
    
    full_mat = initial_mat * initial + raise_mat * (raiseval + initial)
    return full_mat, initial_mat, raise_mat
    

def OneCardPokerWNHDComputeP(nCards, initial=1., raiseval=1., card_distribution=None):
    '''
    :param dist array-like distribution of cards. Should sum to 1
    '''
    if card_distribution is None:
        card_distribution = [1./nCards for i in range(nCards)] # uniform by default
    dist = np.array(card_distribution)
    total_cards = np.sum(dist)
    probability_matrix = np.dot(np.diag(1./(1.-dist)), np.outer(dist, dist) - np.diag(dist**2))

    cmpmat = np.zeros([nCards, nCards])
    cmpmat[np.triu_indices(nCards)] += 1.
    cmpmat[np.tril_indices(nCards)] -= 1.

    # Partial payoff matrix for portions where payoff is only the initial
    initial_mat = np.zeros([nCards*4, nCards*4])
    # 1st player waits, second player waits -- compare cards
    initial_mat[np.ix_(range(0,nCards*2,2), range(0, nCards*2,2))] = cmpmat * probability_matrix
    # 1st player waits, second player raises, first player waits(folds) (1)
    initial_mat[np.ix_(range(nCards*2, nCards*4, 2), range(1, nCards*2, 2))] = np.ones([nCards, nCards]) * probability_matrix
    # First player raises, 2nd player forfeits (-1)
    initial_mat[np.ix_(range(1, nCards*2,2), range(nCards*2, nCards*4,2))] = -np.ones([nCards, nCards]) * probability_matrix # 2nd player forfeits after first player leaves
    
    # Partial payoff matrix for portions where payoff is initial+raiseval
    raise_mat = np.zeros([nCards*4, nCards*4])
    # 1st player waits, second player raises, first follows (+-2)
    raise_mat[np.ix_(range(nCards*2+1, nCards*4,2),range(1, nCards*2,2))] = cmpmat * probability_matrix
    # First player raises, 2nd player fights (+-2). 
    raise_mat[np.ix_(range(1, nCards*2,2), range(nCards*2+1, nCards*4,2))] = cmpmat * probability_matrix
    
    full_mat = initial_mat * initial + raise_mat * (raiseval + initial)
    return full_mat, initial_mat, raise_mat

def OneCardPokerComputeSets(nCards):
    # Actions: 0-7: Even is for no raise, odd is for raise
    # Second stage: Next 4 information states (the previous action was to bet 0, second player raises)
    # Actions: 8-15: Even is for fold, odd is for raise
    # Iu = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]])
    Iu = np.array([[i*2, i*2+1] for i in range(nCards*2)])
    # 8 information states, 4 \times 2 (whether 1st player betted)
    # First 4 information states are for 1st player passing, next 4 if first player raise
    # Infostates 0-3 are for cards 0-3 respectively, 4-7 similarly
    # Even action is for pass, odd for raise
    # Iv = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]) 
    Iv = np.array([[i*2, i*2+1] for i in range(nCards*2)])
    # Au = np.array([[4], [], [5], [], [6], [], [7], [], [], [], [], [], [], [], [], []])
    Au = [[] for i in range(nCards*4)]
    for i in range(nCards): Au[i*2] = [i + nCards]
    Au = np.array(Au)
    Av = np.array([[] for x in range(nCards*4)])

    return Iu, Iv, Au, Av
    

def OneCardPokerSeqSample(nCards, U, V):
    Iu, Iv, Au, Av = OneCardPokerComputeSets(nCards)
    perm = np.random.permutation(range(nCards))
    uCard, vCard = perm[0], perm[1]
    uRet, vRet = np.array([0] * (nCards * 4)), np.array([0] * (nCards * 4))
    
    # Naively simulate random actions according to the behavioral strategy induced by sequence form
    uInfoSet = uCard
    uActionsAvail = Iu[uInfoSet]
    uFirstAction = np.random.choice(uActionsAvail, p=np.squeeze(U[uActionsAvail]))
    uFirstRaise = True if uFirstAction % 2 == 1 else False

    vInfoSet = nCards * (1 if uFirstRaise else 0) + vCard
    vActionsAvail = Iv[vInfoSet]
    vFirstAction = np.random.choice(vActionsAvail, p=np.squeeze(V[vActionsAvail]))
    vFirstRaise = True if vFirstAction % 2 == 1 else False

    if uFirstRaise == vFirstRaise: # Game is over, either both fold or both raise
        uRet[uFirstAction], vRet[vFirstAction] = 1.0, 1.0
        return uFirstAction, vFirstAction

    if uFirstRaise == True and vFirstRaise == False: # Game is over, first player raise, second fold
        uRet[uFirstAction], vRet[vFirstAction] = 1.0, 1.0
        return uFirstAction, vFirstAction

    # At this stage, first player did not raise by second raised.
    uInfoSet = nCards + uCard
    uActionsAvail = Iu[uInfoSet]
    uProbs = U[uActionsAvail]
    uProbs = uProbs / np.sum(uProbs) # Normalize to probability vector
    uSecondAction = np.random.choice(uActionsAvail, p=np.squeeze(uProbs))
    uRet[uSecondAction], vRet[vFirstAction] = 1.0, 1.0
    return uSecondAction, vFirstAction


def OneCardPokerSeqSample_v2(nCards, U, V, dist=None):
    '''
    :param dist unormalized distribution of cards
    '''
    if dist is None:
        dist = np.array([1 for i in range(nCards)])
    Iu, Iv, Au, Av = OneCardPokerComputeSets(nCards)
    uCard = np.random.choice(range(nCards), p=dist/np.sum(dist))
    newdist = np.array(dist)
    newdist[uCard] -= 1
    vCard = np.random.choice(range(nCards), p = newdist/np.sum(newdist))
    uRet, vRet = np.array([0] * (nCards * 4)), np.array([0] * (nCards * 4))
    
    # Naively simulate random actions according to the behavioral strategy induced by sequence form
    uInfoSet = uCard
    uActionsAvail = Iu[uInfoSet]
    uFirstAction = np.random.choice(uActionsAvail, p=np.squeeze(U[uActionsAvail]))
    uFirstRaise = True if uFirstAction % 2 == 1 else False

    vInfoSet = nCards * (1 if uFirstRaise else 0) + vCard
    vActionsAvail = Iv[vInfoSet]
    vFirstAction = np.random.choice(vActionsAvail, p=np.squeeze(V[vActionsAvail]))
    vFirstRaise = True if vFirstAction % 2 == 1 else False

    if uFirstRaise == vFirstRaise: # Game is over, either both fold or both raise
        uRet[uFirstAction], vRet[vFirstAction] = 1.0, 1.0
        return uFirstAction, vFirstAction

    if uFirstRaise == True and vFirstRaise == False: # Game is over, first player raise, second fold
        uRet[uFirstAction], vRet[vFirstAction] = 1.0, 1.0
        return uFirstAction, vFirstAction

    # At this stage, first player did not raise by second raised.
    uInfoSet = nCards + uCard
    uActionsAvail = Iu[uInfoSet]
    uProbs = U[uActionsAvail]
    uProbs = uProbs / np.sum(uProbs) # Normalize to probability vector
    uSecondAction = np.random.choice(uActionsAvail, p=np.squeeze(uProbs))
    uRet[uSecondAction], vRet[vFirstAction] = 1.0, 1.0
    return uSecondAction, vFirstAction


def OneCardPokerWNHDSeqSample(nCards, U, V, dist=None):
    '''
    :param dist unormalized distribution of cards
    '''
    if dist is None:
        dist = np.array([1./nCards for i in range(nCards)])

    Iu, Iv, Au, Av = OneCardPokerComputeSets(nCards)
    uCard = np.random.choice(range(nCards), p=dist)
    newdist = np.array(dist)
    newdist[uCard] = 0.
    vCard = np.random.choice(range(nCards), p=newdist/np.sum(newdist))

    uRet, vRet = np.array([0] * (nCards * 4)), np.array([0] * (nCards * 4))
    # Naively simulate random actions according to the behavioral strategy induced by sequence form
    uInfoSet = uCard
    uActionsAvail = Iu[uInfoSet]
    uFirstAction = np.random.choice(uActionsAvail, p=np.squeeze(U[uActionsAvail]))
    uFirstRaise = True if uFirstAction % 2 == 1 else False

    vInfoSet = nCards * (1 if uFirstRaise else 0) + vCard
    vActionsAvail = Iv[vInfoSet]
    vFirstAction = np.random.choice(vActionsAvail, p=np.squeeze(V[vActionsAvail]))
    vFirstRaise = True if vFirstAction % 2 == 1 else False

    if uFirstRaise == vFirstRaise: # Game is over, either both fold or both raise
        uRet[uFirstAction], vRet[vFirstAction] = 1.0, 1.0
        return uFirstAction, vFirstAction

    if uFirstRaise == True and vFirstRaise == False: # Game is over, first player raise, second fold
        uRet[uFirstAction], vRet[vFirstAction] = 1.0, 1.0
        return uFirstAction, vFirstAction

    # At this stage, first player did not raise by second raised.
    uInfoSet = nCards + uCard
    uActionsAvail = Iu[uInfoSet]
    uProbs = U[uActionsAvail]
    uProbs = uProbs / np.sum(uProbs) # Normalize to probability vector
    uSecondAction = np.random.choice(uActionsAvail, p=np.squeeze(uProbs))
    uRet[uSecondAction], vRet[vFirstAction] = 1.0, 1.0
    return uSecondAction, vFirstAction


if __name__ == '__main__':
    # Training set (uniform training set)
    print('Testing for uniform card distribution')
    nCards=13
    P, _, _ = OneCardPokerComputeP(nCards=nCards, initial=1., raiseval=1.)
    print('Verifying that using general card distributions give same results')
    P2, _, _ = OneCardPokerComputeP_v2(nCards, initial=1., raiseval=1., 
            card_distribution=None)
    print('Difference between P and P2 should be 0:', np.sum(P-P2))
    print('-------')
    print('''Now, we verify that as the lambda parameter grows large,
            we get the correct result''')
    fac = 1000.
    P *= fac
    Iu, Iv, Au, Av = OneCardPokerComputeSets(nCards)
    zssg = ZeroSumSequenceGame(Iu, Iv, Au, Av, P)
    solver = QRESeqLogit(zssg)
    ugt, vgt, _ = solver.Solve(alpha=0.1, beta=0.5, tol=10**-7, 
            epsilon=0., min_t=10**-25, max_it=10000, verify='kkt')
    print('Ground truths u:\n', ugt)
    print('Ground truths v:\n', vgt)
    print('''Payoff contributed after dividing again 
            by the addiitional lambda parameter''')
    print('''This should be the same as the theoretical result 
            (e.g. from Geoff Gordon\' webpage, 
            i.e. -0.64 if lambda is large enough''')
    print(zssg.ComputePayoff(ugt, vgt)[0]/fac)
    print('--------')
    print('We now sample actions randomly from these ground truths')
    nSamples = 100
    U = np.zeros((nSamples, )) # Action distributions
    V = np.zeros((nSamples, ))
    for i in range(nSamples):
        U[i], V[i] = OneCardPokerSeqSample(nCards, ugt, vgt)
    U = np.array(U, dtype=int)
    V = np.array(V, dtype=int)

    print('--------------------------------')
    print('Testing network, gradient and backprop')
    net = OneCardPokerPaynet(nCards)
    net = net.double()
    print(net)
    net.zero_grad()
    target_u = Variable(torch.LongTensor(U), requires_grad=False)
    target_v= Variable(torch.LongTensor(V), requires_grad=False)
    test_in = Variable(torch.DoubleTensor(np.ones([nSamples, 1])))
    out_u, out_v, out_P, out_bets, out_probs = net(test_in)
    
    print('NLL loss:')
    criterion = nn.NLLLoss()
    lossu = criterion(torch.log(out_u), target_u)
    lossv = criterion(torch.log(out_v), target_v)
    print('lossu %s, lossv %s' % (lossu, lossv))

    lossu.backward()
    
    print('Press any key to cont.')
    input()
    print('-------------------------------')
    print('Solving nonuniform poker')
    dist = np.array([i for i in reversed(range(1,nCards+1))], dtype=np.float64)
    # dist += 1000.
    # dist = [1 for i in range(nCards)]
    P, _, _ = OneCardPokerComputeP_v2(nCards, initial=1000., raiseval=1000., card_distribution=dist)
    zssg = ZeroSumSequenceGame(Iu, Iv, Au, Av, P)
    solver = QRESeqLogit(zssg)
    ugt, vgt, _ = solver.Solve(alpha=0.1, beta=0.5, tol=10**-7, epsilon=0., min_t=10**-25, max_it=10000, verify='kkt')
    print('Ground truths u:\n', ugt)
    print('Ground truths v:\n', vgt)

    print('-------------------------------')
    print('Testing sampling method for non-uniform stuff')
    U = np.zeros((nSamples, )) # Action distributions
    V = np.zeros((nSamples, ))
    for i in range(nSamples):
        U[i], V[i] = OneCardPokerSeqSample_v2(nCards, ugt, vgt, dist=dist)
    U = np.array(U, dtype=int)
    V = np.array(V, dtype=int)

    print('-------------------------------')
    print('Testing network v2 on non-uniform data, gradient and backprop')
    net = OneCardPokerPaynet(nCards)
    net = net.double()
    print(net)
    net.zero_grad()
    target_u = Variable(torch.LongTensor(U), requires_grad=False)
    target_v= Variable(torch.LongTensor(V), requires_grad=False)
    test_in = Variable(torch.DoubleTensor(np.ones([nSamples, 1])))
    out_u, out_v, out_P, out_bets, out_probs = net(test_in)

    print('NLL loss:')
    criterion = nn.NLLLoss()
    lossu = criterion(torch.log(out_u), target_u)
    lossv = criterion(torch.log(out_v), target_v)
    print('lossu %s, lossv %s' % (lossu, lossv))

    print('All basic tests passed')
"""
