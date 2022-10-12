import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import *

from ..core.game import ZeroSumSequenceGame 
from ..core.solve import QRESeqLogit
from ..core.fom_paynet import ZSGFOMSolver
from ..core.fom_solve import QRESeqFOMLogit
from ..core.seq_paynet import ZSGSeqSolver

from ..experiments.experiments import GameDataset, GameDataset_
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import sys
import time

import sacred
import faulthandler
faulthandler.enable()

ex = sacred.Experiment('poker_eval')

@ex.config
def config():
    args=dict()

    # args['mode']='train'

    args['solver_type'] = 'dense'
    args['solver_type'] = 'fom'

    args['min_lambda'] = 0.001

    args['loadpath'] = './data/fom_eval_poker.h5'

    args['optimizer'] = 'adam'

    args['datasize'] = 2000
    args['val_size'] = 1000

    args['nCards'] = 200
    args['batchsize']=64
    args['lr'] = 0.0001

    args['nEpochs'] = 20000

    args['monitor_freq'] = 5

    args['tau'] = [1., 0.1]
    args['tol'] = 10.**-8
    args['tol_scheme'] = 'const'
    args['tol_scheme'] = 'scaled'

    args['init_weights'] = [[0.000]]*4

    args['duality_freq'] = 5


class OneCardPokerRatPaynet(nn.Module):
    def __init__(self, nCards,
                verify='none',
                bet=1.0,
                single_feature=False,
                use_inverse_back=False,
                min_lambda=0.001,
                tol=10.**-8,
                tau=[1.,0.1],
                init_weights=[[0.000]]*4,
                duality_freq=5,
                mode = 'fom',
                 ):
        '''
        Inputs: 2 lambda parameters for lambda
        '''

        super(OneCardPokerRatPaynet, self).__init__()
        self.nCards = nCards
        self.P, _, _ = OneCardPokerComputeP(nCards, initial=bet, raiseval=bet)
        normalizing_factor = 1./self.nCards/(self.nCards-1) # Probability (for chance nodes of each path)
        if mode == 'sparse_fom': # used to be  a bug but i think this is faster?
            self.P = scipy.sparse.csr_matrix(self.P)
        self.fc = nn.Linear(1, 4, bias=False)
        self.fc.weight.data = torch.Tensor(-8*np.ones([4,1]))
        # self.fc.weight.data = torch.Tensor([[-5.20516678], [-4.94037811], [-5.11140076], [-5.21235404]])
        # self.fc.weight.data = torch.Tensor([[0.00548814],  [0.00715189],  [0.00602763],  [0.00544883]])
        self.fc.weight.data = torch.Tensor(init_weights)
        self.min_lambda=min_lambda

        self.Iu, self.Iv, self.Au, self.Av = OneCardPokerComputeSets(nCards)

        if mode == 'fom':
            self.solve = QRESeqFOMLogit(tau=1., tol=tol,
                                   max_it=10000, Ptype='keep', verify=verify, duality_freq=duality_freq)
            self.solver = ZSGFOMSolver(self.Iu, self.Iv, self.Au, self.Av,
                     solver=self.solve,
                     single_feature=single_feature,
                     use_inverse_back=use_inverse_back,
                     fixed_P = self.P,
                     tau_both=tau)
        elif mode == 'dense':
            self.g_nonsparse = ZeroSumSequenceGame(self.Iu, self.Iv, self.Au, self.Av, self.P, sp=False)
            self.solver = ZSGSeqSolver(self.Iu, self.Iv, self.Au, self.Av, verify=verify,
                         single_feature=single_feature,
                         fixed_P=self.P)

        self.tform = OneCardPokerRatTransform(self.nCards)

    def forward(self, x):
        batchsize = x.size()[0]
        dummy = Variable(torch.zeros((batchsize, 0)), requires_grad=False)
        # j = torch.exp(self.fc(x)) + self.min_lambda # 2-vector in i2 lambdas
        j = self.fc(x) + self.min_lambda
        lu, lv = self.tform(j)
        u, v = self.solver(dummy, lu, lv)

        return u, v

    def SetTol(self, tol):
        ''' for scaling tolerances'''
        self.solve.tol = tol # update solver, not solve!

class OneCardPokerRatTransform(nn.Module):
    '''
    '''
    def __init__(self, nCards):
        '''
        :param nCard number of unique cards
        '''
        super(OneCardPokerRatTransform, self).__init__()
        self.nCards = nCards
    

    def forward(self, param):
        batchsize = param.size()[0]

        lambdu = Variable(torch.zeros((batchsize, self.nCards*2)))
        lambdv = Variable(torch.zeros((batchsize, self.nCards*2)))

        lambdu[:, :self.nCards] = param[:, 0:1].repeat(1, self.nCards)
        lambdu[:, self.nCards:] = param[:, 1:2].repeat(1, self.nCards)
        lambdv[:, :self.nCards] = param[:, 2:3].repeat(1, self.nCards)
        lambdv[:, self.nCards:] = param[:, 3:4].repeat(1, self.nCards)



        return lambdu, lambdv


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
    Av = [[] for x in range(nCards*4)]

    return Iu, Iv, Au, Av


def OneCardPokerSeqSample(nCards, U, V, dist=None, r=None):
    '''
    :param dist unormalized distribution of cards
    '''
    if dist is None:
        dist = np.array([1./nCards for i in range(nCards)])

    if r is None:
        r = np.random

    Iu, Iv, Au, Av = OneCardPokerComputeSets(nCards)
    uCard = r.choice(range(nCards), p=dist)
    newdist = np.array(dist)
    newdist[uCard] = 0.
    vCard = r.choice(range(nCards), p=newdist/np.sum(newdist))

    uRet, vRet = np.array([0] * (nCards * 4)), np.array([0] * (nCards * 4))
    # Naively simulate random actions according to the behavioral strategy induced by sequence form
    uInfoSet = uCard
    uActionsAvail = Iu[uInfoSet]
    uFirstAction = r.choice(uActionsAvail, p=np.squeeze(U[uActionsAvail]))
    uFirstRaise = True if uFirstAction % 2 == 1 else False

    vInfoSet = nCards * (1 if uFirstRaise else 0) + vCard
    vActionsAvail = Iv[vInfoSet]
    vFirstAction = r.choice(vActionsAvail, p=np.squeeze(V[vActionsAvail]))
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
    uSecondAction = r.choice(uActionsAvail, p=np.squeeze(uProbs))
    uRet[uSecondAction], vRet[vFirstAction] = 1.0, 1.0
    return uSecondAction, vFirstAction


class PokerRatDataset(GameDataset):
    def __init__(self, loadpath, size=-1, offset=0, transform=lambda x: x):
        self.loadpath = loadpath
        super(PokerRatDataset, self).__init__(loadpath, size=size, offset=offset, real=True)
        self.transform = transform

    def __getitem__(self, idx):
        i = self.GetRawIndex(idx)
        return self.feats[i, :], self.Au[i], self.Av[i], self.f['U'][i, :], self. f['V'][i, :]

    def GetW(self):
        return self.f['W']

    def Split(self, frac=0.3, random=False):
        other = PokerRatDataset(self.loadpath, transform=self.transform)
        return GameDataset_.Split(self, frac=frac, random=random, other_object=other)


def run(args):

    # Fixed holdout set. datasize
    train_data = PokerRatDataset(args['loadpath'], size=args['datasize'])
    usize, vsize = args['nCards']*4, args['nCards']*4
    nfeatures = train_data.GetFeatureDimensions()

    test_data = PokerRatDataset(args['loadpath'], offset=-args['val_size'], size=args['val_size'])

    train_dl = DataLoader(train_data, batch_size=args['batchsize'], shuffle=True, num_workers=0)
    test_dl = DataLoader(test_data, batch_size=test_data.__len__(), shuffle=False, num_workers=0)

    # Setup net,  optimizers, and loss
    lr = args['lr']

    if args['solver_type'] == 'fom':
        net = OneCardPokerRatPaynet(args['nCards'],
                         verify='none',
                         bet=1.0,
                         single_feature=False,
                         use_inverse_back=False,
                         min_lambda=args['min_lambda'],
                         tol=args['tol'],
                         tau=args['tau'],
                         init_weights=args['init_weights'],
                         duality_freq=args['duality_freq'],
                         mode='fom'
                         )
    elif args['solver_type'] == 'dense':
        net = OneCardPokerRatPaynet(args['nCards'],
                                    verify='none',
                                    bet=1.0,
                                    single_feature=False,
                                    use_inverse_back=False,
                                    min_lambda=args['min_lambda'],
                                    tol=args['tol'],
                                    tau=args['tau'],
                                    init_weights=args['init_weights'],
                                    duality_freq=args['duality_freq'],
                                    mode='dense'
                                    )
    net = net.double()

    # require_grad filter is a hack for freezing bet sizes
    if args['optimizer'] == 'rmsprop':
        optimizer=optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    elif args['optimizer'] == 'adam':
        optimizer=optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    elif args['optimizer'] == 'sgd':
        optimizer=optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.momentum, nesterov=True)

    start_time = time.time()
    time_store = []
    train_loss_store = []
    val_loss_store = []

    for i_epoch in range(args['nEpochs']):
        print('epoch:', i_epoch)
        t_lossu, t_lossv, t_loss = [], [], []
        accum_loss = []

        optimizer.zero_grad()

        # Update tol if required
        if args['tol_scheme'] == 'scaled':
            tol = 10. ** -(((i_epoch * 2) // 10 ) + 2)
            if tol < args['tol']:
                tol = args['tol']
            print('tol:', tol)
            net.SetTol(tol)

        for i_batch, sample_batched in enumerate(train_dl):

            F, Au, Av, GTu, GTv = sample_batched
            feats = Variable(F.double())
            Av = Variable(Av.long(), requires_grad=False)
            Au = Variable(Au.long(), requires_grad=False)
            GTu = Variable(GTu.double(), requires_grad=False)
            GTv = Variable(GTv.double(), requires_grad=False)

            # print('forward')
            U, V = net(feats)
            # print('end forward')

            if True:
                nll = nn.NLLLoss()
                lossv = nll(torch.log(V), Av)
                lossu = nll(torch.log(U), Au)
                loss = lossu + lossv
            else:
                '''
                print(V)
                print(GTv)
                print('ugh')
                '''
                mse = nn.MSELoss()
                lossv = mse(V, GTv)
                lossu = mse(U, GTu)
                loss = lossu + lossv

            accum_loss.append(loss.data.numpy())


            #print('backward')
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            #print('end backward')
            optimizer.step()
            sys.stdout.write('|')
            sys.stdout.flush()

        cur_time = time.time() - start_time
        time_store.append(cur_time)

        avg_loss = np.mean(accum_loss)
        train_loss_store.append(avg_loss)
        print('')
        print('training_loss', avg_loss)
        print(net.fc.weight.data)

        if i_epoch % args['monitor_freq'] == 0:
            for i_batch, sample_batched in enumerate(test_dl):

                # Reset tol (temporarily) to real value for validation
                if args['solver_type'] == 'fom':
                    net.SetTol(args['tol'])

                F, Au, Av, GTu, GTv = sample_batched
                feats = Variable(F.double())
                Av = Variable(Av.long(), requires_grad=False)
                Au = Variable(Au.long(), requires_grad=False)
                U, V = net(feats)

                nll = nn.NLLLoss()

                lossv = nll(torch.log(V), Av)
                lossu = nll(torch.log(U), Au)
                loss = lossu + lossv
                val_loss_store.append(loss.data.numpy())
                print('val loss', loss)

                # Compute ideal
                nz = len(F)
                ideal_loss = 0.
                GTu_numpy = GTu.numpy()
                GTv_numpy = GTv.numpy()
                Au_numpy, Av_numpy = Au.data.numpy().astype(int), Av.data.numpy().astype(int)
                for k in range(nz):
                    ideal_loss += -np.log(GTu_numpy[k, Au_numpy[k]])
                    ideal_loss += -np.log(GTv_numpy[k, Av_numpy[k]])
                ideal_loss /= nz
                print('ideal_loss', ideal_loss)


                optimizer.zero_grad()

        print('end of epoch', i_epoch)
        print('time store', time_store)
        print('train loss', train_loss_store)
        print('val loss', val_loss_store)


@ex.automain
def main(args, _run):
    run(args)

