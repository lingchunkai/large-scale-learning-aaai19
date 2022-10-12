import numpy as np
import logging, sys, pickle, copy, os
from itertools import *
import itertools
import argparse, types
import h5py as h5

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ..experiments.experiments import GameDataset, GameDataset_, Evaluate
from ..core.game import ZeroSumSequenceGame
from ..core.fom_paynet import ZSGFOMSolver
from ..core.fom_solve import QRESeqFOMLogit
from ..core.solve import QRESeqLogit
from ..experiments.seq_datagen import dict_to_h5

logger = logging.getLogger(__name__)

uGT = [32.3,28.7,39.2]
vGT = [49.3,6.3,44.4]


def main(args=None):
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if args is None:
        logger.warning('No parameters found: training on default dataset')

        args=types.SimpleNamespace()
        args.mode='run'
        args.datasize=10000
        args.lr=0.002 # old
        # args.lr=0.02 # old
        # args.lr=0.0002
        # args.lr=0.00002
        # args.lr=0.05
        # args.lr = 10.**-18
        args.nEpochs=8000
        args.monitorFreq=5
        args.fracval=0.3
        args.batchsize=2000

        args.save_path = './default_results/default_soccer/'
        args.verify=False

        args.data_path='./data/soccer/all'

        args.momentum=0.9
        args.optimizer='sgd'
        # args.optimizer='adam'
        # args.optimizer='rmsprop'

    if args.mode == 'run':
        save_folder = args.save_path

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        logger.info('Training dataset: %s' % args.data_path)
        Train(args)

    elif args.mode == 'gen':
        assert False, 'No gen set yet'
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        f = h5.File(os.path.join(args.save_folder, args.save_name))

        gen = WNHDGen(nCards=args.nCards,
                      dist=args.dist,
                      scale=args.scale,
                      initial=args.initial_bet,
                      raiseval=args.raiseval,
                      seed=args.seed)
        d, _ = gen.GenData(args.nSamples)
        dict_to_h5(d, f)


def Train(args):
    data = GameDataset(loadpath=args.data_path, size=args.datasize, real=False)
    train_data, test_data = data, data.Split(frac=args.fracval, random=False)

    train_dl = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=1)
    test_dl = DataLoader(test_data, batch_size=test_data.__len__(), shuffle=False, num_workers=1)

    net = SoccerPaynet(nFeats=1, verify=args.verify)
    net = net.double()

    params = [x for x in net.parameters() if x.requires_grad]

    if args.optimizer == 'rmsprop':
        optimizer=optim.RMSprop(params, lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer=optim.Adam(params, lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer=optim.SGD(params,
                            lr=args.lr,
                            momentum=args.momentum,
                            nesterov=False)

    from time import gmtime, strftime
    curdatetime = strftime("%Y-%m-%d_%H-%M-%S___", gmtime())
    args.save_path = args.save_path + curdatetime

    # TODO: remove
    np.set_printoptions(threshold=np.nan)

    # log likelihood
    for i_batch, sample_batched in enumerate(test_dl):
        # Actual distribution
        # GTu, GTv = [82., 0., 92.,112.], [18., 0., 141., 127.]

        # Real nash distribution
        GTu, GTv = [57.99, 0., 98.34, 129.67], [31.48, 0., 117.54, 136.98]
        GTu, GTv = GTu/np.sum(GTu), GTv/np.sum(GTv)
        GTu[1] = GTu[2]+GTu[3]
        GTv[1] = GTv[2]+GTv[3]

        print('test Mean log probability', np.mean(np.log(GTu[sample_batched[1].numpy().astype(int)])) +
              np.mean(np.log(GTv[sample_batched[2].numpy().astype(int)])))

    '''
    # log likelihood
    for i_batch, sample_batched in enumerate(train_dl):
        GTu, GTv = np.array([0.4, 0.6, 0.2, 0.2, 0.2]), np.array([0.4, 0.6, 0.2, 0.2, 0.2])
        print('Mean log probability', np.mean(np.log(GTu[sample_batched[1].numpy().astype(int)])) +
              np.mean(np.log(GTv[sample_batched[2].numpy().astype(int)])))
    '''
    input()

    #  Training proper
    train_loss = []
    test_loss = []
    for i_epoch in range(args.nEpochs):
        train_loss_per_mb = []
        for i_batch, sample_batched in enumerate(train_dl):
            nll = nn.NLLLoss()
            optimizer.zero_grad()

            loss, lossu, lossv = Evaluate(net, sample_batched, 'logloss', optimizer, real=False)
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss_per_mb.append(loss.data.numpy())
            # print(loss)
            sys.stdout.write('|')
            sys.stdout.flush()
            # print(len(sample_batched[0]))

        # print('Peek at lambda:', lambds[0, :])

        train_loss.append(np.mean(np.array(train_loss_per_mb)))

        logger.info('Average train loss: %s' % (train_loss[-1]))

        if i_epoch % args.monitorFreq == 0:
            # tracker = SummaryTracker()
            print(net.fc_lambds.weight)
            for i_batch, sample_batched in enumerate(test_dl):

                optimizer.zero_grad()
                loss, lossu, lossv = Evaluate(net, sample_batched, 'mse', optimizer, real=False)
                # loglosstrend.append(loss.data.numpy())
                logger.info('Total validation log-loss: %s, U loss: %s, V loss: %s' % \
                            (loss.data.numpy(), lossu.data.numpy(), lossv.data.numpy()))

                loss, lossu, lossv = Evaluate(net, sample_batched, 'avg_success', optimizer, real=False)
                # loglosstrend.append(loss.data.numpy())
                # print(np.stack([sample_batched[2].numpy().astype(int), loss.data.numpy()]).T)
                print('Avg success')
                print(np.mean(loss.data.numpy()))

                '''
                # TODO: Accuracy for oneill
                other = {'pI': [[-1], pI],
                         'pA': [[0], pA],
                         'cI': [[[0]], cI],
                         'cA': [[[]], cA]}

                loss, lossu, lossv = Evaluate(net, sample_batched, 'accuracy', optimizer, real=True, others=other)
                # loglosstrend.append(loss.data.numpy())
                # print(np.stack([sample_batched[2].numpy().astype(int), loss.data.numpy()]).T)
                print('Mean accuracy')
                print(loss)

                # logger.info('Total validation accuracy: %s' % loss.data.numpy())
                '''

                '''
                test_loss.append(loss.data.numpy())
                logger.info('Average test loss: %s' % (test_loss[-1]))
                '''

            # fname = args.save_path + '%06d' % (i_epoch) + '.p'
            # pickle.dump(sto_dict, open( fname, 'wb'))


class SoccerPaynet(nn.Module):
    def __init__(self,
                 nFeats=[1],
                 verify=False):

        super(SoccerPaynet, self).__init__()

        self.cI = [[0, 1], [2, 3]]
        self.cA = [[], [1], [], []]
        self.pI = [-1, 1]
        self.pA = [0, 0, 1, 1]

        P = np.array([
            [-0.704, -1., -1.],
            [-0.902, -0.4, -0.968],
            [-1, -1, -0.746]]).astype(float)
        P_expanded = np.zeros((4,4))
        P_expanded[np.ix_([0,2,3], [0,2,3])] = P[np.ix_([1,0,2], [1,0,2])]
        # P_expanded[[0,2,3], [0,2,3]] = P[[1,0,2][1,0,2]]
        self.P = torch.Tensor(P_expanded)

        print('Number of information sets', len(self.pI))

        self.num_lambdas = 4
        self.nFeats = 1

        self.fc_lambds = nn.Linear(self.nFeats, self.num_lambdas, bias=False)
        self.fc_lambds.weight = torch.nn.Parameter(
            torch.from_numpy(-3 * np.ones([self.num_lambdas, self.nFeats])))

        #################################
        # Main
        #################################
        solve = QRESeqFOMLogit(tau=0.1, tol=10.**-5,
                               max_it=10000, Ptype='keep', verify=verify)

        # Set single_feature, use_inverse_back to true in order to use old method of inverse
        self.solver = ZSGFOMSolver(self.cI, self.cI, self.cA, self.cA, solver=solve,
                                   single_feature=True, use_inverse_back=True)

    def forward(self, x):
        nbatchsize = x.size()[0]

        j = torch.exp(self.fc_lambds(x)) + 10. ** -8

        Lu = j[:, 0:2]
        Lv = j[:, 2:4]

        # Payoff matrix
        P = Variable(self.P.expand(nbatchsize, self.P.size()[0], self.P.size()[1]))

        # Solve
        u, v = self.solver(P, Lu, Lv)
        print(u[0,:], v[0, :])

        return u, v, j

if __name__ == '__main__':
    main()
