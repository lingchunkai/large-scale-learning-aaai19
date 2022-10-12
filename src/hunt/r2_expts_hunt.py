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

import matplotlib.pyplot as plt

import sacred
import faulthandler
faulthandler.enable()

ex = sacred.Experiment('r2_expts_hunt')

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# In the one-shot case, trialnums are split into 2 halves, first 11 and second 11.
feat_to_nums_onehot = {'const': 1, 'education': 4, 'trialnum': 2, 'age': 8, 'gender': 2}

# In the linear case, each parameter is linearized into a single number.
feat_to_nums_linear = {'const': 1, 'education': 1, 'age': 1, 'trialnum': 1, 'gender': 1}


@ex.config
def config():
    args=dict()

    args['mode']='train'

    ##############################################################
    # Data parameters
    ##############################################################
    args['exptType']='AA'
    # args.exptType='AB'
    args['exptObj']=2

    args['experiment_path'] = './default_results/default_hunt/'
    # args.use_inverse_back=True # not implemented yet

    # args.data_path='./data/default_hunt.h5'
    args['data_path']='./data/hunt/AA_obj2_s'
    # args.data_path='./data/hunt/AB_obj2_s'
    # args.data_path='./data/hunt/AA_obj2_only4moves'
    args['data_path']='./data/hunt/AA_obj2_s_seed'

    ##############################################################
    # Training setup
    ##############################################################
    args['datasize']=15000
    # args.datasize=2000
    # args.datasize=500
    args['lr']=0.00002 # old
    # args.lr=0.02 # old
    # args.lr=0.0002
    # args.lr=0.00002
    # args.lr=0.05
    # args.lr = 10.**-18
    args['nEpochs']=8000
    args['monitorFreq']=5
    args['fracval']=0.3
    args['batchsize']=64

    ##############################################################
    # Training setup
    ##############################################################
    args['test_datasize']=10000

    ##############################################################
    # Game solver settings
    ##############################################################
    args['verify']=False

    ##############################################################
    # Backprop optimizer settings
    ##############################################################
    args['momentum']=0.9
    # args.momentum=0.0
    # args.lr=0.00002
    args['optimizer']='sgd'
    args['optimizer']='adam'
    # args.optimizer='rmsprop'

    ##############################################################
    # Dataset-specific settings
    ##############################################################
    # args.parameterization_type = 'same'
    args['parameterization_type'] = 'num_open'
    # args.parameterization_type = 'all'
    # args.parameterization_type = 'sufficient'

    # args.feature_types = ['const', 'education', 'trialnum']
    args['feature_types'] = ['education', 'trialnum', 'age', 'gender']
    # args.feature_types = ['education', 'trialnum', 'age']
    # args.feature_types = ['const']

    # args.feature_types = ['trialnum']
    # args.feature_types = ['const', 'education']
    # args.feature_types = ['const']

    args['encoding'] = 'one_hot'
    # args.encoding = 'linear'

    # args['suff_type'] = 'simple'
    args['suff_type'] = 'diff'

    ##############################################################
    # Architecture
    ##############################################################
    # old
    # args.arch = types.SimpleNamespace()
    # args.arch.widths = []
    # args.arch.activations = ['relu'] * len(args.arch.widths)

    args['arch'] = dict()
    args['arch']['widths'] = [100, 100, 100]
    args['arch']['activations'] = ['relu'] * len(args['arch']['widths'])


def run(args, _run):
    # Legacy reasons
    args = types.SimpleNamespace(**args)
    args.arch = types.SimpleNamespace(**args.arch)

    print(args)

    if args.encoding == 'one_hot':
        args.nFeats = [feat_to_nums_onehot[x] for x in args.feature_types]
    elif args.encoding == 'linear':
        args.nFeats = [feat_to_nums_linear[x] for x in args.feature_types]

    if args.mode == 'train' or args.mode == 'resume':
        if len(_run.observers) > 0:
            fo = _run.observers[0]
            args.save_path = os.path.join(fo.dir, 'results')
            args.regular_path = os.path.join(args.save_path, 'regular')
            args.val_path = os.path.join(args.save_path, 'val')

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            if not os.path.exists(args.regular_path):
                os.makedirs(args.regular_path)
            if not os.path.exists(args.val_path):
                os.makedirs(args.val_path)

        if args.mode == 'train':
            logger.info('Training dataset: %s' % args.data_path)
            Train(args)
        elif args.mode == 'resume':
            logger.info('Resuming')
            logger.info('Data folder %s' % args.data_path)
            logger.info('Last saved path %s' % args.last_saved)

            checkpoint = torch.load(args.last_saved)
            Train(args, checkpoint)


    elif args.mode == 'test':
        Test(args)
    else:
        assert False, 'Unknown mode of operation %s' % args.mode


#####################################################################
# Testing Procedure
#####################################################################

def Test(args):

    tform = GetEncoding(args)

    dl = GetTestDataloaders(args, tform)

    net, params = GetPaynet(args)

    # Technically we are not training, but we still need this for the Evaluate method
    optimizer = GetOptimizer(args, params)

    # Load in model parameters into network
    checkpoint = torch.load(args.test_modelpath)
    net.load_state_dict(checkpoint['state_dict'])
    '''
    print(net.fc_hidden[0].weight)
    print(net.fc_hidden[1].weight)
    print(net.fc_hidden[2].weight)
    '''

    np.set_printoptions(threshold=np.nan)

    # TODO: diplicate in Train
    if args.exptType == 'AA':
        pI, pA, cI, cA, infosets_d, actions_d, num_open_groupIDs, sufficient_groupIDs = GetAAGame(args.exptObj, args.suff_type)
    elif args.exptType == 'AB':
        pI, pA, cI, cA, infosets_d, actions_d, num_open_groupIDs, sufficient_groupIDs = GetABGame(args.exptObj, args.suff_type)

    other = {'pI': [[-1], pI],
             'pA': [[0], pA],
             'cI': [[[0]], cI],
             'cA': [[[]], cA]}

    log_list = []
    for i_batch, sample_batched in enumerate(dl):

        print(sample_batched)
        loss, lossu, lossv = Evaluate(net, sample_batched, 'logloss', optimizer, real=True)
        # loglosstrend.append(loss.data.numpy())
        logger.info('Total validation log-loss: %s, U loss: %s, V loss: %s' % \
                    (loss.data.numpy(), lossu.data.numpy(), lossv.data.numpy()))
        print(loss)
        log_list.append(loss.data.numpy()[0])

        loss, lossu, lossv = Evaluate(net, sample_batched, 'indiv_logloss', optimizer, real=True, others=other)

        # Map losses to rounds
        def StagewiseLoss(L):
            # TODO: extract out, repeated function
            loglosslist, Linfosets = L[0], L[1]
            Linfosets = np.array(num_open_groupIDs)[np.array(Linfosets).astype(int)]

            return [np.mean(np.array(loglosslist)[Linfosets==x]) for x in range(4)]

        def TotalWeightedLoss(L):
            loglosslist, Linfosets = L[0], L[1]
            return np.mean(L[0])

        print('Stagewise log loss')
        stagewiseloss = StagewiseLoss(lossv)
        print(stagewiseloss)

        print('Total weighted loss')
        twl = TotalWeightedLoss(lossv)
        print(twl)

    print(log_list)
    print(np.mean(log_list))
    print(np.std(log_list))
    plt.plot(log_list)
    plt.show()


#####################################################################
# Training Procedure
#####################################################################

def Train(args, checkpoint=None):
    # data = HuntGameDataset(args.data_path, size=args.datasize)

    tform = GetEncoding(args)

    test_dl, train_dl = GetDataloaders(args, tform)

    net, params = GetPaynet(args)

    optimizer = GetOptimizer(args, params)

    start_epoch = 0

    best_val_loss = float('inf')
    if checkpoint is not None:
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint['i_epoch']) + 1
        if 'best_loss' in checkpoint:
            best_val_loss = checkpoint['best_loss']


    np.set_printoptions(threshold=np.nan)

    ### This portion is just to compute prediction accuracy ###
    if args.exptType == 'AA':
        pI, pA, cI, cA, infosets_d, actions_d, num_open_groupIDs, sufficient_groupIDs = GetAAGame(args.exptObj, args.suff_type)
    elif args.exptType == 'AB':
        pI, pA, cI, cA, infosets_d, actions_d, num_open_groupIDs, sufficient_groupIDs = GetABGame(args.exptObj, args.suff_type)

    # Initialize benchmark
    uniform_benchmark = UniformBenchmark(exptType=args.exptType, exptObj=args.exptObj)
    for i_batch, sample_batched in enumerate(test_dl):
        # print(sample_batched[2].numpy().astype(int))

        # Average probability of correctness
        # print(uniform_benchmark[sample_batched[2].numpy().astype(int)])
        print(' Mean probability of corectness:', np.mean(uniform_benchmark[sample_batched[2].numpy().astype(int)]))

        # log likelihood
        # print(np.log(uniform_benchmark[sample_batched[2].numpy().astype(int)]))
        print('Mean log probability', np.mean(np.log(uniform_benchmark[sample_batched[2].numpy().astype(int)])))

    #######################################

    #  Training proper
    train_loss = []
    test_loss = []

    print(params)

    for i_epoch in range(start_epoch, args.nEpochs):
        print('Epoch:', i_epoch)

        def SaveState(args, fname, meta):
            state = {'i_epoch': i_epoch,
                     'best_loss': best_val_loss,
                    'args': args,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}
            state.update(meta)
            torch.save(state, fname)

        train_loss_per_mb = []
        for i_batch, sample_batched in enumerate(train_dl):
            optimizer.zero_grad()

            """ Old code, replaced by adapter
            ndsize = sample_batched['Av'].size()[0]

            feats = []
            for w in args.feature_types:
                if w == 'const': 
                    feats.append(torch.Tensor(np.ones([ndsize, 1])))
                elif w == 'trialnum':
                    one_hot = torch.Tensor(ndsize, feat_to_nums[w]).zero_()
                    tmp = torch.unsqueeze(sample_batched[w] >= 11, 1)
                    one_hot.scatter_(1,tmp.long(),1.)
                    feats.append(one_hot)
                else:
                    one_hot = torch.Tensor(ndsize, feat_to_nums[w]).zero_()
                    one_hot.scatter_(1,torch.unsqueeze(sample_batched[w], 1).long(),1.)
                    feats.append(one_hot)

            feats = torch.cat(feats, 1)
            Av = sample_batched['Av']

            Av = Variable(Av.long(), requires_grad=False)
            feats = Variable(feats.double(), requires_grad=False)
            """

            """
            ################################################
            # Hacky estimation of second derivative (using old code) #
            print('Hacking out second derivative')
            net.zero_grad()
            u, v, lambdas = net(feats)
            loss = nll(torch.log(v), Av)
            loss.backward(retain_graph=True)

            orig_grad = copy.deepcopy(list(net.parameters())[0].grad.data.numpy())
            # orig_loss = copy.deepcopy(loss.data.numpy())
            # print('original gradient')
            # print(orig_grad)

            orig_weights = copy.deepcopy(net.fc_lambds.weight.data.numpy())
            epsilon = 10.**(-8)
            grads = []
            hess = np.zeros((net.num_lambdas, net.num_lambdas))
            loss_check = []
            for l_idx in range(net.num_lambdas):
                # elementary basis vectors
                ei = [1. if i == l_idx else 0. for i in range(net.num_lambdas)]
                ei = np.atleast_2d(ei).T
                
                ''' 
                # FIRST DERIVATIVE #
                # 2 point formula for finite difference
                # Positive direction 
                net.fc_lambds.weight = torch.nn.Parameter(torch.from_numpy(orig_weights + epsilon*ei))
                optimizer.zero_grad()
                net.zero_grad()
                u, v, lambdas = net(feats)
                loss_positive = nll(torch.log(v), Av)
                loss.backward(retain_graph=True)
                
                # Negative direction
                net.fc_lambds.weight = torch.nn.Parameter(torch.from_numpy(orig_weights - epsilon*ei))
                optimizer.zero_grad()
                net.zero_grad()
                u, v, lambdas = net(feats)
                loss_negative = nll(torch.log(v), Av)

                loss_check.append((loss_positive.data.numpy()-loss_negative.data.numpy())/2/epsilon)
                '''

                # SECOND DERIATIVE (HESSIAN)
                net.fc_lambds.weight = torch.nn.Parameter(torch.from_numpy(orig_weights + epsilon*ei))
                optimizer.zero_grad()
                net.zero_grad()
                u, v, lambdas = net(feats)
                loss_positive = nll(torch.log(v), Av)
                loss.backward(retain_graph=True)

                grads.append(list(net.parameters())[0].grad.data.numpy())
                
                # hess.append(grads[-1] - orig_grad)
                # print(grads[-1] - orig_grad)
                # print(orig_grad)
                # print(grads[-1])
                # print((grads[-1] - orig_grad)/1.)
                # print('-------')

                # FOR HESSIAN 
                hess[l_idx, :] = np.squeeze((grads[-1] - orig_grad))

            hessian = hess/epsilon
            print('hessian:', hessian)
            # print('hacked grad:', loss_check)
            print(np.linalg.eigh(hessian))


            # Revert weights
            net.fc_lambds.weight = torch.nn.Parameter(torch.from_numpy(orig_weights))
            optimizer.zero_grad()
            net.zero_grad()
            # End hacky part 1
            ###################################################################
            """

            """ Old code, running forward pass (no backwards yet)
            # print('forward')
            u, v, lambds = net(feats)
            # print('forward done')
            # 3350 3351
            # 5330 5331
            '''
            if i_batch==0:
                print('3350 vs 3351')
                print(v[0, 3350], v[0, 3351]) # ((0, 0, 1, 0), ('OS', 'OD', 'OD', 'GS/GD'))                
                print('5330 vs 5331')
                print(v[0, 5330], v[0, 5331]) # ((1, 1, 0, 0), ('OS', 'OD', 'OD', 'GS/GD'))
            '''


            loss = nll(torch.log(v), Av)

            '''
            print('--------------------------')
            print(np.argmin(v.data.numpy()).T)
            print(np.min(v.data.numpy()))
            print(lambds[0, :].data.numpy())
            print('--------------------------')
            '''
            """

            # print('forward')
            loss, lossu, lossv = Evaluate(net, sample_batched, 'logloss', optimizer, real=True)
            # print('backward')
            loss.backward(retain_graph=True)
            # print('optimizer')
            # print(lossv)
            optimizer.step()
            # print('others')

            train_loss_per_mb.append(np.copy(loss.data.numpy())) # debug? #

            sys.stdout.write('|')
            sys.stdout.flush()
        
        # print('Peek at lambda:', lambds[0, :])

        train_loss.append(np.mean(np.array(train_loss_per_mb)))

        logger.info('Average train loss: %s' % (train_loss[-1]))

        fname = os.path.join(args.regular_path, str(i_epoch))
        SaveState(args, fname, {})

        if i_epoch % args.monitorFreq == 0:
            meta = dict()
            # print(net.fc_lambds.weight)
            for i_batch, sample_batched in enumerate(test_dl):
                """ Old code
                ndsize = sample_batched['Av'].size()[0]
                feats = []
                for w in args.feature_types:
                    if w == 'const': 
                        feats.append(torch.Tensor(np.ones([ndsize, 1])))
                    elif w == 'trialnum':
                        one_hot = torch.Tensor(ndsize, feat_to_nums[w]).zero_()
                        tmp = torch.unsqueeze(sample_batched[w] >= 11, 1)
                        one_hot.scatter_(1,tmp.long(),1.)
                        feats.append(one_hot)
                    else:
                        one_hot = torch.Tensor(ndsize, feat_to_nums[w]).zero_()
                        one_hot.scatter_(1,torch.unsqueeze(sample_batched[w], 1).long(),1.)
                        feats.append(one_hot)

                feats = torch.cat(feats, 1)
                Av = sample_batched['Av']

                Av = Variable(Av.long(), requires_grad=False)
                feats = Variable(feats.double(), requires_grad=False)

                u, v, lambds = net(feats)

                loss = nll(torch.log(v), Av)
                """
                loss, lossu, lossv = Evaluate(net, sample_batched, 'logloss', optimizer, real=True)
                # loglosstrend.append(loss.data.numpy())
                logger.info('Total validation log-loss: %s, U loss: %s, V loss: %s' % \
                            (loss.data.numpy(), lossu.data.numpy(), lossv.data.numpy()))
                meta['logloss'] = loss.data.numpy()

                loss, lossu, lossv = Evaluate(net, sample_batched, 'avg_success', optimizer, real=True)
                # loglosstrend.append(loss.data.numpy())
                # print(np.stack([sample_batched[2].numpy().astype(int), loss.data.numpy()]).T)
                print('Avg success')
                print(np.mean(loss.data.numpy()))
                meta['avg_success'] = loss.data.numpy()

                other = {'pI': [[-1], pI],
                         'pA': [[0], pA],
                         'cI': [[[0]], cI],
                         'cA': [[[]], cA]}

                loss, lossu, lossv = Evaluate(net, sample_batched, 'accuracy', optimizer, real=True, others=other)
                # loglosstrend.append(loss.data.numpy())
                # print(np.stack([sample_batched[2].numpy().astype(int), loss.data.numpy()]).T)
                print('Mean accuracy')
                print(loss)
                meta['accuracy'] = loss

                loss, lossu, lossv = Evaluate(net, sample_batched, 'indiv_logloss', optimizer, real=True, others=other)

                # Map losses to rounds
                def StagewiseLoss(L):
                    loglosslist, Linfosets = L[0], L[1]
                    Linfosets = np.array(num_open_groupIDs)[np.array(Linfosets).astype(int)]

                    # print(np.array(loglosslist)[Linfosets==3])
                    return [np.mean(np.array(loglosslist)[Linfosets==x]) for x in range(4)]

                print('Stagewise log loss')
                stagewiseloss = StagewiseLoss(lossv)
                print(stagewiseloss)
                meta['stagewise'] = stagewiseloss

                '''
                for k in net.fc_hidden:
                    print(k.weight)
                print(net.fc_lambds.weight)
                '''

                #print('aaaa')
                #print(np.mean(loss.data.numpy()))

                # logger.info('Total validation accuracy: %s' % loss.data.numpy())


                '''
                test_loss.append(loss.data.numpy())
                logger.info('Average test loss: %s' % (test_loss[-1]))
                '''

            # fname = args.save_path + '%06d' % (i_epoch) + '.p'
            # pickle.dump(sto_dict, open( fname, 'wb'))

            # Save
            fname = os.path.join(args.val_path, str(i_epoch))
            SaveState(args, fname, meta)


            if meta['logloss'] < best_val_loss:
                fname = os.path.join(args.save_path, 'best')
                SaveState(args, fname, meta)

            # Save if best
    return


def GetPaynet(args):
    net = HuntBasicPaynet(parameterization_type=args.parameterization_type,
                          exptType=args.exptType, exptObj=args.exptObj,
                          arch=args.arch, nFeats=args.nFeats, verify=args.verify,
                          suff_type=args.suff_type if hasattr(args, 'suff_type') else 'simple')
    net = net.double()
    params = [x for x in net.parameters() if x.requires_grad]
    return net, params


def GetDataloaders(args, tform):
    data = HuntGameDataset2(args.data_path, size=args.datasize, transform=tform)
    # usize, vsize = data.GetGameDimensions() # Fix this
    train_data, test_data = data, data.Split(frac=args.fracval, random=False)
    train_dl = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_data, batch_size=test_data.__len__(), shuffle=False, num_workers=0)
    return test_dl, train_dl


def GetTestDataloaders(args, tform):
    data = HuntGameDataset2(args.data_path, size=args.test_datasize, offset=-args.test_datasize, transform=tform)
    data = DataLoader(data, batch_size=data.__len__(), shuffle=False, num_workers=0)
    # data = DataLoader(data, batch_size=10, shuffle=False, num_workers=0)
    return data


def GetOptimizer(args, params):
    if args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params,
                              lr=args.lr,
                              momentum=args.momentum,
                              nesterov=False)
    return optimizer


def GetEncoding(args):
    if args.encoding == 'one_hot':
        tform = FeatToOneHotEncoding(args)
    elif args.encoding == 'linear':
        tform = FeatToLinearEncoding(args)
    return tform


class FeatToOneHotEncoding(object):
    '''
    Transformer for standardized feats to one hot encoding
    '''

    def __init__(self, args):
        self.feature_types = args.feature_types
        self.str_to_feat_id = {'age':0, 'education':1, 'gender':2, 'location':3, 'trialnum':4}


    def __call__(self, sample):
        feats = []
        for w in self.feature_types:
            if w == 'const':
                # feats.append(torch.Tensor(np.ones([1])))
                feats.append(torch.Tensor(np.zeros([1])))
            else:
                feat_id = self.str_to_feat_id[w]
                one_hot = torch.Tensor(feat_to_nums_onehot[w]).zero_()
                if w != 'trialnum':
                    one_hot[sample[feat_id]] = 1
                else:
                    a = 1 if sample[feat_id] > 11 else 0
                    one_hot[a] = 1
                feats.append(one_hot)

        feats = torch.cat(feats, 0)
        return feats


class FeatToLinearEncoding(object):
    '''
    Transformer for standardized feats to linear encoding
    '''

    def __init__(self, args):
        self.feature_types = args.feature_types
        self.str_to_feat_id = {'age':0, 'education':1, 'gender':2, 'location':3, 'trialnum':4}


    def __call__(self, sample):
        feats = []
        for w in self.feature_types:
            if w == 'const':
                feats.append(torch.Tensor(np.ones([1])))
            else:
                feat_id = self.str_to_feat_id[w]
                if w == 'age':
                    mapper = np.array([18, 21, 27, 35, 45, 55, 65, 70]).astype(np.float64)
                elif w == 'education':
                    mapper = np.array([11, 13, 16, 19]).astype(np.float64)
                elif w == 'trialnum':
                    mapper = np.array(range(25)).astype(np.float64)
                elif w == 'gender':
                    mapper = np.array(range(3)).astype(np.float64)
                else:
                    assert False
                feats.append(torch.Tensor(np.atleast_1d(mapper[sample[feat_id]])))



        feats = torch.cat(feats, 0)
        return feats

#####################################################################
# Network structure
#####################################################################

## Encoding actions and infosets: there are 2 types of games 
## 1) AA -- where the first 2 cards uncovered are in the same row
## 2) AB -- the first 2 cards are in different rows

# game state: 2 tuples
# (cards opened)
# (actions in the past -- 'GS', 'GD', 'OS', 'OD'

# type AA
# 1 cards: ((x_1,), (,))
# 2 cards: ((x_1,x_2), ('OS',))
# 3 cards: ((x_1, x_2, x_3), ('OS', 'OD'))
# 4 cards: ((x_1, x_2, x_3, x_4), ('OS', 'OD', 'OD'))

# type AB
# 1 cards: ((x_1,), (,))
# 2 cards: ((x_1, x_2), ('GD'))
# 3 cards: ((x_1, x-2, x_3), ('GD', 'GS')), ((x_1, x-2, x_3), ('GD', 'GD')) 
# 4 cards: ((x_1, x-2, x_3, x_4), ('GD', 'GS', 'GD')), ((x_1, x-2, x_3, x_4), ('GD', 'GD', 'GS'))

# Construct game structure, cA, cI, pA, pI for AA and AB games separately
# Returns dictionaries mapping the above 2-tuples to numerical ids of the
# actions/infosets.

## Outputs: pI, pA, cI, cA, infosets_d, actions_d

# Experiment objective: 0 = FIND THE BIGGEST, 1 = FIND THE SMALLEST, 2 = ADD BIG, 3 = ADD
# SMALL, 4 = MULTIPLY BIG, 5 = MULTIPLY SMALL
# Operation. can change to whatever operation you'd want as well
OPS_FIND_BIG = 0
OPS_FIND_SMALL = 1
OPS_ADD_BIG = 2
OPS_ADD_SMALL = 3
OPS_MULT_BIG = 4
OPS_MULT_SMALL = 5
# EXPLORE_COST[i] is cost for revealing UP TO the i-th card (1-indexed), 
# padding with 0 at 0-index for convenience.
EXPLORE_COST = [0,0,-10,-25,-45]
ops = [lambda x, y: int(np.max(x) >= np.max(y)) - int(np.max(x) <= np.max(y)),
    lambda x, y: int(np.min(x) <= np.sum(y)) - int(np.min(x) >= np.sum(y)),
    lambda x, y: int(np.sum(x) >= np.sum(y)) - int(np.sum(x) <= np.sum(y)),
    lambda x, y: int(np.sum(x) <= np.sum(y)) - int(np.sum(x) >= np.sum(y)),
    lambda x, y: int(np.prod(x) >= np.prod(y)) - int(np.prod(x) <= np.prod(y)),
    lambda x, y: int(np.prod(x) <= np.prod(y)) - int(np.sum(x) >= np.prod(y))]

# Operation to use for sufficient statistic
suff_stat_op = [lambda x, y: max(x, y),
                lambda x, y: min(x, y),
                lambda x, y: x + y,
                lambda x, y: x + y,
                lambda x, y: x * y,
                lambda x, y: x * y]

def GetAAGame(exptObj, suff_type='simple'):
    nActions, nInfosets = 0, 0
    infosets_d, actions_d = dict(), dict()
    pA = dict()
    pI = dict()

    # Group IDs
    num_open_groupIDs = []   # <number of cards which are open> for given id

    suff_op = suff_stat_op[exptObj]
    sufficient_groupIDs = [] # <sufficient statistic> for given id
    suff_groupIDs_dict = dict() # dictionary for sufficient statistics

    # Suff stat scheme 2
    sufficient_groupIDs_new = [] # sufficient statistic
    suff_groupIDs_new_dict = dict() # dictionary for sufficient statistics

    for x in itertools.product(range(10), repeat=1):
        infosets_d[(x, ())] = nInfosets
        num_open_groupIDs.append(0)
        suff_stat = (1, x[0]) # sufficient statistic is same for scheme 1 and 2 are equal
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
            suff_groupIDs_new_dict[suff_stat] = len(sufficient_groupIDs_new)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        sufficient_groupIDs_new.append(suff_groupIDs_new_dict[suff_stat])
        pI[nInfosets] = None
        nInfosets += 1

    for x in itertools.product(range(10), repeat=2):
        infosets_d[(x, ('OS',))] = nInfosets
        num_open_groupIDs.append(1)
        suff_stat = (2, suff_op(x[0], x[1])) # sufficient_statistic is same for scheme 1 and 2 are equal
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
            suff_groupIDs_new_dict[suff_stat] = len(suff_groupIDs_new_dict)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        sufficient_groupIDs_new.append(suff_groupIDs_new_dict[suff_stat])
        nInfosets += 1

    for x in itertools.product(range(10), repeat=3):
        infosets_d[(x, ('OS', 'OD'))] = nInfosets
        num_open_groupIDs.append(2)
        suff_stat = (3, suff_op(x[0], x[1]), x[2]) # sufficient_statistic is same for scheme 1 and 2 are equal
        suff_stat2 = (3, abs(suff_op(x[0], x[1]) - x[2]))
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
            suff_groupIDs_new_dict[suff_stat2] = len(suff_groupIDs_new_dict)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        sufficient_groupIDs_new.append(suff_groupIDs_new_dict[suff_stat2])
        nInfosets += 1

    for x in itertools.product(range(10), repeat=4):
        infosets_d[(x, ('OS', 'OD', 'OD'))] = nInfosets
        num_open_groupIDs.append(3)
        suff_stat = (4, suff_op(x[0], x[1]), suff_op(x[2], x[3]))
        suff_stat2 = (4, )
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
            suff_groupIDs_new_dict[suff_stat2] = len(suff_groupIDs_new_dict)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        sufficient_groupIDs_new.append(suff_groupIDs_new_dict[suff_stat2])
        nInfosets += 1

    # Insert actions for each infoset and set the parent of their children
    # to that action
    for x in itertools.product(range(10), repeat=1):
        i_id = infosets_d[(x, ())]
        for a in ['GS', 'GD']:
            actions_d[(x, (a,))] = nActions
            pA[nActions] = i_id
            nActions += 1

        actions_d[(x, ('OS',))] = nActions
        for i in range(10):
            pI[infosets_d[(x+(i,), ('OS',))]] = nActions
        pA[nActions] = i_id
        nActions += 1

    for x in itertools.product(range(10), repeat=2):
        i_id = infosets_d[(x, ('OS',))]
        for a in ['GS', 'GD']:
            actions_d[(x, ('OS', a))] = nActions
            pA[nActions] = i_id
            nActions += 1

        actions_d[(x, ('OS', 'OD'))] = nActions
        for i in range(10):
            pI[infosets_d[(x+(i,), ('OS', 'OD',))]] = nActions
        pA[nActions] = i_id
        nActions += 1

    for x in itertools.product(range(10), repeat=3):
        i_id = infosets_d[(x, ('OS','OD'))]
        for a in ['GS', 'GD']: # guessing
            actions_d[(x, ('OS','OD',a))] = nActions
            pA[nActions] = i_id
            nActions += 1

        actions_d[(x, ('OS', 'OD', 'OD'))] = nActions
        for i in range(10): # open
            pI[infosets_d[(x+(i,), ('OS', 'OD','OD'))]] = nActions
        pA[nActions] = i_id
        nActions += 1

    for x in itertools.product(range(10), repeat=4):
        i_id = infosets_d[(x, ('OS','OD','OD'))]
        for a in ['GS', 'GD']: # guessing
            actions_d[(x, ('OS','OD','OD',a))] = nActions
            pA[nActions] = i_id
            nActions += 1

    # TODO: remove impossible situations in that dataset.

    # Update all children using parent information
    cA, cI = [[] for i in range(len(pA))], [[] for i in range(len(pI))]
    # Get children information
    for k, v in pI.items():
        if v is not None: cA[v].append(k)
    for k, v in pA.items():
        if v is not None: cI[v].append(k)

    # Convert pA, pI to list (or array)
    pA_list = [-1] * len(pA)
    pI_list = [-1] * len(pI)
    for k,v in pA.items():
        pA[k] = v
    for k,v in pI.items():
        if v is not None: pI[k] = v
        else: pI[k] = -1

    if suff_type == 'simple':
        return pI, pA, cI, cA, infosets_d, actions_d, num_open_groupIDs, sufficient_groupIDs
    elif suff_type == 'diff':
        return pI, pA, cI, cA, infosets_d, actions_d, num_open_groupIDs, sufficient_groupIDs_new

def GetABGame(exptObj):
    nActions, nInfosets = 0, 0
    infosets_d, actions_d = dict(), dict()
    pA = dict()
    pI = dict()

    # Group IDs
    num_open_groupIDs = []   # <number of cards which are open> for given id

    suff_op = suff_stat_op[exptObj]
    sufficient_groupIDs = [] # <sufficient statistic> for given id
    suff_groupIDs_dict = dict() # dictionary for sufficient statistics

    for x in itertools.product(range(10), repeat=1):
        infosets_d[(x, ())] = nInfosets
        num_open_groupIDs.append(0)
        suff_stat = (1, x[0])
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        pI[nInfosets] = None
        nInfosets += 1

    for x in itertools.product(range(10), repeat=2):
        infosets_d[(x, ('OD',))] = nInfosets
        num_open_groupIDs.append(1)
        suff_stat = (11, x[0], x[1])
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        nInfosets += 1

    for x in itertools.product(range(10), repeat=3):
        infosets_d[(x, ('OD', 'OS'))] = nInfosets
        num_open_groupIDs.append(2)
        suff_stat = (121, suff_op(x[0], x[2]), x[1])
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        nInfosets += 1

    for x in itertools.product(range(10), repeat=3):
        infosets_d[(x, ('OD', 'OD'))] = nInfosets
        num_open_groupIDs.append(2)
        suff_stat = (122, x[0], suff_op(x[1], x[2]))
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        nInfosets += 1

    for x in itertools.product(range(10), repeat=4):
        infosets_d[(x, ('OD', 'OS', 'OD'))] = nInfosets
        num_open_groupIDs.append(3)
        suff_stat = (1212, suff_op(x[0], x[2]), suff_op(x[1], x[3]))
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        nInfosets += 1

    for x in itertools.product(range(10), repeat=4):
        infosets_d[(x, ('OD', 'OD', 'OS'))] = nInfosets
        num_open_groupIDs.append(3)
        suff_stat = (1221, suff_op(x[0], x[3]), suff_op(x[1], x[2]))
        if suff_stat not in suff_groupIDs_dict:
            suff_groupIDs_dict[suff_stat] = len(suff_groupIDs_dict)
        sufficient_groupIDs.append(suff_groupIDs_dict[suff_stat])
        nInfosets += 1

    # Insert actions for each infoset and set the parent of their children
    # to that action
    for x in itertools.product(range(10), repeat=1):
        i_id = infosets_d[(x, ())]
        for a in ['GS', 'GD']:
            actions_d[(x, (a,))] = nActions
            pA[nActions] = i_id
            nActions += 1

        actions_d[(x, ('OD',))] = nActions
        for i in range(10):
            pI[infosets_d[(x+(i,), ('OD',))]] = nActions
        pA[nActions] = i_id
        nActions += 1

    for x in itertools.product(range(10), repeat=2):
        i_id = infosets_d[(x, ('OD',))]
        for a in ['GS', 'GD']:
            actions_d[(x, ('OD', a))] = nActions
            pA[nActions] = i_id
            nActions += 1

        actions_d[(x, ('OD', 'OS'))] = nActions
        for i in range(10):
            pI[infosets_d[(x+(i,), ('OD', 'OS',))]] = nActions
        pA[nActions] = i_id
        nActions += 1

        actions_d[(x, ('OD', 'OD'))] = nActions
        for i in range(10):
            pI[infosets_d[(x+(i,), ('OD', 'OD',))]] = nActions
        pA[nActions] = i_id
        nActions += 1

    for x in itertools.product(range(10), repeat=3):
        i_id = infosets_d[(x, ('OD','OS'))]
        for a in ['GS', 'GD']: # guessing
            actions_d[(x, ('OD','OS',a))] = nActions
            pA[nActions] = i_id
            nActions += 1

        actions_d[(x, ('OD', 'OS', 'OD'))] = nActions
        for i in range(10): # open
            pI[infosets_d[(x+(i,), ('OD', 'OS','OD'))]] = nActions
        pA[nActions] = i_id
        nActions += 1

    for x in itertools.product(range(10), repeat=3):
        i_id = infosets_d[(x, ('OD','OD'))]
        for a in ['GS', 'GD']: # guessing
            actions_d[(x, ('OD','OD',a))] = nActions
            pA[nActions] = i_id
            nActions += 1

        actions_d[(x, ('OD', 'OD', 'OS'))] = nActions
        for i in range(10): # open
            pI[infosets_d[(x+(i,), ('OD', 'OD','OS'))]] = nActions
        pA[nActions] = i_id
        nActions += 1

    for x in itertools.product(range(10), repeat=4):
        i_id = infosets_d[(x, ('OD','OS','OD'))]
        for a in ['GS', 'GD']: # guessing
            actions_d[(x, ('OD','OS','OD',a))] = nActions
            pA[nActions] = i_id
            nActions += 1

    for x in itertools.product(range(10), repeat=4):
        i_id = infosets_d[(x, ('OD','OD','OS'))]
        for a in ['GS', 'GD']: # guessing
            actions_d[(x, ('OD','OD','OS',a))] = nActions
            pA[nActions] = i_id
            nActions += 1

    # TODO: remove impossible situations in that dataset.

    # Update all children using parent information
    cA, cI = [[] for i in range(len(pA))], [[] for i in range(len(pI))]
    # Get children information
    for k, v in pI.items():
        if v is not None: cA[v].append(k)
    for k, v in pA.items():
        if v is not None: cI[v].append(k)

    # Convert pA, pI to list (or array)
    pA_list = [-1] * len(pA)
    pI_list = [-1] * len(pI)
    for k,v in pA.items():
        pA[k] = v
    for k,v in pI.items():
        if v is not None: pI[k] = v
        else: pI[k] = -1

    return pI, pA, cI, cA, infosets_d, actions_d, num_open_groupIDs, sufficient_groupIDs

def GenerateP(pI, pA, cI, cA, infosets_d, actions_d, op, exptObj,
            PAY_WIN=60, PAY_LOSE=-50):
    ''' Construct the payoff matrix P.
    :param op function mapping (x, y) to 1 if win and -1 if lose,
            where x,y are the 2 rows of the game.
    '''
    P = [0] * len(pA)

    # Get rewards -- ties assumed to be 0.
    for a_key, a_id in actions_d.items():
        opened = a_key[0] # numbers for opened cards
        pattern = a_key[1][:-1] # configuration of actions (fixed for AA, but differs for AB)
        lastaction = a_key[1][-1]
        if lastaction not in ['GS', 'GD']:
            P[a_id] = 0.
            continue

        # brute force...
        possible = []
        for c in itertools.product(range(10), repeat=4-len(opened)):
            if exptObj == 'AA':
                merged = opened + c
            elif exptObj == 'AB':
                same, diff = [], []
                same.append(opened[0])
                for idx, p in enumerate(pattern):
                    if p == 'OS': same.append(opened[idx+1])
                    elif p == 'OD': diff.append(opened[idx+1])

                idx = 0
                for z in range(2 - len(same)):
                    same.append(c[idx])
                    idx += 1
                for z in range(2 - len(diff)):
                    diff.append(c[idx])
                    idx += 1

                merged = same + diff

                '''
                print(a_id)
                print(opened)
                print(c)
                print(merged)
                '''

            else:
                assert False, 'invalid exptObj'

            if lastaction == 'GS':
                result = op(merged[0:2], merged[2:])
            elif lastaction == 'GD':
                result = op(merged[2:], merged[0:2])

            if result == 1: reward = PAY_WIN
            elif result == -1: reward = PAY_LOSE
            else: reward = 0 # This is the players' perceived function
            possible.append(reward)
        expected = np.mean(possible)
        # subtract scouting cost
        expected += EXPLORE_COST[len(opened)]
        # Prob we can even reach here
        P[a_id] = expected / (10.**len(opened))

    return P


class HuntGameDataset2(GameDataset):
    def __init__(self, loadpath, size=-1, offset=0, transform=lambda x: x):
        self.loadpath = loadpath
        super(HuntGameDataset2, self).__init__(loadpath, size=size, offset=offset, real=True)
        self.transform = transform

    def __getitem__(self, idx):
        f, Au, Av = super(HuntGameDataset2, self).__getitem__(idx)
        # print(idx, f)
        return self.transform(f), Au, Av

    def Split(self, frac=0.3, random=False):
        other = HuntGameDataset2(self.loadpath, transform=self.transform)
        return GameDataset_.Split(self, frac=frac, random=random, other_object=other)


class HuntGameDataset(Dataset):
    '''
    Old school unformatted dataset
    '''
    def __init__(self, loadpath, size=-1):
        '''
        :param size size of dataset to read -1 if we use the full dataset
        '''
        self.loadpath = loadpath
        f = h5.File(loadpath, 'r')
        self.f = f
        self.age = f['age'][:].astype(np.int32)
        self.education = f['education'][:].astype(np.int32)
        self.gender = f['gender'][:].astype(np.int32)
        self.location = f['location'][:].astype(np.int32)
        self.Av = f['vA'][:].astype(np.int32)
        self.trialnum = f['trialnum'][:].astype(np.int32)

        # By default, we use the full dataset
        self.offset, self.size = 0, self.Av.shape[0] if size == -1 else size

        super(Dataset, self).__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx > self.size:
            raise IndexError('idx > size')
        i = idx + self.offset

        """
        try:
            print(i, self.size)
            print('stats')
            print(self.Av[i])
            print(self.age[i,0])
            print(self.education[i,0])
            print(self.gender[i,0])
            print(self.location[i,0])
            print(self.trialnum[i])
        except:
            print('hahahahahahahaha')
        """
        
        # print('hey', self.size)
        """
        d = {'Av': self.Av[i].astype(np.int32), 
                'age': self.age[i].astype(np.int32),
                'education': self.education[i].astype(np.int32),
                'gender': self.gender[i].astype(np.int32),
                'location': self.location[i].astype(np.int32),
                'trialnum': self.trialnum[i].astype(np.int32)}
        """
        # print(d)
        # print(i)
        # print( self.Av[i], self.age[i,0], self.education[i,0], self.gender[i,0], self.location[i,0], self.trialnum[i])
        # return self.Av[i], self.age[i,0], self.education[i,0], self.gender[i,0], self.location[i,0], self.trialnum[i]

        return {'Av': self.Av[i],
                'age': self.age[i, 0],
                'education': self.education[i,0],
                'gender': self.gender[i, 0],
                'location': self.location[i,0],
                'trialnum': self.trialnum[i]}


    def Split(self, frac=0.3, random=False):
        '''
        :param fraction of dataset reserved for the other side
        :param shuffle range of data
        return new dataset carved of size self.size * frac
        '''
        if random == True:
            pass #TODO: randomly shuffle data. Make sure only to shuffle stuff within the range!

        cutoff = int(float(self.size) * (1.0-frac))
        if cutoff == 0 or cutoff >= self.size:
            raise ValueError('Split set is empty or entire set')

        other = HuntGameDataset(self.loadpath)
        other.offset = self.offset + cutoff
        other.size = self.size - cutoff
        self.size = cutoff
        
        return other


class HuntBasicPaynet(nn.Module):
    def __init__(self, 
                exptType,
                exptObj,
                parameterization_type,
                arch,
                nFeats=[1],
                verify=False,
                suff_type='simple'):
        '''
        Default module. Do not use any features.
        '''

        super(HuntBasicPaynet, self).__init__()
        if exptType == 'AA':
            self.pI, self.pA, self.cI, self.cA, self.infosets_d, \
            self.actions_d, self.num_open_groupIDs, self.sufficient_groupIDs = GetAAGame(exptObj, suff_type=suff_type)
        elif exptType == 'AB': # TODO
            self.pI, self.pA, self.cI, self.cA, self.infosets_d, \
            self.actions_d, self.num_open_groupIDs, self.sufficient_groupIDs = GetABGame(exptObj, suff_type=suff_type)
        print('Number of information sets', len(self.pI))

        self.parameterization_type = parameterization_type 
        self.P = GenerateP(self.pI, self.pA, self.cI, self.cA, self.infosets_d, self.actions_d, ops[exptObj], exptType)
        # print(np.max(self.P))
        # print(np.min(self.P))
        self.P = np.atleast_2d(self.P)
        self.P = torch.Tensor(self.P)
        self.exptType = exptType
        self.Au = [[]]
        self.Iu = [[0]]
        self.Av = self.cA
        self.Iv = self.cI
        
        self.usize = 1
        self.vsize = len(self.pA)
        self.uInfosize = 1
        self.vInfosize = len(self.pI)

        self.arch = arch

        prev_width = sum(nFeats)

        self.fc_hidden = nn.ModuleList()
        self.nHidden = len(arch.widths)
        for w in arch.widths:
            self.fc_hidden.append(nn.Linear(prev_width, w, bias=True))
            prev_width = w
            self.fc_hidden[-1].double()

        if self.parameterization_type == 'same':
            self.num_lambdas = 1

            self.fc_lambds = nn.Linear(prev_width, self.num_lambdas, bias=True)
            self.fc_lambds.bias = torch.nn.Parameter(
                torch.from_numpy(2. * np.ones(self.num_lambdas))
            )
            self.fc_lambds.weight = torch.nn.Parameter(
                torch.from_numpy(0. * np.ones([self.num_lambdas, prev_width])))

        if self.parameterization_type == 'num_open':
            self.num_lambdas = 4 # For fixed number of lambdas

            # self.fc_lambds = nn.Linear(prev_width, self.num_lambdas, bias=False)
            self.fc_lambds = nn.Linear(prev_width, self.num_lambdas, bias=True)
            self.fc_lambds.bias = torch.nn.Parameter(
                torch.from_numpy(0. * np.ones(self.num_lambdas))
            )
            self.fc_lambds.weight = torch.nn.Parameter(
                   torch.from_numpy(0. * np.ones([self.num_lambdas, prev_width])))
            
            #self.fc_lambds.weight = torch.nn.Parameter(
            #      torch.from_numpy(np.array([[0,0,0,0]]).T))
            # self.fc_lambds.weight = torch.nn.Parameter(
            #         torch.from_numpy(np.array([[1,-1,-5,-8]]).T))
            # self.fc_lambds.weight = torch.nn.Parameter(
            #        torch.from_numpy(np.array([[5,2,1,0.1]]).T))
            # self.fc_lambds.weight = torch.nn.Parameter(
            #         torch.from_numpy(np.exp(np.array([[1,-1,-5,-8]]).T)))


            # Case where things don't work
            # self.fc_lambds.weight = torch.nn.Parameter(
            #        torch.from_numpy(np.log(np.array([[0.5, 0.5, 0.5, 1.5]]).T)))

            self.index_ref = torch.LongTensor(self.num_open_groupIDs)

        elif self.parameterization_type == 'all':
            self.num_lambdas = len(self.pI)
            self.fc_lambds = nn.Linear(sum(nFeats), self.num_lambdas, bias=False)
            # self.fc_lambds.weight = torch.nn.Parameter(
            #         torch.from_numpy(np.zeros((self.num_lambdas, 1))))
            self.fc_lambds.weight = torch.nn.Parameter(
                    torch.from_numpy(-1.*np.ones((self.num_lambdas, sum(nFeats)))))
        elif self.parameterization_type == 'sufficient':
            self.num_lambdas = int(np.max(self.sufficient_groupIDs)+1)
            self.fc_lambds = nn.Linear(sum(nFeats), self.num_lambdas, bias=True)
            self.fc_lambds.bias = torch.nn.Parameter(
                torch.from_numpy(0. * np.ones(self.num_lambdas))
            )
            self.fc_lambds.weight = torch.nn.Parameter(
                torch.from_numpy(0. * np.ones([self.num_lambdas, prev_width])))
            '''
            self.fc_lambds.weight = torch.nn.Parameter(
                     torch.from_numpy(-1 * np.ones((self.num_lambdas, sum(nFeats)))))
            '''
            self.index_ref = torch.LongTensor(self.sufficient_groupIDs)
        # self.fc_lambds.weight = torch.nn.Parameter(
        #         torch.from_numpy(-425*np.ones([1,4]).T))
        # self.fc_lambds.weight = torch.nn.Parameter(
        #         torch.from_numpy(np.array([[1,-1,-3,-5]]).T))
        # self.fc_lambds.weight = torch.nn.Parameter(
        #        torch.from_numpy(np.array([[2,0,-2,-3]]).T-2))
        # self.fc_lambds.weight = torch.nn.Parameter(
        #        torch.from_numpy(np.array([[0,0,0,0]]).T))
        # for relu only
        
        


        #################################
        # Main
        #################################
        solve = QRESeqFOMLogit(tau=0.01, tol=10.**-10, 
                    max_it=10000, Ptype='keep', verify=verify)
        # solve = QRESeqFOMLogit(g=None, lambd=None, tau=0.1, tol=10**-10)

        # Set single_feature, use_inverse_back to true in order to use old method of inverse
        if False:
            self.solver = ZSGFOMSolver(self.Iu, self.Iv, self.Au, self.Av, solver=solve,
                                single_feature = True, use_inverse_back = True)
        else:
            self.solver = ZSGFOMSolver(self.Iu, self.Iv, self.Au, self.Av, solver=solve,
                                single_feature = False, use_inverse_back = False)

    def forward(self, x):
        nbatchsize = x.size()[0]
        for layer_idx, activation in enumerate(self.arch.activations):
            if activation == 'relu':
                x = nn.functional.relu(x)
            elif activation == 'linear':
                x = x
            elif activation == 'tanh':
                x = nn.functional.tanh(x)
            else:
                assert False, 'invalid '
            x = self.fc_hidden[layer_idx](x)

        if self.parameterization_type == 'same':
            j = torch.exp(self.fc_lambds(x)) + 10. ** -8
            index_ref = torch.LongTensor([0] * len(self.Iv))
            # self.index_ref = torch.LongTensor(self.num_open_groupIDs)
            loc = Variable(index_ref.expand(nbatchsize, index_ref.size()[0]))

            Lv = torch.gather(j, dim=1, index=loc)
            Lu = Variable(torch.Tensor([1.]).expand(nbatchsize, 1))

        if self.parameterization_type == 'num_open':
            # j of the data
            j = torch.exp(self.fc_lambds(x)) + 10. ** -8
            # j = self.fc_lambds(x)**2 + 10**-8
            # j = self.fc_lambds(x)
            # j = F.relu(self.fc_lambds(x)) + 10. ** -3

            # assignment to locations of lambda.
            loc = Variable(self.index_ref.expand(nbatchsize, self.index_ref.size()[0]))
            Lv = torch.gather(j, dim=1, index=loc)
            Lu = Variable(torch.Tensor([1.]).expand(nbatchsize, 1))

            #print(j)
            #print(self.fc_lambds.weight)
            #print(Lv[0, :])
            # assert False

            # print(j[0, :])
            # print(self.fc_lambds.weight)

        elif self.parameterization_type == 'all':
            j = torch.exp(self.fc_lambds(x)) + 10. ** -6
            Lv = j
            Lu = Variable(torch.Tensor([1.]).expand(nbatchsize, 1))

        elif self.parameterization_type == 'sufficient':
            # j of the data
            j = torch.exp(self.fc_lambds(x)) + 10. **-8
            # j = F.relu(self.fc_lambds(x)) + 10. ** -3

            # assignment to locations of lambda.
            loc = Variable(self.index_ref.expand(nbatchsize, self.index_ref.size()[0]))
            Lv = torch.gather(j, dim=1, index=loc)
            Lu = Variable(torch.Tensor([1.]).expand(nbatchsize, 1))

        # print(self.fc_lambds.weight)

        # Payoff matrix
        P = Variable(self.P.expand(nbatchsize, self.P.size()[0], self.P.size()[1]))
        # print('siz:', P[0][0][5530], P[0][0][5531])

        # Solve
        u, v = self.solver(P, Lu, Lv)
        # print(v)
        '''
        print(np.min(v.data.numpy()))
        print(np.max(v.data.numpy()))
        print(np.linalg.norm(v.data.numpy()))
        '''
        return u, v, j


##################################################################################

def UniformBenchmark(exptType, exptObj):
    '''
    :param exptType: AA or AB
    :param exptObj: Technically not required for uniform benchmarks, just set to 0 if unsure
    :return: strategies in sequence form, assuming a uniform behavior at each infoset
    '''
    if exptType == 'AA':
        pI, pA, cI, cA, infosets_d, actions_d, num_open_groupIDs, sufficient_groupIDs = GetAAGame(exptObj)
    elif exptType == 'AB':
        pI, pA, cI, cA, infosets_d, actions_d, num_open_groupIDs, sufficient_groupIDs = GetAAGame(exptObj)

    nActions, nInfosets = len(cA), len(cI)

    ret = np.zeros(nActions)
    for i in range(nInfosets):
        nChildActions = len(cI[i])
        for a in cI[i]:
            if pI[i] is None or pI[i] < 0:
                ret[a] = 1. / nChildActions
            else:
                ret[a] = ret[pI[i]] / nChildActions
    return ret

def OptimalBenchmark():
    pass



@ex.automain
def main(args, _run):
    run(args, _run)

