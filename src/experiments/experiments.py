import numpy as np
import torch.optim as optim
import h5py
import argparse
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class GameDataset_(Dataset):
    def __init__(self, f, size=-1, offset=0, real=False):
        '''
        :param size: size of dataset to read -1 if we use the full dataset
        :param real: true if we are using a true dataset, do not have access to ground truths
        '''
        self.f = f
        self.real = real
        self.feats, self.Au, self.Av = f['F'][:], f['Au'][:], f['Av'][:]
        if not real:
            self.GTu, self.GTv, self.GTP = f['U'][:], f['V'][:], f['P'][:]
            self.P = f['P']

        # Get other data if specified
        self.others = dict()
        if 'D' in f: 
            self.others['D'] = f['D']

        fullsize = self.feats.shape[0]

        if offset < 0: offset += fullsize
        self.offset = offset

        # By default, we use the full dataset
        self.size = fullsize - self.offset if size == -1 else size

        assert self.size + self.offset -1 <= fullsize

        super(GameDataset_, self).__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        i = self.GetRawIndex(idx)
        # print('Get:', i)
        if not self.real:
            if len(self.others) == 0:
                return self.feats[i, :], self.Au[i], self.Av[i], self.GTu[i, :], self.GTv[i, :], self.GTP[i, :, :]
            else:
                return self.feats[i, :], self.Au[i], self.Av[i], self.GTu[i, :], self.GTv[i, :], self.GTP[i, :, :], self.others['D'][i, :] # TODO: fix
        elif self.real:
            if len(self.others) == 0:
                return self.feats[i, :], self.Au[i], self.Av[i]
            else:
                return self.feats[i, :], self.Au[i], self.Av[i], self.others['D'][i, :] # TODO: fix

    def GetRawIndex(self, idx):
        if idx > self.size:
            raise IndexError('idx > size')
        i = idx + self.offset
        return i

    def GetGameDimensions(self):
        return tuple(self.P.shape[1:])

    def GetFeatureDimensions(self):
        return self.feats.shape[-1]

    def Split(self, frac=0.3, random=False, other_object=None):
        '''
        :param fraction of dataset reserved for the other side
        :param shuffle range of data
        :param other_object Instance of new other half of the class created
        # If we do not want to use GameDataset_ as the new class (e.g. if this was subclassed)
        We do not have to touch other_object if we are not subclassing GameDataset_ (or GameDataset)
        We WILL have to override this if we are subclassing though (e.g. hunt dataset)
        # TODO: is there a better way of subclassing this while reusing code in the Split function? We
        # have the issue of 'other' being of an unknown class

        return new dataset carved of size self.size * frac
        '''
        if random == True:
            pass #TODO: randomly shuffle data. Make sure only to shuffle stuff within the range!

        cutoff = int(float(self.size) * (1.0-frac))
        if cutoff == 0 or cutoff >= self.size:
            raise ValueError('Split set is empty or entire set')

        if other_object is None:
            other = GameDataset_(self.f, real=self.real)
        else:
            other = other_object
        other.offset = self.offset + cutoff
        other.size = self.size - cutoff
        self.size = cutoff
        
        return other


class GameDataset(GameDataset_):
    '''
    Game dataset for synthetic data
    '''
    def __init__(self, loadpath, size=-1, offset=0, real=False):
        self.loadpath = loadpath
        f = h5py.File(loadpath, 'r')
        super(GameDataset, self).__init__(f, size=size, offset=offset, real=real)


class AttackerDefenderDataset(GameDataset):
    '''
    Game dataset for Attacker-Defender games
    '''
    def __init__(self, loadpath, size=-1):
        super(AttackerDefenderDataset, self).__init__(loadpath, size)


    def GetNumResources(self):
        try: nRes = self.f['nDef'].value
        except KeyError as e:
            print('Error in accessing dataset, potentially not an attacker-defender game?')
            raise
            
        return nRes


def Evaluate(net, sample_batched, loss_type, optimizer, real=False, others=None):
    nll = nn.NLLLoss()
    mse = nn.MSELoss()
    KLDivLoss = nn.KLDivLoss()

    optimizer.zero_grad()
    
    batched_data = tuple(sample_batched)
    # TODO: less hacky way of extracting parameters
    if not real:
        if len(batched_data) == 6:
            feats, Au, Av, GTu, GTv, GTP = batched_data
        elif len(batched_data) == 7:
            feats, Au, Av, GTu, GTv, GTP, others = batched_data
        else:
            assert False, 'Unknown type of data returned! Size of batched_data %d' % len(batched_data)
    elif real:
        if len(batched_data) == 3:
            feats, Au, Av = batched_data
        elif len(batched_data) == 4:
            feats, Au, Av, others = batched_data
        else:
            assert False, 'Unknown type of data returned! Size of batched_data %d' % len(batched_data)

    feats = Variable(feats.double())
    Av = Variable(Av.long(), requires_grad=False)
    Au = Variable(Au.long(), requires_grad=False)

    if not real:
        GTu = Variable(GTu.double(), requires_grad=False)
        GTv = Variable(GTv.double(), requires_grad=False)
        GTP = Variable(GTP.double(), requires_grad=False)

    ret = net(feats)
    # TODO: less hacky way of extracting parameters
    if len(ret) == 3:
        U, V, P = ret
    elif len(ret) == 4: # for security game
        U, V, P, target_rewards = ret
    elif len(ret) == 5:
        U, V, P, betVal, cardProbs = ret # for OneCardPoker (TODO: refactor)

    if loss_type == 'mse':
        assert not real, 'MSE loss requires ground truth'
        lossu = mse(U, GTu)
        lossv = mse(V, GTv)
        loss = lossu + lossv

    elif loss_type == 'logloss':
        lossv = nll(torch.log(V), Av)
        lossu = nll(torch.log(U), Au)
        loss = lossu + lossv

    elif loss_type == 'ulogloss':
        lossu = nll(torch.log(U), Au)
        lossv = lossu
        loss = lossu

    elif loss_type == 'vlogloss':
        lossv = nll(torch.log(V), Av)
        lossu = lossv
        loss = lossv

    elif loss_type == 'paymatrixloss':
        assert not real, 'paymatrixloss loss requires ground truth'
        loss, lossu, lossv = mse(P, GTP), None, None

    elif loss_type == 'optimallogloss':
        assert not real, 'optimallogloss requires ground truth'
        GTlossu = nll(torch.log(GTu), Au)
        GTlossv = nll(torch.log(GTv), Av)
        lossu, lossv, loss = GTlossu, GTlossv, GTlossu + GTlossv

    elif loss_type == 'pokerparams_probs':
        GTcardprobs = Variable(others.double(), requires_grad=False)
        loss, lossu, lossv = mse(cardProbs, GTcardprobs), None, None

    elif loss_type == 'accuracy':

        def GetCnts(pI, pA, cI, cA, seq, truth):

            tCntCorrect, tCntPred = 0, 0
            for idx in range(torch.numel(Au.data)):
                cntCorrect, cntPred = StagewiseAccuracy(pI, pA, cI, cA, seq[idx, :], truth[idx])
                tCntCorrect += cntCorrect
                tCntPred += cntPred

            return tCntCorrect, tCntPred

        uCntCorrect, uCntPred = GetCnts(others['pI'][0],
                                        others['pA'][0],
                                        others['cI'][0],
                                        others['cA'][0],
                                        U.data.numpy(),
                                        Au.data.numpy())

        vCntCorrect, vCntPred = GetCnts(others['pI'][1],
                                        others['pA'][1],
                                        others['cI'][1],
                                        others['cA'][1],
                                        V.data.numpy(),
                                        Av.data.numpy())

        print(vCntCorrect, vCntPred)

        tCntCorrect = uCntCorrect + vCntCorrect
        tCntPred = uCntPred + vCntPred

        if tCntPred > 0:
            loss = float(tCntCorrect)/tCntPred
        else: loss = 0.

        if uCntPred > 0:
            lossu = float(uCntCorrect)/uCntPred
        else: lossu = 0.

        if vCntPred > 0:
            lossv = float(vCntCorrect)/vCntPred
        else: lossv = 0.

    elif loss_type == 'indiv_logloss':

        def GetLoglosses(pI, pA, cI, cA, seq, truth):

            loglosslist, infosetlist = [], []
            for idx in range(torch.numel(Au.data)):
                x, y = IndividualLogloss(pI, pA, cI, cA, seq[idx, :], truth[idx])
                loglosslist += x
                infosetlist += y

            return loglosslist, infosetlist

        uloglosses, uinfos = GetLoglosses(others['pI'][0],
                                        others['pA'][0],
                                        others['cI'][0],
                                        others['cA'][0],
                                        U.data.numpy(),
                                        Au.data.numpy())

        vloglosses, vinfos = GetLoglosses(others['pI'][1],
                                        others['pA'][1],
                                        others['cI'][1],
                                        others['cA'][1],
                                        V.data.numpy(),
                                        Av.data.numpy())

        loss = None
        lossu = (uloglosses, uinfos)
        lossv = (vloglosses, vinfos)



    elif loss_type == 'avg_success':
        # lossu = U[:, Au]
        # lossv = V[:, Av]
        lossu = U.gather(1, Au.view(-1, 1)).squeeze()
        lossv = V.gather(1, Av.view(-1, 1)).squeeze()
        loss = lossu * lossv

        '''
        print(U.size())
        print(V.size())
        print(Au.size())
        print(Av.size())
        print(lossu.size())
        print(lossv.size())
        print(loss.size())
        assert False
        '''

    elif loss_type == 'pokerparams_probs_joint_KLDiv':
        batchsize=others.size()[0]
        ncards=others.size()[1]

        GTcardprobs = Variable(others.double(), requires_grad=False)
        GTtotal_cards = GTcardprobs.sum(1)
        GTnormalizing_constant = GTtotal_cards * (GTtotal_cards - 1)
        GTouter = torch.bmm(GTcardprobs.unsqueeze(2), GTcardprobs.unsqueeze(1))
        GTcorrection = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            GTcorrection[i, :, :] = torch.diag(GTcardprobs[i, :])
        GTprobmat = (GTouter-GTcorrection)/GTnormalizing_constant.expand([ncards, ncards, batchsize]).transpose(0,2)
        GTprobmat_flat = GTprobmat.view(batchsize, -1) # technically no need for this

        total_cards = cardProbs.sum(1)
        normalizing_constant = total_cards * (total_cards - 1)
        outer = torch.bmm(cardProbs.unsqueeze(2), cardProbs.unsqueeze(1))
        correction = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            correction[i, :, :] = torch.diag(cardProbs[i, :])
        probmat = (outer-correction)/normalizing_constant.expand([ncards, ncards, batchsize]).transpose(0,2)
        probmat_flat = probmat.view(batchsize, -1) # technically no need for this

        loss, lossu, lossv = KLDivLoss(torch.log(probmat_flat), GTprobmat_flat), None, None

    elif loss_type == 'pokerparams_WNHD_probs_joint_KLDiv':
        
        batchsize=others.size()[0]
        ncards=others.size()[1]

        GTcardprobs = Variable(others.double(), requires_grad=False)
        GTouter = torch.bmm(GTcardprobs.unsqueeze(2), GTcardprobs.unsqueeze(1))
        GTcorrection = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        GTnormalizing_constant = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            GTcorrection[i, :, :] = torch.diag(GTcardprobs[i, :]) ** 2
            GTnormalizing_constant[i, :, :] = torch.diag(1./(1.0 - GTcardprobs[i, :]))
        GTprobmat = torch.bmm(GTnormalizing_constant, GTouter-GTcorrection)
        GTprobmat_flat = GTprobmat.view(batchsize, -1) # technically no need for this

        outer = torch.bmm(cardProbs.unsqueeze(2), cardProbs.unsqueeze(1))
        correction = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        normalizing_constant = Variable(torch.DoubleTensor(np.zeros([batchsize, ncards, ncards])))
        for i in range(batchsize): # TODO: vectorize. As of Jan 2018, no block version of diag exists
            correction[i, :, :] = torch.diag(cardProbs[i, :]) ** 2
            normalizing_constant[i, :, :] = torch.diag(1./(1.0 - cardProbs[i, :]))
        probmat = torch.bmm(normalizing_constant, outer-correction)
        probmat_flat = probmat.view(batchsize, -1) # technically no need for this

        loss, lossu, lossv = KLDivLoss(torch.log(probmat_flat), GTprobmat_flat), None, None

    elif loss_type == 'target_rewards':
        GTtarget_rewards = Variable(others.double(), requires_grad=False)
        loss, lossu, lossv = mse(target_rewards, GTtarget_rewards), None, None

    else: assert False, 'Invalid loss type specified'

    return loss, lossu, lossv


def StagewiseAccuracy(pI, pA, cI, cA, seq, truth):
    '''
    Get all behavioral strategies along the trajectory #
    :param pI:
    :param pA:
    :param cI:
    :param cA:
    :param seq:
    :param truth: index of sequence which was finally taken
    :return: number of accurate predictions, number of predictions
    '''

    if truth is None or truth < 0:
        return 0, 0

    i = pA[truth]
    parAction = pI[i]
    cntCorrect, cntPred = StagewiseAccuracy(pI, pA, cI, cA, seq, parAction)

    # Get max probability
    best_action = np.argmax(seq[cI[i]])
    # print(best_action)
    best_action = cI[i][best_action]
    if best_action == truth:
        cntCorrect += 1
    cntPred += 1

    # print(cntCorrect , cntPred)

    return cntCorrect, cntPred

def IndividualLogloss(pI, pA, cI, cA, seq, truth):

    if truth is None or truth < 0:
        return [], []

    i = pA[truth]
    parAction = pI[i]
    loglosses, infosets = IndividualLogloss(pI, pA, cI, cA, seq, parAction)

    if parAction < 0 or parAction is None:
        beh = seq[truth]
    else:
        beh = seq[truth]/seq[parAction]

    loglosses.append(np.log(beh))
    infosets.append(i)

    return loglosses, infosets



if __name__ == '__main__':
    print('Old tests removed.')    
