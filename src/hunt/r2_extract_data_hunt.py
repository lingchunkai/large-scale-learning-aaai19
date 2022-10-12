import numpy as np
import h5py
import scipy, scipy.io
from .r2_expts_hunt import GetAAGame, GetABGame, ops
import argparse
from ..experiments.datagen import dict_to_h5
import  random

''' Abstracts dataset into smaller h5 files'''

def main():
    parser = argparse.ArgumentParser(description='Generate dataset.',
            epilog='''Sample usage:\n
                    python r2_extract_data_hunt.py --savepath=./data/default_hunt.h5 --exptType='AA' --exptObj='2'
                    ''')
    parser.add_argument('--data_path', type=str, 
                    default='./data/Hunt_infotask/TGBE_cardturn_data_54145.mat',
                    help='path for .mat file')
    parser.add_argument('--exptType', type=str, 
                    required=True,
                    help='''Experiment type, AA or AB''')
    parser.add_argument('--standardForm', type=int,
                        required=True,
                        help= '''Flag if we are using standard form''')
    parser.add_argument('--seed', type=int,
                        required=True,
                        help='''Random seed''')
    parser.add_argument('--exptObj', type=int,
                    required=True,
                    help='''
                    Experiment objective:\n 
                    0 = FIND THE BIGGEST,\n
                    1 = FIND THE SMALLEST,\n
                    2 = ADD BIG,\n
                    3 = ADD SMALL\n, 
                    4 = MULTIPLY BIG\n, 
                    5 = MULTIPLY SMALL''')
    parser.add_argument('--savepath', type=str,
                        required=True,
                        help='path to save .h5 output file')
    args = parser.parse_args()

    z = scipy.io.loadmat(args.data_path)
    filters = []
    d = Extract(z, args.exptType, args.exptObj, filters, shuffle_seed=args.seed)

    if args.standardForm == 1:
        d = ConvertToStandardFeatures(d)

    f = h5py.File(args.savepath)
    print('Saving to %s' % args.savepath)
    dict_to_h5(d, f)


def Extract(z, exptType_filter, exptObj_filter, filters, shuffle_seed):
    '''
    Primary filters
    ---------------
    exptType: AA or AB
    exptObj: 0-5 inclusive, objective of player
    secondary filters
    ---------------
    filters: List of functions which return true if we want to include in the dataset
    e.g. only a certain age group, location, education etc.
    '''
    
    if exptType_filter == 'AA':
        pI, pA, cI, cA, infosets_d, actions_d, _, _ = GetAAGame(exptObj_filter)
    elif exptType_filter == 'AB':
        pI, pA, cI, cA, infosets_d, actions_d, _, _ = GetABGame(exptObj_filter)
    else: assert False, 'Type must be AA or AB'

    d = {'vA': [], 'age': [], 'education': [], 'gender': [], 'location': [], 'trialnum': []}
    nRuns = len(z['trmovelist'])
    # Extract data from .mat file
    for i in range(nRuns):
        nTrials = len(z['trmovelist'][i][0][0])
        for j in range(nTrials):
            # We 'normalize' this for the starting card as the top left
            flipV, flipH = False, False
            
            # Experiment objective: 0 = FIND THE BIGGEST, 1 = FIND THE SMALLEST, 2 = ADD BIG, 3 = ADD
            # SMALL, 4 = MULTIPLY BIG, 5 = MULTIPLY SMALL
            exptObj = z['trtrialtype'][i, j]
            
            # All cards 
            trcardlist = np.reshape(z['trcardlist'][i][:, j], [2,2])
            
            # Moves which were taken, not normalized yet.
            movelist= z['trmovelist'][i][0][0][j][0]
            nCardsOpened = z['trNumMove'][i][j]
            trsecondcard = z['trsecondcard'][i][j]
            succ = z['trsuccess'][i,j] # whether he answered correctly.

            # 0 = top left, 1 = top right, etc.
            if movelist[0] >= 2: # 2 or above is second row, flip everything from here
                flipV = True
            if movelist[0] == 1 or movelist[0] == 3:
                flipH = True
            
            # check if type 1 or type 2.
            if flipV == False and trsecondcard >= 2: # TOP -- BOTTOM
                exptType = 'AB'
            elif flipV == True and trsecondcard < 2: # BOTTOM -- TOP
                exptType = 'AB'
            else: 
                exptType = 'AA'

            # Grab features.
            age = z['age'][i]
            education = z['education'][i]
            gender = z['gender'][i]
            location = z['location'][i]
            tnum = j
            # Apply primary filters
            if exptType != exptType_filter: continue
            if exptObj != exptObj_filter: continue
            
            # Apply secondary filters
            for f in filters:
                if not f(d): continue

            action_raw = ConvertToAction(exptType, movelist, trcardlist, exptObj, succ, exptObj_filter)
            action_id = actions_d[action_raw]
            
            # only take actions > 4 steps 
            # if len(action_raw[0]) < 4: continue

            d['age'].append(age)
            d['education'].append(education)
            d['gender'].append(gender)
            d['location'].append(location)
            d['vA'].append(action_id)
            d['trialnum'].append(tnum)

    random.seed(shuffle_seed)
    random.shuffle(d['age'])
    random.seed(shuffle_seed)
    random.shuffle(d['education'])
    random.seed(shuffle_seed)
    random.shuffle(d['gender'])
    random.seed(shuffle_seed)
    random.shuffle(d['location'])
    random.seed(shuffle_seed)
    random.shuffle(d['vA'])
    random.seed(shuffle_seed)
    random.shuffle(d['trialnum'])

    return d


def ConvertToStandardFeatures(d, formatting=None):
    '''
    Converts Extracted features and data to standard Dataset
    :param d: original features
    :param formatting: TODO
    :return: dictionary with fields F, Au, Av
    '''

    ret = dict()
    ret['Av'] = d['vA']
    ret['Au'] = np.zeros(len(d['vA']))

    # Convert features to long vector
    ret['F'] = np.stack([np.squeeze(d['age']),
                         np.squeeze(d['education']),
                         np.squeeze(d['gender']),
                         np.squeeze(d['location']),
                         d['trialnum']], axis=-1)

    return ret

                
def ConvertToAction(exptType, movelist, cardlist_, exptObj, succ, op):
    # exptType in {AA, AB}
    # exptObj : highest lowest
    cardlist = np.mod(cardlist_-1, 10)

    def GetWinningRow(cardlist):
        '''
        Return winning row from unnormalized card layout
        '''
        R1 = cardlist[0, :]
        R2 = cardlist[1, :]
        return ops[op](R1, R2)

    def xCoord(n):
        return n%2, n//2

    move_coord = [xCoord(i) for i in movelist]

    # Get all cards seen
    cards_seen = []
    for x,y in move_coord:
        cards_seen.append(cardlist[y,x])

    # Compute whether we should flip the card positions vertically
    flipV = False
    if movelist[0] >= 2: # 2 or above is second row, flip everything from here
        flipV = True

    # Get final guess of which row wins
    winning_row = GetWinningRow(cardlist)
    # normalize row
    if (winning_row == 1 and flipV == False) or (winning_row == -1 and flipV == True):
        # First (normalized) row is winning
        if succ > 0: lastmove = 'GS'
        else: lastmove = 'GD'
    elif (winning_row == -1 and flipV == False) or (winning_row == 1 and flipV == True):
        # Second (normalized) row is winning
        if succ > 0: lastmove = 'GD'
        else: lastmove = 'GS'
    else:
        assert False, 'Invalid combination!'

    # Get previous actions
    # First action is by default the first row (normalized)
    norm_moves = []
    for x,y in move_coord:
        if flipV: y = 1-y

        if y == 0: norm_moves.append('OS')
        else: norm_moves.append('OD')

    norm_moves = norm_moves[1:] # ignore first one

    ############################
    ############################
    # Obtain last action played for dataset.
    # We may extract intermediate infosets and actions if needed
    # 1) norm_moves (normalized moves)
    # 2) cards_seen

    A = (tuple(cards_seen), tuple(norm_moves) + (lastmove,))

    return A

if __name__ == '__main__': 
    main()
