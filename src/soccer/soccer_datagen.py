import numpy as np
import h5py
import scipy, scipy.io
import argparse
from ..experiments.datagen import dict_to_h5
import random
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Generate dataset.',
                                     epilog='''Sample usage:\n
                    python -m src.soccer.soccer_datagen --savepath="./data/soccer/all" --nSamples 10000
                    ''')
    parser.add_argument('--uGT', type=str,
                        default='92.,82.,112.',
                        help='Ground truth for min player (kicker)')
    # default='32.2,28.7,39.2',
    parser.add_argument('--vGT', type=str,
                        default='141.,18.,127.',
                        help='Ground truth for max player (keeper)')
    #default='49.3,6.3,44.4',
    parser.add_argument('--savepath', type=str,
                        required=True,
                        help='path to save .h5 output file')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='seed for randomization used for shuffling')
    parser.add_argument('--nSamples', type=int,
                        required=True
                        )
    args = parser.parse_args()

    uGT = [float(x) for x in args.uGT.split(',')]
    vGT = [float(x) for x in args.vGT.split(',')]
    uGT = uGT/np.sum(uGT)
    vGT = vGT/np.sum(vGT)

    np.random.seed(args.seed)
    d = dict()

    def ff(x):
        if x == 1: return 0
        elif x == 0: return 2
        elif x == 2: return 3

    tform = np.vectorize(ff)

    d['Au'] = tform(np.random.choice(range(3), size=[args.nSamples], p=uGT))
    d['Av'] = tform(np.random.choice(range(3), size=[args.nSamples], p=vGT))
    d['U'] = np.tile([uGT[0], uGT[1]+uGT[2], uGT[1], uGT[2]], (args.nSamples, 1))
    d['V'] = np.tile([vGT[0], vGT[1]+vGT[2], vGT[1], vGT[2]], (args.nSamples, 1))

    P = np.array([
        [-0.704, -1., -1.],
        [-0.902, -0.4, -0.968],
        [-1, -1, -0.746]]).astype(float)
    P_expanded = np.zeros((4,4))
    P_expanded[np.ix_([0,2,3], [0,2,3])] = P[np.ix_([1,0,2], [1,0,2])]
    d['P'] = np.tile(P_expanded, (args.nSamples, 1, 1))

    d['F'] = np.atleast_2d(np.ones(args.nSamples)).T
    # d['U'] = np.tile(uGT, (args.nSamples, 1))
    # d['V'] = np.tile(vGT, (args.nSamples, 1))

    f = h5py.File(args.savepath)
    print('Saving to %s' % args.savepath)
    dict_to_h5(d, f)

if __name__ == '__main__':
    main()
