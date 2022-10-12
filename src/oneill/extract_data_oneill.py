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
                    python -m src.oneill.extract_data_oneill --savepath="./data/oneill/all" 
                    ''')
    parser.add_argument('--data_path', type=str,
                        default='./data/ONeill_raw/Data and Programs O\'Neill 4x4 game.xls',
                        help='path for .xks file')
    parser.add_argument('--sheetname', type=str,
                        default='additional table for C1')
    parser.add_argument('--savepath', type=str,
                        required=True,
                        help='path to save .h5 output file')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='seed for randomization used for shuffling')
    args = parser.parse_args()

    raw = pd.read_excel(args.data_path, sheetname=args.sheetname)
    data = raw.iloc[3:-1, 2:4].values # hardcoded locations. TODO: change

    # Shuffle data
    np.random.seed(args.seed)
    np.random.shuffle(data)

    # Replace J with 0
    data[data == 'J'] = 0
    data[data >= 1] += 1

    ndatapoints = data.shape[0]
    d = dict()
    d['F'] = np.atleast_2d(np.ones(ndatapoints)).T
    d['Au'] = data[:, 0].squeeze().astype(int)
    d['Av'] = data[:, 1].squeeze().astype(int)

    print(d['Au'])

    f = h5py.File(args.savepath)
    print('Saving to %s' % args.savepath)
    dict_to_h5(d, f)

if __name__ == '__main__':
    main()