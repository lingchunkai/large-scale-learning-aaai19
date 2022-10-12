#!/usr/bin/env bash

python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[15,15]' 'args.depth=3' 'args.mode="forward"'  'args.sparse=True' -F 'results_default/synthetic/forward/d3'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[12,12]' 'args.depth=3' 'args.mode="forward"'  'args.sparse=True' -F 'results_default/synthetic/forward/d3'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[10,10]' 'args.depth=3' 'args.mode="forward"'  'args.sparse=True' -F 'results_default/synthetic/forward/d3'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[5,5]' 'args.depth=3' 'args.mode="forward"'  'args.sparse=True' -F 'results_default/synthetic/forward/d3'

