#!/usr/bin/env bash

# python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[2,2]' 'args.depth=3' 'args.mode="backward_exact"'  -F 'results_default/synthetic'
# python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[42,42]' 'args.depth=1' 'args.mode="backward_exact"'  -F 'results_default/synthetic'
# python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[2,2]' 'args.depth=4' 'args.mode="backward_exact"'  -F 'results_default/synthetic'
# python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[170,170]' 'args.depth=1' 'args.mode="backward_exact"'  -F 'results_default/synthetic'
# python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[2,2]' 'args.depth=5' 'args.mode="backward_exact"'  -F 'results_default/synthetic'
# python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[682,682]' 'args.depth=1' 'args.mode="backward_exact"'  -F 'results_default/synthetic'
# python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[2,2]' 'args.depth=6' 'args.mode="backward_exact"'  -F 'results_default/synthetic'
# python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[2730,2730]' 'args.depth=1' 'args.mode="backward_exact"'  -F 'results_default/synthetic'

python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[3000,3000]' 'args.depth=1' 'args.mode="backward_exact"'  'args.sparse=True' -F 'results_default/synthetic/back/d1'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[2000,2000]' 'args.depth=1' 'args.mode="backward_exact"'  'args.sparse=True' -F 'results_default/synthetic/back/d1'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[1000,1000]' 'args.depth=1' 'args.mode="backward_exact"'  'args.sparse=True' -F 'results_default/synthetic/back/d1'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[500,500]' 'args.depth=1' 'args.mode="backward_exact"'  'args.sparse=True' -F 'results_default/synthetic/back/d1'

