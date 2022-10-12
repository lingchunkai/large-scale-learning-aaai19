#!/usr/bin/env bash

python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[3000,3000]' 'args.depth=1' 'args.mode="backward_exact"'  'args.sparse=True' 'args.tau_backward=0.1' 'args.lambda_range=[0.9,1.1]' 'args.num_samples=50' -F 'results_default/synthetic/dd_back/d1'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[2000,2000]' 'args.depth=1' 'args.mode="backward_exact"'  'args.sparse=True' 'args.tau_backward=0.1' 'args.lambda_range=[0.9,1.1]' 'args.num_samples=50' -F 'results_default/synthetic/dd_back/d1'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[1000,1000]' 'args.depth=1' 'args.mode="backward_exact"'  'args.sparse=True' 'args.tau_backward=0.1' 'args.lambda_range=[0.9,1.1]' 'args.num_samples=50' -F 'results_default/synthetic/dd_back/d1'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[500,500]' 'args.depth=1' 'args.mode="backward_exact"'  'args.sparse=True' 'args.tau_backward=0.1' 'args.lambda_range=[0.9,1.1]' 'args.num_samples=50' -F 'results_default/synthetic/dd_back/d1'

