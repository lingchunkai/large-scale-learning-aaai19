#!/usr/bin/env bash



python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[30,30]' 'args.depth=2' 'args.mode="backward_exact"'  'args.sparse=True' 'args.tau_backward=0.1' 'args.lambda_range=[0.9,1.1]' 'args.num_samples=50' -F 'results_default/synthetic/dd_back/d2'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[25,25]' 'args.depth=2' 'args.mode="backward_exact"'  'args.sparse=True' 'args.tau_backward=0.1' 'args.lambda_range=[0.9,1.1]' 'args.num_samples=50' -F 'results_default/synthetic/dd_back/d2'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[20,20]' 'args.depth=2' 'args.mode="backward_exact"'  'args.sparse=True' 'args.tau_backward=0.1' 'args.lambda_range=[0.9,1.1]' 'args.num_samples=50' -F 'results_default/synthetic/dd_back/d2'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[15,15]' 'args.depth=2' 'args.mode="backward_exact"'  'args.sparse=True' 'args.tau_backward=0.1' 'args.lambda_range=[0.9,1.1]' 'args.num_samples=50' -F 'results_default/synthetic/dd_back/d2'
python -m src.fom_evaluation.experiments_synthetic with 'args.actions_size=[10,10]' 'args.depth=2' 'args.mode="backward_exact"'  'args.sparse=True' 'args.tau_backward=0.1' 'args.lambda_range=[0.9,1.1]' 'args.num_samples=50' -F 'results_default/synthetic/dd_back/d2'


