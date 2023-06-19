from datetime import datetime
import gym
import gym
import torch
import json
import numpy as np
from tqdm import trange
#import yaml

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns

from torch.utils.tensorboard import SummaryWriter

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        #policy_filename = os.path.join(args.output_folder, 'policy.th')
        #config_filename = os.path.join(args.output_folder, 'config.json')
        #log_filename = os.path.join(args.output_folder, 'logs.txt')
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        summary_file_path = os.path.join(args.output, 'tensorboard')
        baseline_filename = os.path.join(args.train_output_folder, 'baseline.pth')
        #with open(config_filename, 'w') as f:
        #    config.update(vars(args))
        #    json.dump(config, f, indent=2)
    writer = SummaryWriter(summary_file_path +
                           datetime.now().strftime("%y-%m-%d-%H-%M"))

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))
    if os.path.exists(baseline_filename):
        print("load exist baseline")
        with open(baseline_filename, 'rb') as f:
            baseline.weight = torch.load(f)
            
    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)
    
    logs = {'tasks': []}
    train_returns, valid_returns = [], []
    dumb_returns, valid_dumb_returns = [], []
    epoch = 1
    for batch in trange(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)

        dumb_episodes, valid_dumb_episodes = sampler.sample(tasks,
                                                            num_steps=config['num-steps'],
                                                            fast_lr=config['fast-lr'],
                                                            gamma=config['gamma'],
                                                            gae_lambda=config['gae-lambda'],
                                                            device=args.device,
                                                            dumb_action=True)

        dumb_returns.append(get_returns(dumb_episodes[0]))
        valid_dumb_returns.append(get_returns(valid_dumb_episodes))

        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=config['num-steps'],
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)

        logs['tasks'].extend(tasks)
        train_returns.append(get_returns(train_episodes[0]))
        valid_returns.append(get_returns(valid_episodes))

        
        writer.add_scalar("reward/sample_train",
                          get_returns(train_episodes[0]).mean(), epoch)
        writer.add_scalar("reward/sample_valid",
                          get_returns(valid_episodes).mean(), epoch)

        writer.add_scalar("reward/dumb",
                          get_returns(dumb_episodes[0]).mean(), epoch)
        writer.add_scalar("reward/dumb_valid",
                          get_returns(valid_dumb_episodes).mean(), epoch)
        epoch += 1

    logs['train_returns'] = np.concatenate(train_returns, axis=0)
    logs['valid_returns'] = np.concatenate(valid_returns, axis=0)
    logs['dumb_returns'] = np.concatenate(dumb_returns, axis=0)
    logs['valid_dumb_returns'] = np.concatenate(valid_dumb_returns, axis=0)


    with open(args.output, 'wb') as f:
        np.savez(f, **logs)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file')
    parser.add_argument('--policy', type=str, required=True,
        help='path to the policy checkpoint')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=10,
        help='number of batches (default: 10)')
    evaluation.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch (default: 40)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--train-output-folder', type=str,
                      help='name of the train output folder')
    misc.add_argument('--output', type=str, required=True,
        help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=3,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)