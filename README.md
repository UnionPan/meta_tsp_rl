# meta_tsp_rl
meta reinforcement learning project for transportation


python train.py --config configs/maml/Turnpike.yaml --output-folder testoutput  --seed 1 --num-workers 5

python test.py --config testoutput/config.json --policy testoutput/policy.th --output testresult --train-output-folder testoutput

python ppo_turnpike.py