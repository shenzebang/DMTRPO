#!/bin/bash

for seed in 0 100;
do
    for env_name in Navigation2DEnv-FL ;
    do
        for alg in local_ppo global_ppo ;
        do
            rm /home/peng/anaconda3/envs/pytorch/lib/python3.6/site-packages/mujoco_py/generated/*.lock
            cd /home/peng/Documents/trpo_ppo_sac/ppo
            echo "Experiment:" $alg "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $alg \
                                    --workers 16 \
                                    --device cuda \
                                    --seed $seed \
                                    --episode 500
        done
        for alg in local_trpo3 hmtrpo global_trpo ;
        do
            rm /home/peng/anaconda3/envs/pytorch/lib/python3.6/site-packages/mujoco_py/generated/*.lock
            cd /home/peng/Documents/trpo_ppo_sac/trpo
            echo "Experiment:" $alg "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $alg \
                                    --workers 16 \
                                    --device cuda \
                                    --seed $seed \
                                    --episode 500
        done
    done
done