#!/bin/bash

for seed in 0 100 200 ; # 300 400 ;
do
    for env_name in Hopper-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2 HumanoidStandup-v2 ;
    do
        for algo in local_trpo dmtrpo global_trpo ;
        do
            cd /home/peng/Documents/rl_algorithms/trpo
            echo $algo "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $algo \
                                    --agent 12 \
                                    --device cuda \
                                    --seed $seed
        done
        for algo in local_ppo global_ppo ;
        do
            cd /home/peng/Documents/rl_algorithms/ppo
            echo $algo "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $algo \
                                    --agent 12 \
                                    --device cuda \
                                    --seed $seed
        done
    done
done