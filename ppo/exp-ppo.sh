#!/bin/bash

for env_name in Hopper-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2 HumanoidStandup-v2 ;
do
    for algo in local_ppo global_ppo ;
    do
        python parallel_main.py --env_name $env_name \
                                --agent 12 \
                                --device cuda
    done
done