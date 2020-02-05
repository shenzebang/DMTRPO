#!/bin/bash
for env_name in Hopper-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2 Swimmer-v2 Humanoid-v2 HumanoidStandup-v2 ;
do
    python plot.py --alg-list local_trpo3 dmtrpo global_trpo local_ppo global_ppo \
                --env-name $env_name \
                --workers 12
done