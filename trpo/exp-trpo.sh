#!/bin/bash
for env_name in HalfCheetah-v2 ; #Walker2d-v2 Ant-v2 Humanoid-v2 Swimmer-v2 HumanoidStandup-v2 Hopper-v2 ;
do
    for alg in local_trpo local_trpo2 local_trpo3 hmtrpo global_trpo ;
    do
        python parallel_main.py --env_name $env_name \
                                --alg $alg \
                                --agent 2 \
                                --device cuda \
                                --reward_step 1
    done
done