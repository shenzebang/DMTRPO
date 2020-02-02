#!/bin/bash

for seed in 0 ; # 100 200 300 400 ;
do
    for env_name in HalfCheetah-v2 ; # Walker2d-v2 Ant-v2 Humanoid-v2 Swimmer-v2 HumanoidStandup-v2 Hopper-v2 ;
    do
        for alg in local_ppo global_ppo ;
        do
            echo $alg "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $alg \
                                    --agent 2 \
                                    --device cuda \
                                    --seed $seed \
                                    --reward_step 1
        done
    done
done