#!/bin/bash
export OMP_NUM_THREADS=1
for env_name in HopperQuantized-v0 ;
do
	for algo in hmtrpo localtrpo trpo ;
	do
		for batchsize in 500 ;
		do
		python run_script.py --env-name $env_name --agent-count 100 --num-workers 10  --max-kl 0.1 --algo $algo --seed 31 --use-running-state --max-episode 200 --step-size 20 --batch-size $batchsize ;
		done
	done
done

# python run_script.py --env-name HopperQuantized-v0 --agent-count 100 --num-workers 10  --max-kl 0.1 --algo hmtrpo --seed 31 --use-running-state --max-episode 200 --step-size 20 --batch-size 500

