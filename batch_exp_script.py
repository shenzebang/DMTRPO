import os
os.system("export OMP_NUM_THREADS=1")
for env_name in ['Hopper-v2']:
    for algo in ['hmtrpo']:
        for max_episode in [200]:
            for batch_size in [500]:
                for seed in [1]:
                    os.system("python run_script.py --env-name {} --agent-count 200 --num-workers 25 --batch-size {} --seed {} --max-kl 1e-1 --max-episode {} --use-running-state --step-size 1.0"
                              .format(env_name, batch_size, seed, max_episode))

