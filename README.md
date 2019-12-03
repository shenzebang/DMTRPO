# PyTorch implementation of Harmonic Mean TRPO
Additional environments:
1. Mujoco with biased reward
2. Mujoco with quantized reward
3. 2D-Navigation
## Usage

```
python run_script.py --algo hmtrpo --env-name Hopper-v2 --agent-count 200 --num-workers 10 --max-kl 1e-2 --batch-size 300 --plot --max-episode 200 --init-std 10.0 --device cuda --cg-iter 20 --num-repeat 10 
```

## Recommended hyper parameters

Hopper-v2:

## Results


## Todo
