srun --nodes=1 --cpus-per-task=16 --gres=gpu:1 --ntasks-per-node=1 --mem=128G --time=0-06:00 --reservation=GPU seisbench-training
