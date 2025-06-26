build-docker:
	docker build --build-arg USERNAME=$$(whoami) -t vladbuff:latest docker/

start-docker:
	docker start -ai vladbuff || docker run --gpus all --name vladbuff -u $$(whoami) -it -v $$(pwd):/workspace -v ~/workspaces/datasets:/datasets -v /mnt/datasets:/mnt/datasets --shm-size=16g vladbuff:latest

run-docker:
	docker run -it vladbuff:latest bash

build-sif:
	apptainer build vladbuff.sif Apptainer.def

train-in-sif:
	apptainer exec --nv -B $(PWD):/workspace -B /mnt/beegfs/xliu0001/datasets:/mnt/datasets vladbuff.sif pixi run train-xfeat

test-cuda-in-sif:
	apptainer exec --nv -B $(PWD):/workspace vladbuff.sif pixi run test

sync-to-hpc:
	rsync -avz --exclude=".git" --exclude="logs" --exclude=".pixi" ./ xliu0001@turing1.ucy.ac.cy:/mnt/beegfs/xliu0001/vladbuff

sing-train:
	singularity exec --nv \
  -B "$(PWD)":/workspace \
  -B /mnt/beegfs/xliu0001/datasets:/mnt/datasets \
  vladbuff.sif \
  bash -c "cd /workspace && pixi run train-xfeat"

sing-test:
	singularity exec --nv \
  -B "$(PWD)":/workspace \
  -B /mnt/beegfs/xliu0001/datasets:/mnt/datasets \
  vladbuff.sif \
  bash -c "cd /workspace && pixi run test"

sing-bash:
	singularity exec --cleanenv --nv \
  -B "$(PWD)":/workspace \
  -B /mnt/beegfs/xliu0001/datasets:/mnt/datasets \
  vladbuff.sif \
  bash 

sing-test-cuda:
	singularity exec --nv \
		-B "$(PWD)":/workspace \
		-B /mnt/beegfs/xliu0001/datasets:/mnt/datasets \
		vladbuff.sif \
		bash -c 'cd /workspace && pixi run python -c "import torch; x = torch.randn(4096, 4096, device=\"cuda\"); print(\"OK\")"'

ls-job:
	squeue -u ${USER}

srun-bash:
	srun --partition=GPU --gres=gpu:v100:1 --pty bash

srun-pixi-install:
	srun --partition=GPU --gres=gpu:v100:1 --pty bash -c "cd /workspace && pixi install"

start-tb:
	cd logs && ./../pixi/bin/pixi run python -m tensorboard.main --logdir=logs --port=8008 --bind_all
