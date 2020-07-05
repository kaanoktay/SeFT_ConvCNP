# Instructions

## Cluster setup

- Go inside the remote GPU server: (personal step, skip this if you are already in the GPU server)

  ```bash
  ssh -J koktay@bs-borgwardt02.ethz.ch koktay@bs-gpu08.ethz.ch
  ```

- Load the necessary modules:

  ```bash
  module use /usr/local/borgwardt/modulefiles
  module load python/3.7.7
  module load cuda/10.1
  ```

- Setup the GPU:

  - Check the available GPUs:

    ```bash
    nvidia-smi
    ```

  - Set `CUDA_VISIBLE_DEVICES` to a free GPU. For instance:

    ```bash
    export CUDA_VISIBLE_DEVICES=0
    ```

## Running the code

- Download and go inside the repository:

  ```bash
  git clone git@github.com:kaanoktay/SeFT_ConvCNP.git
  cd SeFT_ConvCNP
  ```

- Run the main file as below:

  ```bash
  poetry install
  poetry run main --batch_size 16 --points_per_hour 20 --num_epochs 5 --init_learning_rate 0.001 --kernel_size 5 --dropout_rate_conv 0.2 --dropout_rate_dense 0.2 --filter_size 64
  ```
