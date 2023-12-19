# DMM

An attempt to use Diffusion Models to predict the parameters of other models.

The model structure and algorithms are mainly inspired by [DDIM](https://github.com/ermongroup/ddim) and [U-ViT](https://github.com/baofff/U-ViT).

### Dependency

To reproduce the training process of DMM, we should install these dependencies first by typing these commands:

```shell
pip install torch accelerate ml_collections einops tqdm
```

### Training

Before running our training code, we should use some simple scripts to generate the dataset we need.
In fact, they are just the weights of other smaller models focusing on a specific task.
For example, you can run these commands in the root folder of this project

```shell
cd assets/scripts
python3 MNIST_linear_models.py
```

By doing this, the python script would automatically download the MNIST dataset and training 2048 different linear models for the classification of MNIST dataset, and save their weights to a predifined path.
We can also modify the parameters in these scripts if we want to obtain more data for training.
After that we can go back to the root folder and start training our diffusion models.

Here we listed the commands used in our training process:

```shell
accelerate launch --multi_gpu --num_processes 2 main.py --config mnist-linear --train # used for linear models on MNIST dataset
```
