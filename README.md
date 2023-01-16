# KRLS: Improving E2E Response Generation in TOD with Reinforced Keywords Learning

This is the repository for the paper: [KRLS: Improving End-to-End Response Generation in Task Oriented
Dialog with Reinforced Keywords Learning](https://arxiv.org/abs/2211.16773)

# Datasets

We use the [MultiWoZ](https://github.com/budzianowski/multiwoz) dataset for training and evaluation. You can download already preprocessed dataset files from [HERE](https://drive.google.com/file/d/1OsACUuaMHjA-bggCVaAWpSYN2_XpjvQT/view?usp=sharing), and place them under the `dataset` folder like this:

```bash
dataset
├── MultiWOZ_2.0
├── MultiWOZ_2.1
└── MultiWOZ_2.2
```

If you want to preprocess the dataset by yourself, we used the same preprocessing scripts implemented by [Zhang et al., 2020](https://arxiv.org/abs/1911.10484), and [Lee, 2021](https://github.com/bepoetree/MTTOD). Please refer to their repositories for more details.

# Training

> Note: we track our experiments using `wandb` on [Weights & Biases](https://wandb.ai/). We highly recommend you to setup `wandb` to make sure our code works properly.

## Pretrained Weights

We use [GODEL-base](https://github.com/microsoft/GODEL) as pretrained weights, which you can download from the official github repo or from here [HERE](https://drive.google.com/file/d/18rtiS9twUVaecl1ycELskdcXwEW8Kr2T/view?usp=sharing). Place the downloaded weights under the `model_checkpoints` folder like this:

```bash
model_checkpoints
└── godel_cp
    └── GODEL-Base # here
```

In our experiments, we also train from a LM-finetuned checkpoint of GODEL-base, which you can download from [HERE](https://drive.google.com/file/d/1EjBOAIGpAjbgKSsUB4dTVdiHwE2SzL0v/view?usp=sharing). **This will be used for model training later**. Place the downloaded weights and place them under the `model_checkpoints` folder like this:

```bash
model_checkpoints
├── godel_cp
└── ppotod
    └──baseline # here
		├── ckpt-epoch8
		├── log.log
		└── run_config.json
```

## Training Script

To train with the best hyperparameters, run the following command:

```bash
MODEL_DIR = # where to save the model
python main.py -mode train \
	--config_path model_checkpoints/preset_configs/best.json \
	--model_dir $MODEL_DIR --log_file $MODEL_DIR/log.log \
	--exp_group_name debug \  # saves exp log to wandb
	--batch_size 4 --epochs 4 \
	--is_policy_optimization false 
```

We trained all of our models on a single NVIDIA RTX A4000, which takes about 1 day to train. Alternatively, you can directly download our best checkpoint from [HERE](https://drive.google.com/file/d/1N11DQPctJ5f-EUwSD2ppkQaFrMET4swH/view?usp=sharing), and place it under the `model_checkpoints` folder like this:

```bash
model_checkpoints
└── ppotod_reprod
	└── best # here
	    ├── ckpt-epoch2
	    └── preds
```

# Evaluation

We use the [standard multi-woz evaluation script](https://github.com/Tomiinek/MultiWOZ_Evaluation) to evaluate the predictions.

For example, to evaluate the best checkpoint:
```bash
CKPT_DIR = model_checkpoints/ppotod_reprod/best/ckpt-epoch2
```

1. to generate predictions (E2E response generation) from a checkpoint path:
	```bash
	python main.py -mode predict \
		--ckpt $CKPT_DIR --version 2.2 \
		--output $CKPT_DIR/preds_4.json \
		--batch_size 4 \
		--use_true_curr_aspn false \
		--is_policy_optimization false
	```
	to generate with gold DST (policy optimization):
	```bash
	python main.py -mode predict \
		--ckpt $CKPT_DIR --version 2.2 \
		--output $CKPT_DIR/preds_4.json \
		--batch_size 4 \
		--use_true_curr_aspn false \
		--is_policy_optimization true
	```
	to generate with both gold DST and gold system ACT:
	```bash
	python main.py -mode predict \
		--ckpt $CKPT_DIR --version 2.2 \
		--output $CKPT_DIR/preds_4.json \
		--batch_size 4 \
		--use_true_curr_aspn true \
		--is_policy_optimization true
	```
2. to evaluate the predictions:
	```bash
	python evaluate.py -input $CKPT_DIR/preds_4.json
	```

# Other Experiments

## Reward Function Ablation

You can swap out different reward functions which we implemented by:
```bash
python main.py -mode train \
	# other args omitted
	--reward token_prob  # here
```
available options include:
- "zeros": zero per-token reward, hence only using the terminal reward
- "token_error": hard penalty of ±μ for each correct/incorrect token
- "token_contextual_sim": BERTScore like contextual similarity
- "token_sim": cosine similarity from GODEL embeddings
- "token_prob": used in the paper

## Training with PG instead of PPO

Uncomment the following line in `main.py`:
```python
if __name__ == "__main__":
	# train with PPO
	# cfg, trainer = train_ppotod()
	# train with PG 
	cfg, trainer = train_pgtod() ### UNCOMMENT THIS LINE
```

## Training from Scratch/Backbone

To train from the backbone instead of the LM-finetuned checkpoint:
```bash
python main.py -mode train \
	--config_path model_checkpoints/preset_configs/best.json \
	--model_dir $MODEL_DIR --log_file $MODEL_DIR/log.log \
	--exp_group_name debug \  # saves exp log to wandb
	--batch_size 10 --epochs 4 \
	--is_policy_optimization false \
	--ckpt none
```

## Training with LM Only

You can train with only performing LM by switching off the ppo training. For example:
```bash
python main.py -mode train \
	--config_path model_checkpoints/preset_configs/best.json \
	--model_dir $MODEL_DIR --log_file $MODEL_DIR/log.log \
	--exp_group_name debug \  # saves exp log to wandb
	--batch_size 10 --epochs 4 \
	--is_policy_optimization false \
	--ckpt none --kl_loss_coeff 0.0 \
	--use_ppo false
```
finetunes the backbone only using the LM objective.

## Parameter Tuning

Please see `sweep.py` for the hyperparameter sweeps using ray tune and wandb. If you already have a wandb account and have setup ray tune on your machine, you can run the following command to start the sweep:
```bash
export CUDA_VISIBLE_DEVICES=#specify how many GPUs to use
python sweep.py
```