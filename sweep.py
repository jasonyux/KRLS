import os
import numpy as np

from data_utils.mttod.reader import MultiWOZReader
from models.ppotod import PPOTODModel
from trainers.ppotod_trainer import PPOLMTrainer
from utils.globals import _PPOTOD_CONFIG
from utils.utils import seed_everything
from main import __init_kl_model

from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.air.config import RunConfig


model_base_dir = "model_checkpoints_multilingual"
# model_base_dir = "model_checkpoints"
data_dir = "dataset_multilingual/mttod_multiwoz"
# data_dir = "dataset/mttod_multiwoz"


def update_config(args: dict):
	"""CONFIG doesn't work if not functional
	"""
	DEFAULT_CFG = _PPOTOD_CONFIG
	DEFAULT_CFG.mode = "train"
	DEFAULT_CFG.model_dir = f"{model_base_dir}/ppotod/debug"
	DEFAULT_CFG.backbone = f"{model_base_dir}/godel_cp/GODEL-Base"
	DEFAULT_CFG.save_model = False
	DEFAULT_CFG.log_file = f"{model_base_dir}/ppotod/debug/log.log"
	DEFAULT_CFG.token_embedding_path = f"{model_base_dir}/ppotod/embeddings/baseline_embedding_layer.pt"
	DEFAULT_CFG.alternate_epoch = False
	DEFAULT_CFG.use_sl = True
	DEFAULT_CFG.use_ppo = True
	DEFAULT_CFG.reward = "token_prob"
	DEFAULT_CFG.add_kl_divergence = True  # required by above
	DEFAULT_CFG.kl_ckpt = f"{model_base_dir}/ppotod/baseline/ckpt-epoch8"  # should be same as below
	DEFAULT_CFG.ckpt = f"{model_base_dir}/ppotod/baseline/ckpt-epoch8"
	DEFAULT_CFG.kl_loss_coeff = 0.0
	DEFAULT_CFG.lm_scale = "none"
	DEFAULT_CFG.val_watch_key = "total_score"
	DEFAULT_CFG.lm_head_model_action_value = False
	DEFAULT_CFG.adv_use_returns = True
	DEFAULT_CFG.add_terminal_reward = True
	DEFAULT_CFG.terminal_reward_scale = 1.0
	DEFAULT_CFG.sample_action = "sample"
	DEFAULT_CFG.sep_act_n_resp_gen = True
	DEFAULT_CFG.deterministic = False
	DEFAULT_CFG.skip_val_predictions = True  # only for debugging
	DEFAULT_CFG.save_val_predictions = False
	DEFAULT_CFG.score_each_dialog = False

	DEFAULT_CFG.epochs = 10
	DEFAULT_CFG.data_dir = data_dir
	DEFAULT_CFG.batch_size = 4
	DEFAULT_CFG.grad_accum_steps = 1
	DEFAULT_CFG.pilot_run = False  # whether to use a smaller dset
	DEFAULT_CFG.no_predict = True  # we are sweeping
	DEFAULT_CFG.exp_group_name = f"{'' if DEFAULT_CFG.pilot_run else 'full_'}token_prob_hparam_cmp"

	# update attributes from args to cfg
	for key, value in args.items():
		setattr(DEFAULT_CFG, key, value)
	return DEFAULT_CFG


def __adjust_epoch(cfg: _PPOTOD_CONFIG):
	# default is 5r-4 with 10 epochs
	if cfg.ckpt is not None:
		return 4
	# train from scratch
	return 10


def train_single(config: dict) -> None:
	os.chdir("/home/xy2437/6998_project")
	cfg: _PPOTOD_CONFIG = update_config(config)
	cfg.epochs = __adjust_epoch(cfg)

	seed_everything(cfg.seed, deterministic=cfg.deterministic)

	group_name = cfg.exp_group_name
	dataset = MultiWOZReader(cfg.backbone, cfg.version, base_dir=cfg.data_dir)
	test_dataset = MultiWOZReader(cfg.backbone, "2.2", base_dir=cfg.data_dir)
	model = PPOTODModel(cfg, dataset.vocab_size, dataset.tokenizer)
	kl_model = None
	if cfg.add_kl_divergence:
		assert(cfg.kl_ckpt is not None)
		print(f"loading kl model from {cfg.kl_ckpt}")
		kl_model = __init_kl_model(cfg, dataset)
	trainer = PPOLMTrainer(dataset, cfg, model, group_name, run_name=None, test_dataset=test_dataset, kl_model=kl_model)
	
	trainer.train()
	return


if __name__ == "__main__":
	search_space = {
		"ckpt": tune.choice([f"{model_base_dir}/ppotod/baseline/ckpt-epoch8"]),  # [None, f"{model_base_dir}/ppotod/baseline/ckpt-epoch8"]),
		"learning_rate": tune.choice([5e-5]),  # tune.choice([5e-4, 1e-4, 5e-5]),
		"grad_accum_steps": tune.choice([1]),  # tune.choice([1, 2]),
		"dropout": tune.choice([0.1]),  # tune.choice([0.1, 0.2]),
		"reward": tune.choice(["token_prob"]),  # "token_prob", "token_confidence"
		"correct_reward": tune.choice([1.0]),
		"token_prob_scale": tune.choice([1.0]),  # tune.choice([0.5, 0.8]),
		"token_prob_temperature": tune.choice([1.0]),  # in pilot_sweep_2, 0.8 works better than 0.7
		"normalize_return": tune.choice([True]),
		"real_normalize_return": tune.choice([False]), # should stay False
		"resp_loss_coeff": tune.choice([5.0]),  # [1.0, 5.0, 10.0, 15.0] mainly to get a balance with belief states. Lower if ckpt is not None
		"special_token_error_scale": tune.grid_search([2.0, 10.0]),
		"terminal_reward_scale": tune.choice([5.0]),  # in pilot_sweep_2, 5.0 works well. larger seems to make network confused what is key token
		"terminal_reward_fn": tune.choice(["sp_f1"]),
		"penalize_da_tokens": tune.choice([True]),
		"sample_temperature": tune.choice([1.1]),
		"sample_temperature_decay": tune.choice([1.0]),
		"kl_loss_coeff": tune.choice([0.01]),  # in pilot_sweep_2, 0.01 works better than 0.0 or 0.1
		"ppo_epoch": tune.choice([1]),
		"rl_gamma": tune.choice([0.99]),
		"alternate_step_k": tune.grid_search([0.1, 1.0]),  # in pilot_sweep_2, 0.5 seems to work better
		"update_old_policy_interval": tune.choice([1]),
		"num_per_sample": tune.grid_search([3]),
		# "use_sl": tune.choice([False]),
		"is_policy_optimization": tune.choice([False]),
	}

	trainable_with_cpu_gpu = tune.with_resources(
		train_single,
		{"cpu": 1, "gpu": 1.0}
	)

	tuner = tune.Tuner(
		trainable_with_cpu_gpu,
		param_space=search_space,
		tune_config=tune.TuneConfig(
			# metric="_metric/total_score",
			# mode="max",
			# search_alg=BasicVariantGenerator(constant_grid_search=True),
			num_samples=1,
			max_concurrent_trials=4
		),
		run_config=RunConfig(
			local_dir="./ray_results",
			# include_dashboard=False,
		)
	)
	results = tuner.fit()
