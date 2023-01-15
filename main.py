import argparse
import os
import json
import wandb

from typing import NamedTuple
from data_utils.mttod.reader import MultiWOZReader
from models.mttod import MTTODModel
from models.pgmttod import PGMTTODModel
from models.ppotod import PPOTODModel
from trainers.mttod_trainer import MTTODTrainer
from trainers.pgtod_trainer import PGLMTrainer
from trainers.ppotod_trainer import PPOLMTrainer
from trainers.realrl_trainer import PPORealRLTrainer
from utils.globals import _MTTOD_CONFIG, _PPOTOD_CONFIG, _PGTOD_CONFIG
from utils.utils import save_cfg, str2bool, str2number, str2none, seed_everything


def __update_base_dirs(path, new_base_dir):
	if path is None or path == '':
		return None
	base_dir_idx = path.find('/')
	path = path[base_dir_idx+1:]
	return os.path.join(new_base_dir, path)


def update_necessary_fields(args, run):
	# necessary fields include data_dir, model_dir, save_model, log_file, max_to_keep_ckpt, exp_group_name

	# update all the directories
	args.backbone = __update_base_dirs(run.config['backbone'], args.model_base_dir)
	args.token_embedding_path = __update_base_dirs(run.config['token_embedding_path'], args.model_base_dir)
	args.kl_ckpt = __update_base_dirs(run.config['kl_ckpt'], args.model_base_dir)
	args.ckpt = __update_base_dirs(run.config['ckpt'], args.model_base_dir) if args.ckpt else None
	print(f"""
		Overwriting necessary fields:
		data_dir: {run.config['data_dir']} to {args.data_dir}
		model_dir: {run.config['model_dir']} to {args.model_dir}
		log_file: {run.config['log_file']} to {args.log_file}
		backbone: {run.config['backbone']} to {args.backbone}
		token_embedding_path: {run.config['token_embedding_path']} to {args.token_embedding_path}
		kl_ckpt: {run.config['kl_ckpt']} to {args.kl_ckpt}
		ckpt: {run.config['ckpt']} to {args.ckpt}

		You might have changed the following fields:
		save_model: {run.config['save_model']} to {args.save_model}
		max_to_keep_ckpt: {run.config['max_to_keep_ckpt']} to {args.max_to_keep_ckpt}
		pilot_run: {run.config['pilot_run']} to {False}
		debug: {run.config['debug']} to {args.debug}
		batch_size: {run.config['batch_size']} to {args.batch_size}
		epochs: {run.config['epochs']} to {args.epochs}
		exp_group_name: {run.config['exp_group_name']} to {args.exp_group_name}
		save_val_predictions: {run.config['save_val_predictions']} to {args.save_val_predictions}
		score_each_dialog: {run.config['score_each_dialog']} to {args.score_each_dialog}

		You are probably changing those for ablation study:
		reward: {run.config['reward']} to {args.reward}
		use_ppo: {run.config['use_ppo']} to {args.use_ppo}
		use_lm: {run.config['use_lm']} to {args.use_lm}
		alternate_step_k: {run.config['alternate_step_k']} to {args.alternate_step_k}
		num_per_sample: {run.config['num_per_sample']} to {args.num_per_sample}
		terminal_reward_fn: {run.config['terminal_reward_fn']} to {args.terminal_reward_fn}
		is_policy_optimization: {run.config.get('is_policy_optimization')} to {args.is_policy_optimization}
		use_true_curr_aspn: {run.config.get('use_true_curr_aspn')} to {args.use_true_curr_aspn}
		kl_loss_coeff: {run.config.get('kl_loss_coeff')} to {args.kl_loss_coeff}
	""")
	run.config['backbone'] = args.backbone
	run.config['token_embedding_path'] = args.token_embedding_path
	run.config['kl_ckpt'] = args.kl_ckpt
	run.config['ckpt'] = args.ckpt

	run.config['save_model'] = args.save_model
	run.config['max_to_keep_ckpt'] = args.max_to_keep_ckpt
	run.config['pilot_run'] = False
	run.config['debug'] = args.debug
	run.config['batch_size'] = args.batch_size
	run.config['epochs'] = args.epochs
	run.config['exp_group_name'] = args.exp_group_name
	run.config['save_val_predictions'] = args.save_val_predictions
	run.config['score_each_dialog'] = args.score_each_dialog

	run.config['reward'] = args.reward
	run.config['use_ppo'] = args.use_ppo
	run.config['use_lm'] = args.use_lm
	run.config['alternate_step_k'] = args.alternate_step_k
	run.config['num_per_sample'] = args.num_per_sample
	run.config['terminal_reward_fn'] = args.terminal_reward_fn
	run.config['is_policy_optimization'] = args.is_policy_optimization
	run.config['use_true_curr_aspn'] = args.use_true_curr_aspn
	run.config['kl_loss_coeff'] = args.kl_loss_coeff
	return run

def get_arg_configs(cfg:_MTTOD_CONFIG, parser:argparse.ArgumentParser):
	parser.add_argument('-mode', type=str, required=True, default='train', choices=["train", "predict"])

	# only used for retraining from wandb or local config files
	parser.add_argument('--wandb_url', type=str, required=False, default=None, help="wandb url for reproducibility")
	parser.add_argument('--config_path', type=str, required=False, default=None, help="config path for reproducibility")
	parser.add_argument('--model_base_dir', type=str, required=False, default="model_checkpoints", help="model base directory. different in different server")

	# general
	parser.add_argument('--seed', type=int, required=False, default=cfg.seed)
	parser.add_argument('--debug', action="store_true", default=cfg.debug, help="enable debug mode")
	parser.add_argument('--pilot_run', action="store_true", default=cfg.pilot_run, help="pilot_run using a small training set")
	parser.add_argument('--version', type=str, required=False, default=cfg.version, choices=['2.1', '2.2'])
	parser.add_argument('--ckpt', type=str2none, required=False, default=cfg.ckpt, help='checkpoint path for model')
	parser.add_argument('--device', type=str, required=False, default=cfg.device, help='cpu, cuda, cuda:0, cuda:1, ...')
	# training specific
	parser.add_argument('--learning_rate', type=float, required=False, default=cfg.learning_rate)
	parser.add_argument('--data_dir', type=str, required=False, default=cfg.data_dir)
	parser.add_argument('--deterministic', type=str2bool, required=False, default=cfg.deterministic, help='use deterministic mode or not')
	parser.add_argument('--val_watch_key', type=str, required=False, default=cfg.val_watch_key, help="key to watch for validation, in the format of a.b.c if nested")
	parser.add_argument('--resp_use_bspn', type=str2bool, required=False, default=cfg.resp_use_bspn, help='whether to use dspn in decoder')
	parser.add_argument('--use_tod_loss_scale', type=str2bool, required=False, default=cfg.use_tod_loss_scale, help='whether if to scale resp loss by tod error')
	parser.add_argument('--log_file', type=str, required=False, default=cfg.log_file, help='where to log the training process')
	parser.add_argument('--save_model', type=str2bool, required=False, default=cfg.save_model, help='whether to save_model')
	parser.add_argument('--model_dir', type=str, required=False, default=cfg.model_dir, help="directory to save model")
	parser.add_argument('--backbone', type=str, required=False, default=cfg.backbone)
	parser.add_argument('--batch_size', type=int, required=False, default=cfg.batch_size)
	parser.add_argument('--grad_accum_steps', type=int, required=False, default=cfg.grad_accum_steps)
	parser.add_argument('--epochs', type=int, required=False, default=cfg.epochs, help="number of training epochs")
	parser.add_argument('--max_to_keep_ckpt', type=int, required=False, default=cfg.max_to_keep_ckpt, help="maximum number of checkpoints to keep")
	parser.add_argument('--resp_loss_coeff', type=float, required=False, default=cfg.resp_loss_coeff, help="loss weight on response generation")
	parser.add_argument('--no_validation', action='store_true', help="skip validation")
	parser.add_argument('--no_predict', action='store_true', help="skip prediction")
	parser.add_argument('--score_each_dialog', type=str2bool, required=False, default=cfg.score_each_dialog, help="whether to score each dialog in addition to whole session")
	parser.add_argument('--save_val_predictions', type=str2bool, required=False, default=cfg.save_val_predictions, help="whether to save validation predictions")
	# prediction specific
	parser.add_argument('--sep_act_n_resp_gen', type=str2bool, required=False, default=cfg.sep_act_n_resp_gen, help="file to save predictions")
	parser.add_argument('--output', type=str, required=False, default=cfg.output, help='output directory for usually output predictions')
	parser.add_argument('--repetition_penalty', type=float, required=False, default=cfg.repetition_penalty, help='repetition_penalty for generation')
	parser.add_argument('--top_p', type=float, required=False, default=cfg.top_p, help='top_p for generation')

	## if it is prediction, then we expect a run_config file to be found in the checkpoints. Those should be the default instead
	args = parser.parse_args()
	if args.mode == 'predict':
		# load the run_config file from the checkpoint
		checkpoint_root_path = os.path.dirname(args.ckpt)
		run_config_path = os.path.join(checkpoint_root_path, 'run_config.json')
		json_args = argparse.Namespace()
		with open(run_config_path, 'r') as f:
			json_args.__dict__ = json.load(f)
		return parser.parse_args(namespace=json_args)
	elif args.wandb_url is not None:
		# load the run_config file from the wandb url
		print(f"Loading run_config from wandb url: {args.wandb_url}")
		api = wandb.Api()
		run = api.run(args.wandb_url)
		# # still needs to change, IF we are training in a new server
		update_necessary_fields(args, run)
		print(f"using config: {json.dumps(run.config, indent=4)}")

		json_args = argparse.Namespace()
		json_args.__dict__ = run.config
		return parser.parse_args(namespace=json_args)
	elif args.config_path is not None:
		# load the run_config file from the wandb url
		print(f"Loading run_config from path: {args.config_path}")
		run = NamedTuple('Config', [('config', dict)])
		json_config = json.load(open(args.config_path, 'r'))
		run.config = json_config

		# # still needs to change, IF we are training in a new server
		update_necessary_fields(args, run)
		print(f"using config: {json.dumps(run.config, indent=4)}")

		json_args = argparse.Namespace()
		json_args.__dict__ = run.config
		return parser.parse_args(namespace=json_args)
	else:
		return args


def update_config(args, cfg):
	# update attributes from args to cfg
	if not isinstance(args, dict):
		args = vars(args)
	for key, value in args.items():
		setattr(cfg, key, value)
	return cfg


def train_mttod():
	cfg = _MTTOD_CONFIG
	parser = argparse.ArgumentParser()
	args = get_arg_configs(cfg, parser)
	cfg = update_config(args, cfg)

	dataset = MultiWOZReader(cfg.backbone, cfg.version)
	model = MTTODModel(cfg, dataset.vocab_size, dataset.tokenizer)
	trainer = MTTODTrainer(dataset, cfg, model)
	return cfg, trainer


def add_ppotod_arg_configs(cfg: _PPOTOD_CONFIG, parser: argparse.ArgumentParser):
	parser.add_argument('--dropout', type=float, required=False, default=cfg.dropout)
	parser.add_argument('--ppo_epoch', type=int, default=cfg.ppo_epoch, help="number of ppo epochs per lm epoch, or steps per lm steps")
	parser.add_argument('--adv_use_returns', type=str2bool, default=cfg.adv_use_returns, help="use returns as advantage, and value loss is zero")
	parser.add_argument('--normalize_return', type=str2bool, default=cfg.normalize_return, help="'normalize' return by dividing the number of elements in a batch")
	parser.add_argument('--real_normalize_return', type=str2bool, default=cfg.real_normalize_return, help="normalize returns to have zero mean and unit variance")
	parser.add_argument('--use_gold_prob', type=float, default=cfg.use_gold_prob, help="probability of using gold response during data collection")
	parser.add_argument('--add_gold_demonstrations', type=str2bool, default=cfg.add_gold_demonstrations, help="use gold demonstrations in addition to collected experiences")
	parser.add_argument('--update_old_policy_interval', type=int, default=cfg.update_old_policy_interval, help="how often to update old policy in terms of epochs")
	parser.add_argument('--use_ppo', type=str2bool, default=cfg.use_ppo, help="if use_ppo=False, and use_lm is True, then it is just a lm hence a baseline")
	parser.add_argument('--use_lm', type=str2bool, default=cfg.use_lm)  # whether if to use lm loss in addition to PPO
	parser.add_argument('--is_policy_optimization', type=str2bool, default=cfg.is_policy_optimization, help="if None then it is set to (self.cfg.use_ppo and not self.cfg.use_lm)")
	parser.add_argument('--use_true_curr_aspn', type=str2bool, default=cfg.use_true_curr_aspn, help="use true current aspn instead of predicted current aspn. Only could be true when is_policy_optimization")
	parser.add_argument('--freeze_encoder', type=str2bool, default=cfg.freeze_encoder, help="freeze encoder during training")
	parser.add_argument('--lm_head_model_action_value', type=str2bool, default=cfg.lm_head_model_action_value)
	parser.add_argument('--val_loss_coeff', type=float, default=cfg.val_loss_coeff)
	parser.add_argument('--lm_head_init', type=str2bool, default=cfg.lm_head_init)
	parser.add_argument('--alternate_epoch', type=str2bool, default=cfg.alternate_epoch, help="alternate lm and rl per epoch or per step")
	parser.add_argument('--alternate_step_k', type=str2number, default=cfg.alternate_step_k, help="how often to alternate between LM and PPO within a epoch")
	parser.add_argument('--reward', type=str, default=cfg.reward, help="which reward function to use", choices=["token_error", "sentence_error", "token_sim", "token_prob", "token_contextual_sim", "token_confidence", "zeros"])
	parser.add_argument('--correct_reward', type=float, default=cfg.correct_reward, help="basically specifies max reward achievable")
	parser.add_argument('--token_prob_temperature', type=float, default=cfg.token_prob_temperature, help="temperature for token probability")
	parser.add_argument('--token_prob_scale', type=float, default=cfg.token_prob_scale, help="scale for token probability")
	parser.add_argument('--token_embedding_path', type=str, default=cfg.token_embedding_path, help="path to token embedding")
	parser.add_argument('--lm_scale', type=str, default=cfg.lm_scale, help="which lm CE loss scaling to use", choices=["none", "token_error", "sentence_error"])
	parser.add_argument('--special_token_error_scale', type=float, default=cfg.special_token_error_scale, help="relative importance of special tokens")
	parser.add_argument('--penalize_da_tokens', type=str2bool, default=cfg.penalize_da_tokens)
	parser.add_argument('--terminal_reward_scale', type=float, default=cfg.terminal_reward_scale)
	parser.add_argument('--terminal_reward_fn', type=str, default=cfg.terminal_reward_fn, choices=["all", "sp_f1"])
	parser.add_argument('--use_bert_score', type=str2bool, default=cfg.use_bert_score)
	parser.add_argument('--add_kl_divergence', type=str2bool, default=cfg.add_kl_divergence)
	parser.add_argument('--precompute_kl_logits', type=str2bool, default=cfg.precompute_kl_logits)
	parser.add_argument('--precompute_token_prob_logits', type=str2bool, default=cfg.precompute_token_prob_logits)
	parser.add_argument('--kl_ckpt', type=str, default=cfg.kl_ckpt)
	parser.add_argument('--kl_loss_coeff', type=float, default=cfg.kl_loss_coeff)
	parser.add_argument('--add_terminal_reward', type=str2bool, default=cfg.add_terminal_reward)
	parser.add_argument('--sample_action', type=str, default=cfg.sample_action, choices=['sample', 'greedy'])
	parser.add_argument('--num_per_sample', type=int, default=cfg.num_per_sample, help="number of samples per action")
	parser.add_argument('--sample_top_p', type=float, default=cfg.sample_top_p)
	parser.add_argument('--sample_temperature', type=float, default=cfg.sample_temperature)
	parser.add_argument('--sample_temperature_decay', type=float, default=cfg.sample_temperature_decay)
	parser.add_argument('--rl_gamma', type=float, default=cfg.rl_gamma)
	parser.add_argument('--exp_group_name', type=str, default=cfg.exp_group_name, help="for wandb")
	return parser


def add_pgtod_arg_configs(cfg:_PGTOD_CONFIG, parser: argparse.ArgumentParser):
	parser = add_ppotod_arg_configs(cfg, parser)
	parser.add_argument('--rl_algo', type=str, default=cfg.rl_algo, choices=['off-policy', 'ssemi-on-policy'])
	return parser


def __init_kl_model(cfg, dataset):
	class _KL_PPOTOD_CONFIG(_PPOTOD_CONFIG):
		pass

	checkpoint_root_path = os.path.dirname(cfg.kl_ckpt)
	run_config_path = os.path.join(checkpoint_root_path, 'run_config.json')
	kl_cfg = None
	with open(run_config_path, 'r') as f:
		kl_cfg = json.load(f)
	kl_cfg = update_config(kl_cfg, _KL_PPOTOD_CONFIG)
	kl_cfg.ckpt = cfg.kl_ckpt
	kl_model = PPOTODModel(kl_cfg, dataset.vocab_size, dataset.tokenizer)
	return kl_model


def train_ppotod():
	cfg = _PPOTOD_CONFIG
	parser = argparse.ArgumentParser()
	parser = add_ppotod_arg_configs(cfg, parser)
	args = get_arg_configs(cfg, parser)
	cfg = update_config(args, cfg)

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
	return cfg, trainer


def train_pgtod():
	cfg = _PGTOD_CONFIG
	parser = argparse.ArgumentParser()
	parser = add_pgtod_arg_configs(cfg, parser)
	args = get_arg_configs(cfg, parser)
	cfg = update_config(args, cfg)

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
	trainer = PGLMTrainer(dataset, cfg, model, group_name, run_name=None, test_dataset=test_dataset, kl_model=kl_model)
	return cfg, trainer


def train_realrl_tod():
	cfg = _PPOTOD_CONFIG
	parser = argparse.ArgumentParser()
	parser = add_ppotod_arg_configs(cfg, parser)
	args = get_arg_configs(cfg, parser)
	cfg = update_config(args, cfg)

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
	trainer = PPORealRLTrainer(dataset, cfg, model, group_name, run_name=None, test_dataset=test_dataset, kl_model=kl_model)
	return cfg, trainer


if __name__ == "__main__":
	# train with PPO
	cfg, trainer = train_ppotod()
	# train with PG
	# cfg, trainer = train_pgtod()
	# train with real RL only
	# cfg, trainer = train_realrl_tod()

	if cfg.mode == 'train':
		# save run_config
		run_config_path = os.path.join(cfg.model_dir, 'run_config.json')
		save_cfg(cfg, run_config_path)
		
		trainer.train()
	elif cfg.mode == 'predict':
		trainer.predict()
