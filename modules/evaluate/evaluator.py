import argparse
import os
from typing import Any
from config.config import *
import torch
import articulate as art
import tqdm
try:
	from fairmotion.ops import conversions, quaternion
except:
	print("Did not find package fairmotion.ops")
from modules.evaluate.eval_utils import *
from modules.model import MODELS
from collections import OrderedDict
from prettytable import PrettyTable

def evaluate_model(net, data_dir, sequence_ids=None, flush_cache=False, pose_evaluator=PoseEvaluator(),
			 evaluate_pose=True, evaluate_tran=False, evaluate_zmp=False, plt_tran=False,**kwargs):
	r"""
	Evaluate poses and translations of `net` on all sequences in `sequence_ids` from `data_dir`.
	`net` should implement `net.name` and `net.predict(glb_acc, glb_rot)`.
	"""
	data_name = os.path.basename(data_dir)
	result_dir = os.path.join(paths.result_dir, data_name, net.name)
	print_title('Evaluating "%s" on "%s"' % (net.name, data_name))

	_, _, pose_t_all, tran_t_all, *res = torch.load(os.path.join(data_dir, 'test.pt')).values() #[45,Size([4113, 24, 3])],[45,torch.Size([4113, 3])]

	if isinstance(res[-1][0],str):
		file_path = res[-1]
	else:
		file_path = None
  
	if sequence_ids is None:
		sequence_ids = list(range(len(pose_t_all))) #[0~44]
	if flush_cache and os.path.exists(result_dir):
		shutil.rmtree(result_dir)

	missing_ids = [i for i in sequence_ids if not os.path.exists(os.path.join(result_dir, '%d.pt' % i))]
	cached_ids = [i for i in sequence_ids if os.path.exists(os.path.join(result_dir, '%d.pt' % i))]
	print('Cached ids: %s\nMissing ids: %s' % (cached_ids, missing_ids))
	if len(missing_ids) > 0:
		run_pipeline(net, data_dir, missing_ids,**kwargs)
		#run_pipeline_w_gt(net, data_dir, missing_ids,**kwargs)

	output_eval_res = OrderedDict()
	pose_errors = []
	tran_errors = {window_size: [] for window_size in list(range(1, 8))}
	zmp_errors = []
	for i in tqdm.tqdm(sequence_ids):
		result = torch.load(os.path.join(result_dir, '%d.pt' % i))

		pose_p, tran_p = result[0], result[1]   #torch.Size([4113, 24, 3, 3]) torch.Size([4113, 3])
		pose_t, tran_t = pose_t_all[i], tran_t_all[i]
		if evaluate_pose:
			pose_t = art.math.axis_angle_to_rotation_matrix(pose_t).view_as(pose_p) #torch.Size([4113, 24, 3, 3])
			pose_errors.append(pose_evaluator(pose_p, pose_t, tran_p, tran_t))
		if evaluate_tran:
			# compute gt move distance at every frame
			move_distance_t = torch.zeros(tran_t.shape[0])
			v = (tran_t[1:] - tran_t[:-1]).norm(dim=1)
			for j in range(len(v)):
				move_distance_t[j + 1] = move_distance_t[j] + v[j]

			for window_size in tran_errors.keys():
				# find all pairs of start/end frames where gt moves `window_size` meters
				frame_pairs = []
				start, end = 0, 1
				while end < len(move_distance_t):
					if move_distance_t[end] - move_distance_t[start] < window_size:
						end += 1
					else:
						if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
							frame_pairs.append((start, end))
						start += 1

				# calculate mean distance error
				errs = []
				for start, end in frame_pairs:
					vel_p = tran_p[end] - tran_p[start] 
					vel_t = tran_t[end] - tran_t[start]
					errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
				if len(errs) > 0:
					tran_errors[window_size].append(sum(errs) / len(errs))

		if evaluate_zmp:
			zmp_errors.append(evaluate_zmp_distance(pose_p, tran_p))

	if evaluate_pose:
		pose_errors_mean = torch.stack(pose_errors).mean(dim=0)
		pose_errors_max = torch.stack(pose_errors).max(dim=0)
		pose_worse_idx = torch.mode(pose_errors_max.indices[:,0]).values.item()
		pose_errors_min = torch.stack(pose_errors).min(dim=0)
		pose_best_idx = torch.mode(pose_errors_min.indices[:,0]).values.item()
		
		for name, error in zip(pose_evaluator.names, pose_errors_mean):
			#print('%s: %.4f' % (name, error[0]))
			output_eval_res[name] = error[0]
   
		if file_path is None:
			output_eval_res['best'] = (os.path.join(result_dir, '%d.pt' % pose_best_idx), pose_errors[pose_best_idx][:,0])
			output_eval_res['worst'] = (os.path.join(result_dir, '%d.pt' % pose_worse_idx), pose_errors[pose_worse_idx][:,0])
		else:
			output_eval_res['best'] = (file_path[pose_best_idx], pose_errors[pose_best_idx][:,0])
			output_eval_res['worst'] = (file_path[pose_worse_idx], pose_errors[pose_worse_idx][:,0])
   
		print([(str(seq_id),e) for seq_id,e in enumerate(pose_errors)])
			
	if evaluate_zmp:
		print('ZMP Distance (m): %.4f' % (sum(zmp_errors) / len(zmp_errors)))
		output_eval_res["ZMP Distance"] = sum(zmp_errors) / len(zmp_errors)
		
	if evaluate_tran:
		if plt_tran:
			plt.plot([0] + [_ for _ in tran_errors.keys()], [0] + [torch.tensor(_).mean() for _ in tran_errors.values()], label=net.name)
			plt.legend(fontsize=15)
			plt.show()
		output_eval_res["trans_error"] = tran_errors
		output_eval_res["trans_error_2m"] = torch.tensor(tran_errors[2]).mean().item()
		output_eval_res["trans_error_5m"] = torch.tensor(tran_errors[5]).mean().item()	
		print('%s: %.4f' % ("trans error at 2 m", torch.tensor(tran_errors[2]).mean().item()))
		print('%s: %.4f' % ("trans error at 5 m", torch.tensor(tran_errors[5]).mean().item()))

	return output_eval_res
 

def print_eval_result(eval_dicts: list, filter_keys: set, title = ["   ", "DIP-IMU","TotalCapture"]):
	tab = PrettyTable(title)
	key_list = [key for key in eval_dicts[0].keys() if key in filter_keys]
	for key in key_list:
		if key != "trans_error":
			tab.add_row([key] + [f"{d[key]}" for d in eval_dicts])
	print(tab)
	return tab

def get_args():
	parser = argparse.ArgumentParser(description='Evaluation process')
	parser.add_argument('--network', type=str, default="UIP",
						help='network name for evaluating')
	parser.add_argument('--ckpt_path', type=str, default="",
						help='model weight for evaluating')
	parser.add_argument('--data_dir', type=str, default=0.0,
						help='test data directory for evaluation')
	parser.add_argument('--seq_id', nargs='+',
						help="Specify the sequence id for test dataset")
	parser.add_argument('--render', action='store_true',
						help='whether to use pybullet to render test results')
	parser.add_argument('--flush_cache', action='store_true',
						help='whether to flush cached results')
	parser.add_argument('--eval_trans', action='store_true',
						help='whether to evaluate global translation')
	parser.add_argument('--eval_save_dir', type=str, default="",
						help='dir to save evaluation results')
	parser.add_argument('--normalize_uwb', action='store_true',
						help='whether to normalize uwb value by head-pelvis distance')
	parser.add_argument('--flatten_uwb', action='store_true',
						help='whether to flatten uwb into vector of size 15')
	parser.add_argument('--add_guassian_noise', action='store_true',
						help='whether to add gaussian noise')
	parser.add_argument('--no_rnn_init', action='store_true',
						help='whether to remove rnn initial')
	parser.add_argument('--model_args_file', type=str, default="",
						help='Config file for the model .ini')
	parser.add_argument('--exp_name', type=str, default="",
						help='name add after the network name')
	args = parser.parse_known_args()
	return args[0], parser

def main():
	args, parser = get_args()
	sequence_ids = [int(i) for i in args.seq_id] if args.seq_id else None
	print(f"data dir  : {args.data_dir}")
	if args.network in MODELS:
		model_cls = MODELS[args.network]
		print(model_cls)
		if args.model_args_file:
			net = model_cls.load_model_with_args(args.model_args_file)
		else:
			model_cls.add_args(parser)
			model_args = parser.parse_args()
			net = model_cls(model_args)
		net.load_state_dict(load_ckpt(args.ckpt_path))
		net.name = net.name + args.exp_name
		print("network name:",net.name)
		eval_res = evaluate_model(net,data_dir=args.data_dir,sequence_ids=sequence_ids,flush_cache=args.flush_cache,evaluate_tran=args.eval_trans,pose_evaluator=PoseEvaluator(),
                          			normalize_uwb=args.normalize_uwb,flatten_uwb=args.flatten_uwb)
		y_axis_up = True
		gt_dir = args.data_dir
	else:
		raise KeyError("Invalid network name")

	if args.eval_save_dir:
		os.makedirs(args.eval_save_dir,exist_ok=True)
     
		tab = print_eval_result([eval_res],filter_keys=set(eval_res),title=["Dataset",args.data_dir])

		with open(os.path.join(args.eval_save_dir,f"[Eval_tab]{args.network}.csv"), 'w', newline='') as fid:
			fid.write(tab.get_csv_string())

		if "trans_error" in eval_res:
			plt.plot([0] + [_ for _ in eval_res["trans_error"].keys()], [0] + [torch.tensor(_).mean() for _ in eval_res["trans_error"].values()], label=args.network)
			plt.legend(fontsize=15)
			plt.savefig(os.path.join(args.eval_save_dir,f"Translation_error_{args.network}.png"))
			plt.close("all")
		
if __name__ == "__main__":
	main()
