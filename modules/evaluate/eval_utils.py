'''
Adapted from PIP evaluate script https://github.com/Xinyu-Yi/PIP/blob/main/evaluate.py
'''

import torch
import tqdm
from config.config import *
from modules.utils import *
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import articulate as art
from articulate.utils.rbdl import *
try:
    from fairmotion.ops import conversions, quaternion
except:
    print("Did not find package fairmotion.ops")
from modules.dataset.data_utils import D_Batch
device = "cpu"
def print_title(s):
    print('============ %s ============' % s)

def load_ckpt(ckpt_path):
    weight = torch.load(ckpt_path,map_location=torch.device('cpu'))
    if "net" in weight:
        return weight["net"]
    else:
        return weight

class _PoseEvaluator:
    names = ['SIP Error (deg)', 'Angle Error (deg)', 'Joint Error (cm)', 'Vertex Error (cm)', 'Jitter Error (km/s^3)']
    def __init__(self):
        non_root_active_idx = list(range(-1,19))
        non_root_active_idx.remove(14);non_root_active_idx.remove(18) #remove l/r wrist
        self.selected_mask = [SMPL_JOINT_IDX_MAPPING[bvh_map[idx]] for idx in non_root_active_idx]
        self.selected_mask_tip = [SMPL_JOINT_IDX_MAPPING[bvh_map[idx]] for idx in non_root_active_idx]
        self.selected_mask.remove(0);self.selected_mask.remove(7);self.selected_mask.remove(8)
        self._base_motion_loss_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]), device="cpu")
        self.ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
    def __call__(self, pose_p, pose_t, tran_p, tran_t):
        pass
        
class PoseEvaluator(_PoseEvaluator):
    def __init__(self):
        super().__init__()

    def __call__(self, pose_p, pose_t, tran_p, tran_t,reduction="mean"):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, self.ignored_joint_mask] = torch.eye(3, device=pose_p.device)
        pose_t[:, self.ignored_joint_mask] = torch.eye(3, device=pose_t.device)
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 1000])


class FullPoseEvaluator:
    names = ['Absolute Jitter Error (km/s^3)']

    def __init__(self):
        self._base_motion_loss_fn = art.FullMotionEvaluator(paths.smpl_file, device=device)

    def __call__(self, pose_p, pose_t, tran_p, tran_t):
        # errs = self._base_motion_loss_fn(pose_p=pose_p[:-1], pose_t=pose_t[:-1], tran_p=tran_p[:-1], tran_t=tran_t[:-1])  # bad data -1
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([errs[4] / 1000])


def evaluate_zmp_distance(poses, trans, fps=60, foot_radius=0.1):
    qs = smpl_to_rbdl(poses, trans)
    qdots = np.empty_like(qs)
    qdots[1:, :3] = (qs[1:, :3] - qs[:-1, :3]) * fps
    qdots[1:, 3:] = art.math.angle_difference(qs[1:, 3:], qs[:-1, 3:]) * fps
    qdots[0] = qdots[1]
    qddots = (qdots[1:] - qdots[:-1]) * fps
    qddots = np.concatenate((qddots[:1], qddots))
    rbdl_model = RBDLModel(paths.physics_model_file)

    floor_height = []
    for q in qs[2:30]:
        lp = rbdl_model.calc_body_position(q, Body.LFOOT)
        rp = rbdl_model.calc_body_position(q, Body.RFOOT)
        floor_height.append(lp[1])
        floor_height.append(rp[1])
    floor_height = torch.tensor(floor_height).mean() + 0.01

    dists = []
    for q, qdot, qddot in zip(qs, qdots, qddots):
        lp = rbdl_model.calc_body_position(q, Body.LFOOT)
        rp = rbdl_model.calc_body_position(q, Body.RFOOT)
        if lp[1] > floor_height and rp[1] > floor_height:
            continue

        zmp = rbdl_model.calc_zero_moment_point(q, qdot, qddot)
        ap = (zmp - lp)[[0, 2]]
        ab = (rp - lp)[[0, 2]]
        bp = (zmp - rp)[[0, 2]]
        if lp[1] <= floor_height and rp[1] <= floor_height:
            # point to line segment distance
            r = (ap * ab).sum() / (ab * ab).sum()
            if r < 0:
                d = np.linalg.norm(ap)
            elif r > 1:
                d = np.linalg.norm(bp)
            else:
                d = np.sqrt((ap * ap).sum() - r * r * (ab * ab).sum())
        else:
            # point to point distance
            d = np.linalg.norm(ap if lp[1] <= floor_height else bp)
        dists.append(max(d - foot_radius, 0))

    return sum(dists) / len(dists)

def run_position_estimator(net, data_dir, sequence_ids=None,normalize_uwb=False,flatten_uwb=False,**kwargs):
    r"""
    Run `net` using the imu data loaded from `data_dir`.
    Save the estimated [Pose[num_frames, 24, 3, 3], Tran[num_frames, 3]] for each of `sequence_ids`.
    """
    print('Loading imu data from "%s"' % data_dir)
    v
    #accs, rots, poses, *res  = torch.load(os.path.join(data_dir, 'test.pt')).values()
    #test_db = D_Batch(torch.load(os.path.join(data_dir, 'test.pt')))
    test_db = D_Batch(torch.load(os.path.join(data_dir, 'test.pt')))
    init_poses = [art.math.axis_angle_to_rotation_matrix(_[0]) for _ in test_db.pose]
    data_name = os.path.basename(data_dir)
    output_dir = os.path.join(paths.result_dir, data_name, net.name)
    os.makedirs(output_dir, exist_ok=True)
    
    if sequence_ids is None:
        sequence_ids = list(range(len(test_db.acc)))

    print('Saving the results at "%s"' % output_dir)
    for i in tqdm.tqdm(sequence_ids):
        if "vuwb" in net.imu_m:
            vuwb = test_db.vuwb[i] if not normalize_uwb else test_db.vuwb[i]/test_db.vuwb[i][0,4,5]
            if flatten_uwb and vuwb.size(1) == 6 and vuwb.size(2) == 6:
                index = torch.triu_indices(6, 6, 1)#hard code
                vuwb = vuwb[:,index[0],index[1]]
            torch.save(net(test_db.acc[i], test_db.ori[i], init_poses[i], glb_uwb = vuwb, offset=test_db.offset), os.path.join(output_dir, '%d.pt' % i))
        else:
            torch.save(net(test_db.acc[i], test_db.ori[i], init_poses[i]), os.path.join(output_dir, '%d.pt' % i))
            
def run_pipeline(net, data_dir, sequence_ids=None,normalize_uwb=False,flatten_uwb=False,**kwargs):
    r"""
    Run `net` using the imu data loaded from `data_dir`.
    Save the estimated [Pose[num_frames, 24, 3, 3], Tran[num_frames, 3]] for each of `sequence_ids`.
    """
    print('Loading imu data from "%s"' % data_dir)
    test_db = D_Batch(torch.load(os.path.join(data_dir, 'test.pt')))
    init_poses = [art.math.axis_angle_to_rotation_matrix(_[0]) for _ in test_db.pose]
    data_name = os.path.basename(data_dir)
    output_dir = os.path.join(paths.result_dir, data_name, net.name)
    os.makedirs(output_dir, exist_ok=True)
    
    if sequence_ids is None:
        sequence_ids = list(range(len(test_db.acc)))
    
    net.train(False)
    print('Saving the results at "%s"' % output_dir)
    for i in tqdm.tqdm(sequence_ids):
        if "vuwb" in net.imu_m:
            vuwb = test_db.vuwb[i] if not normalize_uwb else test_db.vuwb[i]/test_db.vuwb[i][0,4,5]
            if flatten_uwb and vuwb.size(1) == 6 and vuwb.size(2) == 6:
                index = torch.triu_indices(6, 6, 1)#hard code
                vuwb = vuwb[:,index[0],index[1]]
            torch.save(net.predict(test_db.acc[i], test_db.ori[i], init_poses[i], glb_uwb = vuwb, offset=test_db.offset), os.path.join(output_dir, '%d.pt' % i))
        else:
            torch.save(net.predict(test_db.acc[i], test_db.ori[i], init_poses[i]), os.path.join(output_dir, '%d.pt' % i))
            

