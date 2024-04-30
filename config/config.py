r"""
    Config for paths, joint set, and normalizing scales.
"""

import torch
# datasets (directory names) in AMASS
# e.g., for ACCAD, the path should be `paths.raw_amass_dir/ACCAD/ACCAD/s001/*.npz`
amass_data = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
              'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
              'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust_67']

amass_test_data = ["DanceDB"]

class paths:
    raw_amass_dir = 'data/AMASS_raw'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = 'data/processed_data/AMASS_syn'         # output path for the synthetic AMASS dataset

    raw_dipimu_dir = 'data/DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = 'data/processed_data/DIP_IMU'      # output path for the preprocessed DIP-IMU dataset
    
    raw_uwbimu_dir = '/local/shared_data/data_UWB_IMU/SIGGRAPH_dataset'   # raw UWB IMU dataset path
    uwbimu_dir = 'data/processed_data/UWB_IMU/SIGGRAPH_dataset'      # output path for the preprocessed UWB IMU dataset

    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP authors.
    raw_totalcapture_dip_dir = 'data/TotalCapture_Real_60FPS'  # contain ground-truth SMPL pose (*.pkl)
    raw_totalcapture_official_dir = 'data/TotalCapture'    # contain official gt (S1/acting1/gt_skel_gbl_pos.txt)
    totalcapture_dir = 'data/processed_data/TotalCapture'          # output path for the preprocessed TotalCapture dataset

    result_dir = 'data/result'                      # output directory for the evaluation results

    smpl_file = 'data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'     # official SMPL model path
    physics_model_file = 'data/urdfmodels/physics.urdf'      # physics body model path
    plane_file = 'data/urdfmodels/plane.urdf'                # (for debug) path to plane.urdf    Please put plane.obj next to it.
    weights_file = 'data/weights.pt'                # network weight file
    physics_parameter_file = 'config/physics_parameters.json'   # physics hyperparameters

class joint_set:
    root = [0]
    leaf = [4, 5, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]
    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


class IMU_placement:
    vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    ji_mask = torch.tensor([18, 19, 4, 5, 15, 0]) 
      
vel_scale = 3
uwb_collision_thr = 0.3 
'''
SMPL mapping 
'''
SMPL_JOINTS = [
    "root",#        
    "lhip",
    "rhip",
    "lowerback",
    "lknee",
    "rknee",
    "upperback",
    "lankle",
    "rankle",
    "chest",
    "ltoe",
    "rtoe",
    "lowerneck",
    "lclavicle",
    "rclavicle",
    "upperneck",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
    "lhand",
    "rhand",
]
SMPL_JOINT_IDX_MAPPING = {x: i for i, x in enumerate(SMPL_JOINTS)}
SMPL_IDX_JOINT_MAPPING = {i: x for i, x in enumerate(SMPL_JOINTS)}

''' 
Definition of Link/Joint (In our character definition, one joint can only have one link)
'''
root = -1
lhip = 0
lknee = 1
lankle = 2
rhip = 3
rknee = 4
rankle = 5
lowerback = 6
upperback = 7
chest = 8
lowerneck = 9
upperneck = 10
lclavicle = 11
lshoulder = 12
lelbow = 13
lwrist = 14
rclavicle = 15
rshoulder = 16
relbow = 17
rwrist = 18

import collections
bvh_map = collections.OrderedDict()

bvh_map[root] = "root"
bvh_map[lhip] = "lhip"
bvh_map[lknee] = "lknee"
bvh_map[lankle] = "lankle"
bvh_map[rhip] = "rhip"
bvh_map[rknee] = "rknee"
bvh_map[rankle] = "rankle"
bvh_map[lowerback] = "lowerback"
bvh_map[upperback] = "upperback"
bvh_map[chest] = "chest"
bvh_map[lowerneck] = "lowerneck"
bvh_map[upperneck] = "upperneck"
bvh_map[lclavicle] = "lclavicle"
bvh_map[lshoulder] = "lshoulder"
bvh_map[lelbow] = "lelbow"
bvh_map[lwrist] = "lwrist"
bvh_map[rclavicle] = "rclavicle"
bvh_map[rshoulder] = "rshoulder"
bvh_map[relbow] = "relbow"
bvh_map[rwrist] = "rwrist"


IMU_NUM = 6
INPUT_DATA_SIZE = {"vacc":IMU_NUM * 3,"vrot": IMU_NUM * 9, "vuwb" :IMU_NUM * 6,"acc_sum":IMU_NUM * 3,"f_vuwb": int(IMU_NUM * (IMU_NUM - 1) / 2)}
