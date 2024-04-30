# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import pickle as pkl
import os
import torch
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
import argparse

def load_uwb_imu_dataset(path,seq_id,rgb,seq_end=-1,stride=2,beta=None):
    data = torch.load(path)

    # Get the data.
    poses = data["pose"][seq_id].view(-1,72)
    tran = data["tran"][seq_id].view(-1,3)
    oris = data["ori"][seq_id].numpy().reshape(-1,6,3,3)
    accs = data["acc"][seq_id].numpy().reshape(-1,6,3)
    if "vuwb" in data:
        uwb = data["vuwb"][seq_id].numpy().reshape(-1,6,6)
        uwb = uwb[:seq_end:2]
    else:
        uwb = None

    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"

    # Downsample to 30 Hz.
    poses = poses[:seq_end:stride]

    oris = oris[:seq_end:stride]
    tran = tran[:seq_end:stride]
    accs = accs[:seq_end:stride]
    tran = torch.zeros_like(tran)
    # DIP has no shape information, assume the mean shape.
    if beta is None:
        betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    else:
        betas = beta.repeat(poses.shape[0], 1).float().to(C.device)
        
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)
    poses[:,20*3:22*3] = 0 #hard code hand pose

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=poses[:, 3:].to(C.device),
        poses_root=poses[:, :3].to(C.device),
        betas=betas,
        trans=tran.to(C.device),
    )

    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    print(betas)
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3],trans=tran,betas=betas)
    smpl_seq.mesh_seq.color = rgb + (1.0,)
    
    return smpl_seq,joints,oris,accs,uwb

def visualize_smpl_models(path,rgb=(0.62, 0.62, 0.62),seq_end=-1,stride = 2,vis_leaf_joint_position=False):
    data = torch.load(path)
    uwb_imu_rot = np.array([[1, 0, 0], [0, 0, 1.0], [0, -1, 0]])
    # Get the data.
    poses = matrix_to_axis_angle(data[0]).view(-1,72)
    tran = data[1].view(-1,3)
    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"

    # Downsample to 30 Hz.
    poses = poses[:seq_end:stride]
    tran = tran[:seq_end:stride]
    #just for vis fig
    tran = torch.zeros_like(tran)
    # DIP has no shape information, assume the mean shape.
    betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=poses[:, 3:].to(C.device),
        poses_root=poses[:, :3].to(C.device),
        betas=betas,
        trans=tran.to(C.device)
    )

    #rbs_v = RigidBodies(joints[:, joint_idxs].cpu().numpy(), v_oris,color=(0,0,0,1))
    
    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3],trans=tran)
    smpl_seq.mesh_seq.color = rgb + (1.0,)
    
    return smpl_seq,joints

def visualize_leaf_joint_position(path,joints,seq_end=-1,stride = 2):
    data = torch.load(path)
    
    # Get the data.
    tran = data[1].view(-1,3)
    root_ori = data[0][:,0]
    leaf_joint_position = data[2].view(-1,5,3)
    
    # Downsample to 30 Hz.
    #leaf_joint_position = leaf_joint_position[:seq_end:stride,[2,3,4,0,1]]
    leaf_joint_position = leaf_joint_position[:seq_end:stride]
    root_ori = root_ori[:seq_end:stride]
    tran = tran[:seq_end:stride]
    f,n,_ = leaf_joint_position.size()
    leaf_joint_position = leaf_joint_position @ root_ori.permute(0,2,1)
  
    root_position = np.tile(joints[:,0].cpu().numpy(),(1,5)).reshape(f,n,3)
    arr_head = Arrows(origins=root_position[:,[2]], tips=root_position[:,[2]] - leaf_joint_position[:,[2]].cpu().numpy(),color=(0,0,0.5,1))#b
    arr_leg = Arrows(origins=root_position[:,[0,1]], tips=root_position[:,[0,1]] - leaf_joint_position[:,[0,1]].cpu().numpy(),color=(0,0.5,0,1))#g
    arr_upper = Arrows(origins=root_position[:,[3,4]], tips=root_position[:,[3,4]] - leaf_joint_position[:,[3,4]].cpu().numpy(),color=(0.5,0,0,1))#r
    #rb = RigidBodies(leaf_joint_position.cpu().numpy(), np.tile(np.eye(3),(f,n,1,1)).reshape(f,n,3,3),color=(0,0,0,1))
    
    return [arr_head,arr_leg,arr_upper]

def get_args():
	parser = argparse.ArgumentParser(description='Evaluation process')
	parser.add_argument('--seq_res_path',type=str,
						help="Specify the sequence id path for test dataset")
	parser.add_argument('--seq_id', type=int, default=1,
						help='result sequence id to run')
	args = parser.parse_args()
	return args, parser

if __name__ == "__main__":
    args,_ = get_args()
    show_lj_pos = False
    seq_id = args.seq_id
    paths = [os.path.join(args.seq_res_path,str(seq_id)+".pt")]
    stride = 1
    colors = [(0.348,0.395,0.628)]

    smpl_seqs = [];rb_seq = []
    for p,c in zip(paths,colors):
        smpl_s,joint = visualize_smpl_models(p,rgb=c,seq_end=12000,stride=stride)
        smpl_seqs.append(smpl_s)
        if show_lj_pos:
            rb_seq.extend(visualize_leaf_joint_position(p,joint,seq_end=12000,stride=stride))
            
        print("frame_number",joint.size(0))
    
    # Add everything to the scene and display at 30 fps.
    v = Viewer()
    v.playback_fps = 30.0

    v.scene.add(*smpl_seqs)
    if rb_seq:
        v.scene.add(*rb_seq)
    v.run()