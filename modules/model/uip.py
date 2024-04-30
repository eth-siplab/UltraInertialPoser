'''
# --------------------------------------------
# Ultral Inertial Poser Network
# --------------------------------------------
# Ultra Inertial Poser: Scalable Motion Capture and Tracking from Sparse Inertial Sensors and Ultra-Wideband Ranging (SIGGRAPH 2024)
# https://github.com/eth-siplab/UltraInertialPoser
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''
import configargparse
import json
import os
from torch.nn.utils.rnn import *
import torch.nn as nn
import articulate as art
from articulate.utils.torch import *
from config.config import *
from modules.utils import *
from modules.dynamics import PhysicsOptimizer
from modules.dataset.data_utils import Batch, D_Batch
import torch
from modules.model.gcn_encoder import Graph_JP_estimator 
from modules.model.model_base import BaseModel   
NO_INIT_POSE = False

class RNN_JP_estimator(nn.Module):
    name = "RNN_JP"
    imu_m = ["vacc","vrot"]
    model_output = ["imu_node_position"]

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("RNN_JP_estimator Config")
        group.add_argument('--with_rnn_init', action='store_true', help='Whether to use RNN init from PIP')
        group.add_argument('--n_hidden', type=int, default=256, help='Hidden layer width')

    @staticmethod
    def get_config(args):
        config_str = f"{args.network}-input{RNN_JP_estimator.imu_m},-output{RNN_JP_estimator.model_output}"
        return config_str

    def __init__(self, args):
        super().__init__()
        self.input_size = sum([INPUT_DATA_SIZE[k] for k in self.imu_m])
        self.with_rnn_init = args.with_rnn_init
        if self.with_rnn_init:
            self.rnn_leaf_joint_mapper = RNNWithInit(input_size=self.input_size,
                                    output_size=joint_set.n_leaf * 3,
                                    hidden_size=args.n_hidden,
                                    num_rnn_layer=2,
                                    dropout=0.4)
        else:
            self.rnn_leaf_joint_mapper = RNN(input_size=self.input_size,
                                    output_size=joint_set.n_leaf * 3,
                                    hidden_size=args.n_hidden,
                                    num_rnn_layer=2,
                                    dropout=0.4)
            
    def forward(self,x):
        x, lj_init, jvel_init = list(zip(*x))
        x_imu = [_[:,:-INPUT_DATA_SIZE["vuwb"]] for _ in list(x)]
        leaf_joint = self.rnn_leaf_joint_mapper(list(zip(x_imu, lj_init))) if self.with_rnn_init \
                else self.rnn_leaf_joint_mapper(list(x_imu))
                
        return leaf_joint    


class UIP(BaseModel):
    name = "Ultra Inertial Poser"
    imu_m = ["vacc","vrot","vuwb"]
    model_output = ["imu_node_position_rnn", 
                    "imu_node_position_gnn",
                    "global_6d_pose", 
                    "joint_velocity", 
                    "contact"]
    @staticmethod
    def add_args(parser):
        RNN_JP_estimator.add_args(parser)
        Graph_JP_estimator.add_args(parser)
        group = parser.add_argument_group("UIP Config")
        group.add_argument('--add_gaussian', action='store_true', help='Whether to add gaussian to leaf joints estimation')
        group.add_argument('--acc_min_fuse', type=float, default=1.0, help='The lowerb threshold to fuse joint position of uwb and imu measurement')
        group.add_argument('--acc_max_fuse', type=float, default=3.0, help='The upperb threshold to fuse joint position of uwb and imu measurement')
        group.add_argument('--fuse_window_size', type=int, default=60, help='window size to fuse ')
        
    @staticmethod
    def get_config(args):
        config_str = f"{args.network}-input{UIP.imu_m},-output{UIP.model_output}"
        config_str += f"-input_sensor{UIP.imu_m}-hidden_dim{args.hidden_dim}"
        return config_str
    
    def _get_input_size(self):
        return sum([INPUT_DATA_SIZE[k] for k in self.imu_m])
    
    def __init__(self,args) -> None:
        super().__init__(args)
        self.rnn_jp_mapper = RNN_JP_estimator(args)
        self.gnn_jp_mapper = Graph_JP_estimator(args)
        self.acc_lb = args.acc_min_fuse
        self.acc_ub = args.acc_max_fuse
        self.with_rnn_init = args.with_rnn_init
        self.input_size = self._get_input_size()
        self.imu_num = args.node_number
        self.max_pool = nn.MaxPool1d(kernel_size=args.fuse_window_size,stride=args.fuse_window_size)
        if self.with_rnn_init:
            self.rnn4 = RNNWithInit(input_size=self.input_size + joint_set.n_leaf * 3,
                                    output_size=24 * 3,
                                    hidden_size=args.n_hidden,
                                    num_rnn_layer=2,
                                    dropout=0.4)
        else:
            self.rnn4 = RNN(input_size=self.input_size + joint_set.n_leaf * 3,
                                    output_size=24 * 3,
                                    hidden_size=args.n_hidden,
                                    num_rnn_layer=2,
                                    dropout=0.4)
                
        self.rnn3 = RNN(input_size=self.input_size + joint_set.n_leaf * 3,
                        output_size=joint_set.n_reduced * 6,
                        hidden_size=args.n_hidden,
                        num_rnn_layer=2,
                        dropout=0.4)

        self.rnn5 = RNN(input_size=self.input_size + joint_set.n_leaf * 3,
                        output_size=2,
                        hidden_size=64,
                        num_rnn_layer=2,
                        dropout=0.4)
        
        body_model = art.ParametricModel(paths.smpl_file)
        self.inverse_kinematics_R = body_model.inverse_kinematics_R
        self.forward_kinematics = body_model.forward_kinematics
        self.dynamics_optimizer = PhysicsOptimizer(debug=False)
        self.rnn_states = [None for _ in range(5)]
        self.add_guassian = args.add_gaussian
        
        #self.save_config(args, os.path.join(args.log_dir,"model_args.txt"))
    
    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def fuse_joint_position_estimation(self,jp1,jp2,beta):
        """_summary_

        Args:
            jp1 (_type_): bz,f,15
            jp2 (_type_): bz,f,15
            beta (_type_): bz,f,5
        """
        bz,f,n = beta.size()
        padding = max(0, self.max_pool.kernel_size - f % self.max_pool.stride)
        if padding > 0:
            pad_tensor = torch.zeros(bz, padding, 5, dtype=beta.dtype, device=beta.device)
            beta_padded = torch.cat([beta, pad_tensor], dim=1)
            
        pooled_acc_mag = self.max_pool(beta_padded.permute(0,2,1)) #b,5,4
        #beta_unfold = beta_padded.unfold(1, self.max_pool.kernel_size,self.max_pool.stride).permute(0,2,1,3)#b,5,4,ks

        beta_pooled = torch.repeat_interleave(pooled_acc_mag,repeats=self.max_pool.kernel_size,dim=-1)[:,:,:f].permute(0,2,1)
        assert beta_pooled.size() == beta.size() 
        
        deno = self.acc_ub - self.acc_lb
        weight_jp1 = torch.min(torch.max((beta_pooled - self.acc_lb) / deno, torch.zeros_like(beta_pooled)), torch.ones_like(beta_pooled)).unsqueeze(-1)
        weight_jp2 = torch.min(torch.max((beta_pooled - self.acc_ub) / -deno, torch.zeros_like(beta_pooled)), torch.ones_like(beta_pooled)).unsqueeze(-1)
        fuse_jp = jp1.view(bz,f,n,3) * weight_jp1 + jp2.view(bz,f,n,3) * weight_jp2

        return [fuse_jp.view(bz,f,-1)[i] for i in range(bz)]
        
    def forward(self, x):
        """
        Forward.

        :param x: A list in length [batch_size] which contains 3-tuple
                  (tensor [num_frames, 72], tensor [15], tensor [72]).
        :return: [torch.Size([num_frames, 15]), 
                  torch.Size([num_frames, 69]), 
                  torch.Size([num_frames, 90]), 
                  torch.Size([num_frames, 72]), 
                  torch.Size([num_frames, 2])]
        """
        # x_imu = [_[:,:-INPUT_DATA_SIZE["vuwb"]] for _ in list(x)]
        # x_uwb = [_[:,-INPUT_DATA_SIZE["vuwb"]:] for _ in list(x)]#hard code
        
        leaf_joint_rnn = self.rnn_jp_mapper(x)
        leaf_joint_gnn = self.gnn_jp_mapper(x)
        
        x, lj_init, jvel_init = list(zip(*x))
        imu_acc_leaf_norm = torch.stack([torch.norm(_[:,:(self.imu_num - 1) * 3].view(-1,self.imu_num - 1,3),dim=-1) for _ in list(x)])
        
        leaf_joint = self.fuse_joint_position_estimation(jp1=torch.stack(leaf_joint_rnn),jp2=torch.stack(leaf_joint_gnn),beta=imu_acc_leaf_norm)
        leaf_joint_input = [leaf_joint[i] + torch.normal(0,0.04,size=leaf_joint[0].size()).to(leaf_joint[i].device) for i in range(len(leaf_joint))] if self.training and self.add_guassian else leaf_joint
        
        global_6d_pose = self.rnn3([torch.cat(_, dim=-1) for _ in zip(leaf_joint_input, x)]) 
        joint_velocity = self.rnn4(list(zip([torch.cat(_, dim=-1) for _ in zip(leaf_joint_input, x)], jvel_init))) if self.with_rnn_init \
                    else self.rnn4(list([torch.cat(_, dim=-1) for _ in zip(leaf_joint_input, x)]))
                    
        contact = self.rnn5([torch.cat(_, dim=-1) for _ in zip(leaf_joint_input, x)])
        
        return leaf_joint_rnn,leaf_joint_gnn,global_6d_pose,joint_velocity,contact
    
    def _process_input_imu(self,glb_acc,glb_rot,glb_uwb):
        return torch.cat([normalize_and_concat(glb_acc, glb_rot), glb_uwb.flatten(1)], dim=1)
    
    def _from_local_to_global_leaf_joint_position(self,ljp,pose,tran):
        leaf_joint_position = ljp.view(-1,5,3).permute(0,2,1)
        root_rot_mat = pose[:, 0,...]
        glb_lj_position = torch.bmm(root_rot_mat,leaf_joint_position).permute(0,2,1)
        return glb_lj_position
    
   
    @torch.no_grad()
    def predict(self, glb_acc, glb_rot, init_pose, **kwargs):
        r"""
        Predict the results for evaluation.

        :param glb_acc: A tensor that can reshape to [num_frames, 6, 3].
        :param glb_rot: A tensor that can reshape to [num_frames, 6, 3, 3].
        :param init_pose: A tensor that can reshape to [1, 24, 3, 3].
        :param glb_uwb: A tensor that can reshape to [num_frames, 6, 6].
        :return: Pose tensor in shape [num_frames, 24, 3, 3] and
                 translation tensor in shape [num_frames, 3].
        """
        if "vuwb" in self.imu_m:
            glb_uwb = kwargs["glb_uwb"]
            offset = kwargs["offset"]
        else:
            glb_uwb = None
        
        self.dynamics_optimizer.reset_states()
        if NO_INIT_POSE:
            init_x = (self._process_input_imu(glb_acc[[0]],glb_rot[[0]],glb_uwb[[0]]),)
            lj_init = self.gnn_jp_mapper([init_x])[0].view(-1)
        else:
            init_pose = init_pose.view(1, 24, 3, 3)
            init_pose[0, 0] = torch.eye(3)
            lj_init = self.forward_kinematics(init_pose)[1][0, joint_set.leaf].view(-1)
            
        jvel_init = torch.zeros(24 * 3)
        x = (self._process_input_imu(glb_acc,glb_rot,glb_uwb), lj_init, jvel_init)
        leaf_joint_rnn, leaf_joint_gnn, global_6d_pose, joint_velocity, contact = [_[0] for _ in self.forward([x])]
        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
        joint_velocity = joint_velocity.view(-1, 24, 3).bmm(glb_rot[:, -1].transpose(1, 2)) #* vel_scale
        pose_opt, tran_opt = [], []
        for p, v, c, a in zip(pose, joint_velocity, contact, glb_acc):
            p, t = self.dynamics_optimizer.optimize_frame(p, v, c, a)
            pose_opt.append(p)
            tran_opt.append(t)
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)
        glb_leaf_joint_rnn = self._from_local_to_global_leaf_joint_position(leaf_joint_rnn,pose_opt,tran_opt)
        glb_leaf_joint_gnn = self._from_local_to_global_leaf_joint_position(leaf_joint_gnn,pose_opt,tran_opt)
        return pose_opt, tran_opt, glb_leaf_joint_rnn,glb_leaf_joint_gnn       



