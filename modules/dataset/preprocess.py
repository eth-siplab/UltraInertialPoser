r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.
    
    Adapted from https://github.com/Xinyu-Yi/PIP/blob/main/preprocess.py
"""
import articulate as art
import torch
import os
import pickle
from config.config import paths,amass_data,amass_test_data
import numpy as np
from tqdm import tqdm
import glob
import pytorch3d.transforms as pytf
try:
	from fairmotion.ops import conversions, quaternion
except:
	print("Did not find package fairmotion.ops")
    
vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021]) #lr wrist, lr knee, head, root
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
uwb_m_mapping= torch.tensor([[1,2],[1,4],[1,5],[1,3],[1,0]
                                 ,[2,4],[2,5],[2,3],[2,0]
                                       ,[4,5],[4,3],[4,0]
                                             ,[5,3],[5,0]
                                                   ,[3,0]])
uwb_f_mapping = torch.tensor([5,7,8,6,0,10,11,9,1,14,12,3,13,4,2])
uwb_imu_mapping = torch.tensor([1, 2, 4, 5, 3, 0])
#ji_mask = torch.tensor([20, 21, 4, 5, 15, 0])
body_model = art.ParametricModel(paths.smpl_file)

def plot_multidimensional_trajectory(data,title=None,block=True):
    import matplotlib.pyplot as plt
    """
    Plot each dimension vs frames 

    Parameters:
    - data: A numpy array with shape (frame_num, dim) representing the input sequence data.

    """
    num_frames, num_dimensions = data.shape

    # Create an array of frame numbers
    frames = np.arange(num_frames)

    # Set up a colormap for distinguishing dimensions
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, num_dimensions)]

    # Create a line plot for each dimension with color and legend
    for dim in range(num_dimensions):
        plt.plot(frames, data[:, dim], label=f'Dimension {dim}', color=colors[dim])

    # Add labels and title
    plt.xlabel('Frames')
    plt.ylabel('Dimension Value')
    p_title = 'Change of Multidimensional Data Over Frames' if title is None else title
    plt.title(p_title)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show(block=block)
    
    # plt.savefig(f'{title}.png')
    # plt.close()
    
def _compute_imu_local_offset(j_pos_glb,v_pos_glb,j_ori_glb):
    """
    Compute local offset between imu_placement and the corresponding joint position
    
    """
    r_glb = (j_pos_glb[:, ji_mask] - v_pos_glb[:,vi_mask])
    j_ori_glb = j_ori_glb[:, ji_mask]
    r_local_mean = torch.einsum('bij,bijk->bik', r_glb, j_ori_glb).mean(dim=0)
    return r_local_mean # 6,3
    

def _syn_uwb(p):
    """_summary_
    Synthesize UWB value from joint positions
    """
    return torch.cdist(p,p).view(-1,6,6)

def _syn_acc(v, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def process_amass(test_split=False):
    data_pose, data_trans, data_beta, length = [], [], [], []
    if test_split:
        ds_names = amass_test_data
    else:
        ds_names = amass_data
    
    for ds_name in ds_names:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran) #x,y,z -> x,z,-y
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    out_offset = []
    out_uwb = []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3  IMU is on l/r wrist
        #out_vacc.append(_syn_acc(joint[:, ji_mask]))
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3 IMU measures the orientation of l/r elbow

            #out_uwb.append(_syn_uwb(joint[:, ji_mask]))
        out_uwb.append(_syn_uwb(vert[:, vi_mask]))

        offset = _compute_imu_local_offset(joint, vert, grot)
        out_offset.append(offset)
        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    if test_split:
        test_folder = os.path.join(paths.amass_dir,f"test_split")
        os.makedirs(test_folder, exist_ok=True)
        torch.save(out_pose, os.path.join(test_folder, 'pose.pt'))
        torch.save(out_shape, os.path.join(test_folder, 'shape.pt'))
        torch.save(out_tran, os.path.join(test_folder, 'tran.pt'))
        torch.save(out_joint, os.path.join(test_folder, 'joint.pt'))
        torch.save(out_vrot, os.path.join(test_folder, 'vrot.pt'))
        torch.save(out_vacc, os.path.join(test_folder, 'vacc.pt'))
        torch.save(out_uwb, os.path.join(test_folder, 'vuwb.pt'))
        torch.save(out_offset, os.path.join(test_folder, 'offset.pt'))
        torch.save({'acc': out_vacc, 'ori': out_vrot, 'pose': out_pose, 'tran': out_tran, "vuwb":out_uwb,"offset":offset}, os.path.join(test_folder, "test.pt"))
    else:
        torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))
        torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))
        torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))
        torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))
        torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))
        torch.save(out_vacc, os.path.join(paths.amass_dir, 'vacc.pt'))
        torch.save(out_uwb, os.path.join(paths.amass_dir, 'vuwb.pt'))
        torch.save(out_offset, os.path.join(paths.amass_dir, 'offset.pt'))
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)
    #print('Synthetic AMASS dataset is saved at', os.path.join(paths.amass_dir,f"no_tc"))

    

def process_dipimu(data_split="train",sigma = 0):
    # with new data format, lw, rw, lk, rk, head, root
    imu_mask = [7, 8, 11, 12, 0, 2]
    
    if data_split == "train":
        split = ['s_01', 's_02','s_03', 's_04','s_05', 's_06','s_07'] #train sub
    elif data_split == "test":
        split = ['s_09', 's_10'] #test sub
    elif data_split == "validation":
        split = ['s_08']
    else:
        raise KeyError(f"Invalid split {split} for DIP IMU")
    accs, oris, poses, trans, v_uwb = [], [], [], [], []
    out_offset = []

    for subject_name in split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()
            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])
                
            pose_rot = art.math.axis_angle_to_rotation_matrix(pose[6:-6]).view(-1, 24, 3, 3)
            grot, joint,vert = body_model.forward_kinematics(pose_rot, None, None, calc_mesh=True)
            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
                uwb = _syn_uwb(vert[:, vi_mask])
                v_uwb.append(uwb)
                
                offset = _compute_imu_local_offset(joint, vert, grot)
                out_offset.append(offset)
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    file_name = f"{data_split}.pt"
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans, "vuwb": v_uwb, "offset":out_offset}, os.path.join(paths.dipimu_dir, file_name))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir)


def process_totalcapture(sigma = 0):
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'
    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([ 0, 1, 2, 3, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([ 0, 1, 2, 3, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue   # no SMPL poses
            f = open(os.path.join(paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    
    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    out_uwb,out_offset = [],[]
    # remove acceleration bias
    for iacc, pose, tran in zip(accs, poses, trans):
        pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        
        uwb = _syn_uwb(vert[:, vi_mask])
        out_uwb.append(uwb)
        
        offset = _compute_imu_local_offset(joint, vert, grot)
        out_offset.append(offset)
        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans, "vuwb": out_uwb, "offset":out_offset},
               os.path.join(paths.totalcapture_dir,'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir)

if __name__ == '__main__':
    process_amass(test_split=True)
    process_totalcapture()
    process_dipimu(data_split="test")

    
    