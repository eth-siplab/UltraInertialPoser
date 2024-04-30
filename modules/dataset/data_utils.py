import numpy as np
import torch
from dataclasses import dataclass,fields

IMU_NUM = 6
def syc_signal(signal,sampling_ratio):
    frame_size = signal.size(0)
    s = frame_size // sampling_ratio + 1
    signal_interp = signal[torch.floor(torch.arange(s) * sampling_ratio).to(dtype=torch.long).clamp(min=0,max=frame_size-1)]
    indexs = np.arange(frame_size) // sampling_ratio
    signal_output = signal_interp[indexs]
    return signal_output.view(frame_size,-1)

@dataclass
class SeqInfo:
    """
    Seq Information saved as dataset cache
    """
    seq_id: int = None 
    frame_size: int = None # #N
    leaf_joint: torch.Tensor = None # N, 5, 3
    joint_vel: torch.Tensor = None #N, 24, 3
    pose_global_6d: torch.Tensor = None #N, 24, 6
    feet_contact: torch.Tensor = None #N, 2
    non_root_joint: torch.Tensor = None #N, 23, 3
    imu_ori: torch.Tensor = None #N, 6, 3, 3
    imu_acc: torch.Tensor = None #N, 6, 3
    uwb_m: torch.Tensor = None #N, 6, 6
    uwb_gt: torch.Tensor = None
    uwb_c: torch.Tensor = None #N, 6, 6
    offset: torch.Tensor = None #6, 3
    acc_sum: torch.Tensor = None #N, 6, 3
    
    def with_uwb(self):
        if self.uwb_m is not None and self.uwb_c is not None:
            return True
        return False
    
    def with_acc_sum(self):
        if self.acc_sum is not None:
            return True
        return False

    def syc_measurement(self,sampling_ratio,keys=['uwb_m','uwb_c']):
        for k in keys:
            signal = getattr(self,k)
            signal_syc = syc_signal(signal,sampling_ratio)
            setattr(self,k,signal_syc)
    
    def flatten_uwb_measurement(self):
        index = torch.triu_indices(IMU_NUM, IMU_NUM, 1)
        self.uwb_m = self.uwb_m.view(-1,IMU_NUM,IMU_NUM)[:,index[0],index[1]]
        self.uwb_c = self.uwb_c.view(-1,IMU_NUM,IMU_NUM)[:,index[0],index[1]]
        
@dataclass
class Batch:
    """
    Data Batch as input to the trained model
    """
    device = "cuda"
    x_imu: torch.Tensor = None
    lj_init: torch.Tensor = None
    jvel_init: torch.Tensor = None
    lfj_gt: torch.Tensor = None
    joint_gt: torch.Tensor = None
    jrot_6d: torch.Tensor = None
    jvel_gt: torch.Tensor = None
    contact_p: torch.Tensor = None
    uwb_gt: torch.Tensor = None
    vuwb: torch.Tensor = None
    uwb_offset: torch.Tensor = None
    uwb_normalized: bool = False
    
    @property
    def batch_size(self):
        return self.x_imu.size(0)
    
    def to_device(self,device:torch.device=None):
        _device = self.device if device is None else device
        for field in fields(self):
            name = field.name
            field_value = getattr(self, name)
            if field_value is not None and isinstance(field_value,torch.Tensor):
                setattr(self,name,field_value.to(_device)) 
        self.device = _device
        return self
    
    def get_listed_batch(self,keys=["x_imu","lj_init","jvel_init"]):
        x_input = []
        for b_i in range(self.batch_size):
            record = [getattr(self,k)[b_i].to(device=self.device) for k in keys]
            x_input.append(tuple(record))
        return x_input

    def get_tensor_batch(self,keys=["x_imu","lj_init","jvel_init"]):
        return tuple(getattr(self,k).to(device=self.device) for k in keys)

class D_Batch:
    """
    Turn dict into class attributes
    """
    def __init__(self, dictionary:dict):
        for k, v in dictionary.items():
            setattr(self, k, v)
    
    def __repr__(self) -> np.str:
        return str(self.__dict__)
    