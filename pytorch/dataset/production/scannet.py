'''
仿照iclnuim写scannet的接口
参考 nice-slam之前自己写的
'''
import cv2
import os, glob
import torch
from dataset.production import *
from pyquaternion import Quaternion
from pathlib import Path
from utils import motion_util
# -*- coding:utf-8 -*-

class ScanNet(RGBDSequence):
    def __init__(self, path: str, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False):
        super().__init__()
        self.path = Path(path) # dataset/scannet/scans/scene0000_00/frames    color/0.jpg  depth/0.png
        self.color_names = sorted([f"color/{t}" for t in os.listdir(self.path / "color")], key=lambda t: int(t[6:].split(".")[0]))
        self.depth_names = sorted([f"depth/{t}" for t in os.listdir(self.path / "depth")], key=lambda t: int(t[6:].split(".")[0])) # 排序的 rgb-d文件名
        self.calib = [577.590698, 578.729797, 318.905426, 242.683609, 1000.] #内参写死 scannet dscale 1000
        if first_tq is not None:
            self.first_iso = motion_util.Isometry(q=Quaternion(array=first_tq[3:]), t=np.array(first_tq[:3]))
        else: # 绕x轴旋转90度
            self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.7071067811865476, -0.7071067811865475, -0.0, -0.0])) #此函数 为 w i j k 实部在前

        if end_frame == -1: #直到末帧
            end_frame = len(self.color_names)

        self.color_names = self.color_names[start_frame:end_frame]# 截取子序列
        self.depth_names = self.depth_names[start_frame:end_frame]

        if load_gt: #默认不读入  
            gt_traj_path = os.path.join(self.path, 'pose') #Datasets/scannet/scans/scene0000_00/frames/pose dir
            self.gt_trajectory = self._parse_traj_file(gt_traj_path) # list 某个坐标系
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame] #截取
            change_iso = self.first_iso # .dot(self.gt_trajectory[0].inv()) # self.first_iso.dot以首帧为世界系并 改为新的世界系(参数给出的first pose)
            self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory] #change_iso.dot
            assert len(self.gt_trajectory) == len(self.color_names)
        else:
            self.gt_trajectory = None

    def _parse_traj_file(self, traj_path): #这里的输入参数 是pose 文件夹
        camera_ext = {} # 字典 key 是时间戳(frameid)
        pose_paths = sorted(glob.glob(os.path.join(traj_path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        nimg = len(pose_paths)
        cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0)) #旋转 绕z轴180 why？ R=-I[:,0:2] 即x y 取反 [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        for i in range(nimg):
            pose_path = os.path.join(traj_path, str(i)+'.txt')
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4) #所以scannet原始位姿是世界系
            # if (True in np.isinf(c2w)) or (True in np.isnan(c2w)): #scannet里 会有-inf
            #     print('c2w nan or inf at frame-', i)
            #     # continue
            cur_q = c2w[0:3, 0:3]# Quaternion(imaginary=cur_p[3:6], real=cur_p[-1]).rotation_matrix # 虚部 实部 转旋转矩阵
            cur_t = c2w[0:3, 3]
            # cur_q[1] = -cur_q[1] # 第二行取反 why
            # cur_q[:, 1] = -cur_q[:, 1] # 再对上面的第二列取反 最后等价于[0,1],[2,1],[1,0],[1,2] 都取反
            # cur_t[1] = -cur_t[1] #平移y取反
            # cur_q[:3, 1] *= -1 #nice-slam的做法
            # cur_q[:3, 2] *= -1
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q), t=cur_t)
            camera_ext[i] = (cur_iso) #cano_quat.dot为何变换？ 对于tartanaira来说可以把坐标系颠倒过来

        return [camera_ext[t] for t in range(len(camera_ext))]

    def __len__(self):
        return len(self.color_names)

    def __next__(self):
        if self.frame_id >= len(self):
            raise StopIteration

        depth_img_path = self.path / self.depth_names[self.frame_id]
        rgb_img_path = self.path / self.color_names[self.frame_id]

        # Convert depth image into point cloud.
        if '.png' in str(depth_img_path):
            depth_data = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
        elif '.npy' in str(depth_img_path): #对于tartanqir来说 gt depth 保存真实数值 m 0~ 10000（infinite）
            depth_data = np.load(str(depth_img_path))
            # 实际上给的深度是有大于 10000 即使是office 所以先clip吧
            depth_data = np.clip(depth_data, 0, 10000)
        
        depth_data = torch.from_numpy(depth_data.astype(np.float32)).cuda() / self.calib[4]
        rgb_data = cv2.imread(str(rgb_img_path))
        H, W = depth_data.shape # scannet 两者大小不同
        rgb_data = cv2.resize(rgb_data, (W, H))
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.

        frame_data = FrameData()
        frame_data.gt_pose = self.gt_trajectory[self.frame_id] if self.gt_trajectory is not None else None
        frame_data.calib = FrameIntrinsic(self.calib[0], self.calib[1], self.calib[2], self.calib[3], self.calib[4])
        frame_data.depth = depth_data
        frame_data.rgb = rgb_data

        self.frame_id += 1
        return frame_data
