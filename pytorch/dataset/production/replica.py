'''
仿照iclnuim写replica的接口
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

class Replica(RGBDSequence):
    def __init__(self, path: str, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, w_depth_noise = False,
                 create_noise = False):
        super().__init__()
        self.path = Path(path) # dataset/Replica/office0    results/frame000000.jpg  results/depth000000.png
        # self.color_names = sorted([f"results/{t}" for t in os.listdir(self.path / "results")], key=lambda t: int(t[11:].split(".")[0][0:6]))
        # self.depth_names = sorted([f"results/{t}" for t in os.listdir(self.path / "results")], key=lambda t: int(t[11:].split(".")[0][0:6])) # 排序的 rgb-d文件名
        self.color_names = sorted([f"results/{os.path.basename(t)}" 
                                   for t in glob.glob(f'{self.path}/results/frame*.jpg')])
        self.depth_names = sorted([f"results/{os.path.basename(t)}" 
                                   for t in glob.glob(f'{self.path}/results/depth*.png')])
        self.calib = [600.0, 600.0, 599.5, 339.5, 6553.5, 0.050] #内参写死 replica uint16 10m内 增加baseline 模拟stereo kinect 0.075 m 0.05
        if first_tq is not None:
            self.first_iso = motion_util.Isometry(q=Quaternion(array=first_tq[3:]), t=np.array(first_tq[:3]))
        else: # 绕x轴旋转90度
            self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.7071067811865476, -0.7071067811865475, -0.0, -0.0])) #此函数 为 w i j k 实部在前

        if end_frame == -1: #直到末帧
            end_frame = len(self.color_names)

        self.color_names = self.color_names[start_frame:end_frame]# 截取子序列
        self.depth_names = self.depth_names[start_frame:end_frame]

        if load_gt: #默认不读入  
            gt_traj_path = os.path.join(self.path, 'traj.txt') # 4*4 SE3
            self.gt_trajectory = self._parse_traj_file(gt_traj_path) # list 某个坐标系
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame] #截取
            change_iso = self.first_iso#.dot(self.gt_trajectory[0].inv()) # self.first_iso.dot以首帧为世界系并 改为新的世界系(参数给出的first pose)
            self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory] #change_iso.dot
            assert len(self.gt_trajectory) == len(self.color_names)
        else:
            self.gt_trajectory = None
        self.depth_noise = w_depth_noise # 是否使用噪声深度
        self.create_noise = create_noise # 是否同时生成噪声深度图
        print('use noise depth: {}, create it: {}'.format(self.depth_noise, self.create_noise))

    def _parse_traj_file(self, traj_path):
        camera_ext = {} # 字典 key 是时间戳(frameid)
        traj_data0 = np.genfromtxt(traj_path) # N,16 4*4 SE3
        nimg = traj_data0.shape[0]
        with open(traj_path, "r") as f:
            lines = f.readlines()
        cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0)) #旋转 绕z轴180 why？ R=-I[:,0:2] 即x y 取反 [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        for i in range(nimg):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            cur_q = c2w[0:3, 0:3]# Quaternion(imaginary=cur_p[3:6], real=cur_p[-1]).rotation_matrix # 虚部 实部 转旋转矩阵
            cur_t = c2w[0:3, 3]
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
        
        depth_data = depth_data.astype(np.float32)/ self.calib[4] #/ self.calib[4] # 真实尺度
        # depth_data = torch.from_numpy(depth_data.astype(np.float32)).cuda() / self.calib[4]
        rgb_data = cv2.imread(str(rgb_img_path))
        H, W = depth_data.shape # 
        rgb_data = cv2.resize(rgb_data, (W, H))
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.

        frame_data = FrameData()
        frame_data.gt_pose = self.gt_trajectory[self.frame_id] if self.gt_trajectory is not None else None
        frame_data.calib = FrameIntrinsic(self.calib[0], self.calib[1], self.calib[2], self.calib[3], self.calib[4])
        # frame_data.depth = depth_data
        frame_data.rgb = rgb_data

        if (self.depth_noise and self.create_noise): # 生成并使用
            # 转为视差 bf/z
            bf = self.calib[5] * self.calib[0] # 3
            disp_data = bf / depth_data #depth_data _shuffle
            disp_data_wnoise = model_noise(disp_data, depth_data, bf, sigma_d=(6.5), sigma_s=1.5) # 3. 2.5 1. 1./6
            # 整体的schuffle
            disp_data_wnoise_shuffle = disp_data_wnoise # shuffle_rand(disp_data_wnoise, sigma_s=0.75) # shuffle_rand(disp_data_wnoise, sigma_s=0.75) # 0.25 0.5 
            # 再对增加噪声的视差转为噪声深度图
            depth_data_wnoise_shuffle = bf / disp_data_wnoise_shuffle
            # 也保存噪声图 单位m
            frame_data.noisedepth = torch.from_numpy(depth_data_wnoise_shuffle.astype(np.float32)).cuda() #  / 100.
            frame_data.depth = torch.from_numpy(depth_data_wnoise_shuffle.astype(np.float32)).cuda()
        elif self.depth_noise and not(self.create_noise):
            # 只是读入已有的噪声图
            mydir = os.path.join(self.path, 'mydnoise')
            noisepath = os.path.join(mydir, os.path.basename(str(depth_img_path)))
            depth_data_wnoise_shuffle = cv2.imread(noisepath, cv2.IMREAD_UNCHANGED) # 下面别忘了尺度
            assert  depth_data_wnoise_shuffle.shape[0] > 0 
            frame_data.noisedepth = torch.from_numpy(depth_data_wnoise_shuffle.astype(np.float32) / self.calib[4]).cuda() #  / 100.
            frame_data.depth = torch.from_numpy(depth_data_wnoise_shuffle.astype(np.float32) / self.calib[4]).cuda()

        elif not(self.depth_noise):
            # 正常用原图
            frame_data.depth = torch.from_numpy(depth_data.astype(np.float32)).cuda()
        
        self.frame_id += 1
        return frame_data
