'''
仿照iclnuim写tartanair的接口
参考 nice-slam之前自己写的
'''
import cv2
import os
import torch
from dataset.production import *
from pyquaternion import Quaternion
from pathlib import Path
from utils import motion_util
# -*- coding:utf-8 -*-
# import model_noise, shuffle_rand

class TartanAir(RGBDSequence):
    def __init__(self, path: str, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, w_depth_noise = False,
                 create_noise = False):
        super().__init__()
        self.path = Path(path) # dataset/TartanAir/hospital/Easy/P000    image_left/000000_left.png  depth_left/000000_left_depth.npy
        self.color_names = sorted([f"image_left/{t}" for t in os.listdir(self.path / "image_left")], key=lambda t: int(t[11:].split(".")[0][0:6]))
        self.depth_names = sorted([f"depth_left/{t}" for t in os.listdir(self.path / "depth_left")], key=lambda t: int(t[11:].split(".")[0][0:6])) # 排序的 rgb-d文件名
        self.calib = [320.0, 320.0, 320.0, 240.0, 1.0, 0.050] #内参写死 tartanair dscale 1 增加baseline 模拟stereo kinect 0.075 m 0.05
        if first_tq is not None:
            self.first_iso = motion_util.Isometry(q=Quaternion(array=first_tq[3:]), t=np.array(first_tq[:3]))
        else: #为啥默认 把首帧坐标系这样设置 zx 取反？ 平移0  绕x轴转-180 其实就是R[:,1:3]=-I[:,1:3]取反 和nice-slam那边一样
            self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0])) #此函数 为 w i j k 实部在前

        if end_frame == -1: #直到末帧
            end_frame = len(self.color_names)

        self.color_names = self.color_names[start_frame:end_frame]# 截取子序列
        self.depth_names = self.depth_names[start_frame:end_frame]

        if load_gt: #默认不读入  
            gt_traj_path = os.path.join(self.path, 'pose_left.txt')
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
        traj_data0 = np.genfromtxt(traj_path) # N,7 tx ty tz i j k w
        nimg = traj_data0.shape[0]
        # https://github.com/princeton-vl/DROID-SLAM/blob/main/evaluation_scripts/validate_tartanair.py#L93
        # 若提前转换坐标 [1, 2, 0, 4, 5, 3, 6] ned -> xyz 效果和变换矩阵一样
        traj_data = traj_data0[:, [1, 2, 0, 4, 5, 3, 6]].astype(np.float64)
        cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0)) #旋转 绕z轴180 why？ R=-I[:,0:2] 即x y 取反 [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        # for cur_p in traj_data:# 首帧pose不是I 对于ICL-nium
        for i in range(nimg):
            cur_p = traj_data[i]
            cur_q = Quaternion(imaginary=cur_p[3:6], real=cur_p[-1]).rotation_matrix # 虚部 实部 转旋转矩阵
            cur_t = cur_p[0:3]
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
            # 实际上给的深度是有大于 10000 即使是office 所以先clip吧 25.
            depth_data = np.clip(depth_data, 0, 10000.) #因为反正算法就不能建出很远的 就这里直接cut
        
        depth_data = depth_data.astype(np.float32)/ self.calib[4] #/ self.calib[4] # 真实尺度
        # depth_data = torch.from_numpy(depth_data.astype(np.float32)).cuda() / self.calib[4]
        rgb_data = cv2.imread(str(rgb_img_path))
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.

        frame_data = FrameData()
        frame_data.gt_pose = self.gt_trajectory[self.frame_id] if self.gt_trajectory is not None else None
        frame_data.calib = FrameIntrinsic(self.calib[0], self.calib[1], self.calib[2], self.calib[3], self.calib[4])
        # frame_data.depth = depth_data
        frame_data.rgb = rgb_data

        if (self.depth_noise and self.create_noise): # 生成并使用
            # 先shuffle (前后没差)
            # depth_data = depth_data.cpu().numpy().astype(np.float32) # m -> cm  * 100.
            # 转为视差 bf/z
            bf = self.calib[5] * self.calib[0] # 351.30 试下论文里说的  * 100.
            disp_data = bf / depth_data #depth_data _shuffle
            disp_data_wnoise = model_noise(disp_data, depth_data, bf, sigma_d=(0.5), sigma_s=1.5) # 3. 2.5 1. 1./6
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
            # depth_data_wnoise_shuffle = cv2.imread(noisepath, cv2.IMREAD_UNCHANGED) # 下面别忘了尺度
            depth_data_wnoise_shuffle = np.load(noisepath)
            assert  depth_data_wnoise_shuffle.shape[0] > 0 
            frame_data.noisedepth = torch.from_numpy(depth_data_wnoise_shuffle.astype(np.float32) / self.calib[4]).cuda() #  / 100.
            frame_data.depth = torch.from_numpy(depth_data_wnoise_shuffle.astype(np.float32) / self.calib[4]).cuda()

        elif not(self.depth_noise):
            # 正常用原图
            frame_data.depth = torch.from_numpy(depth_data.astype(np.float32)).cuda()

        self.frame_id += 1
        return frame_data
