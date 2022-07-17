from ast import Not
import cv2
import os, copy, scipy, math
import torch
from dataset.production import *
from pyquaternion import Quaternion
from pathlib import Path
from utils import motion_util
from scipy import interpolate
# -*- coding:utf-8 -*-
from numba import jit

#在给定2d array上某个索引位置进行双线性插值
def bilinear_interp(dataary, qx, qy):
    dst = copy.deepcopy(dataary)
    h ,w = dataary.shape
    # 计算在源图上 4 个近邻点的位置
    # i,j h w
    i0 = max(int(np.floor(qx)), 0)
    if i0 >= h-1:
        i0 = i0-1
    i1 = min(i0+1,h-1)
    j0 = max(int(np.floor(qy)), 0) # 可能会有负值
    if j0 >= w-1:
        j0 = j0-1
    j1 = min(j0+1, w-1)
    
    tmp0 = ((j1-qy)*dst[i0,j0] + (qy-j0)*dst[i0,j1]) / (j1-j0)
    tmp1 = ((j1-qy)*dst[i1,j0] + (qy-j0)*dst[i1,j1])  / (j1-j0)
    
    outv = ((i1-qx)*tmp0 + (qx-i0)*tmp1) / (i1-i0)
    if outv==0.: # j0 和 j1 都是639
        print('here!')
    return outv
#nopython=True
@jit(nopython=True)
def model_noise(disp_data, depth_data, bf, sigma_d=3., sigma_s=0.5):
    hf, wf = disp_data.shape
    newdisp = np.zeros_like(disp_data)# copy.deepcopy(disp_data) float32
    mean = (0, 0)
    cov = [[sigma_s**2, 0], [0, sigma_s**2]]
    # wx = np.arange(0, wf, 1)
    # hy = np.arange(0, hf, 1)
    # xx, yy = np.meshgrid(wx, hy)
    
    for i in range(hf):
        for j in range(wf):
            # dx, dy = np.random.multivariate_normal(mean, cov) # numba不支持 要替换
            # qx = i+dx
            # qy = j+dy
            # # if i >= hf-1 or j >= wf-1:
            # #     print('--')
            # if qx>=hf-1:
            #     qx = hf-1 #qx - 1.
            # if qy>=wf-1:
            #     qy = wf-1 #qy - 1.
            # d_query = depth_data[qx, qy]
            # d_query = bilinear_interp(depth_data, qx, qy)
            # if d_query == 0.: # j=639
            #     print('warning: d_query==0 !')
            # disp_query = bf / (d_query+0.0) #可能会有0
            # disp_query = bilinear_interp(disp_data, qx, qy)
            gtz = depth_data[i,j]
            gtd = disp_data[i,j]
            stdij = sigma_d * gtz * gtz / (bf) # 实际方差和深度平方 正相关 bf/d__2
            # stdij = sigma_d * bf / (gtd*gtd) # 那个截断形状和  乘方差的因子无关
            gaussdelta = np.random.normal(0, stdij, (1)).astype(np.float32)[0] # 均值0的正态分布
            # newdisp[i,j] = np.round(disp_query + gaussdelta + 0.5) # np.floor round 向下取整 或者四舍五入 (截断效果的原因！)参照icl-nuim的噪声建模 也可能为0
            newdisp[i,j] = gtd + gaussdelta # disp_query  + 0.5
    # 返回增加噪声的视差
    return newdisp

# @jit(nopython=True)
def shuffle_rand(datarr, sigma_s=0.25):
    #对数据整体 shuffle 以标准差sigma_s
    outdata = copy.deepcopy(datarr)
    hf, wf = outdata.shape
    mean = (0, 0)
    cov = [[sigma_s**2, 0], [0, sigma_s**2]]
    for i in range(hf):
        for j in range(wf):
            dx, dy = np.random.multivariate_normal(mean, cov)
            # qx = min(int(np.floor(i+dx)), hf-1)
            # qy = min(int(np.floor(j+dy)), wf-1)
            qx = i+dx
            qy = j+dy
            # if i >= hf-1 or j >= wf-1:
            #     print('--')
            if qx>=hf-1:
                qx = hf-1 #round(qx - 1.)
            else:
                qx = max(0, math.ceil(qx))
            if qy>=wf-1:
                qy = wf-1 #round(qy - 1.)
            else:
                qy = max(0, math.ceil(qy))
            
            outdata[i, j] = datarr[qx, qy]
    
    return outdata
    
            
            

class ICLNUIMSequence(RGBDSequence):
    def __init__(self, path: str, start_frame: int = 0, end_frame: int = -1, first_tq: list = None, load_gt: bool = False, w_depth_noise = False,
                 create_noise = False):
        super().__init__()
        self.path = Path(path) #dataset/ICL/living_room_traj0_frei_png
        self.color_names = sorted([f"rgb/{t}" for t in os.listdir(self.path / "rgb")], key=lambda t: int(t[4:].split(".")[0]))
        self.depth_names = [f"depth/{t}.png" for t in range(len(self.color_names))] # 排序的 rgb-d文件名
        self.calib = [481.2, 480.0, 319.50, 239.50, 5000.0, 0.050] #内参写死 5000是depth scale(ICL 的tum rgb-d格式) 增加baseline 模拟stereo kinect 0.075 m 
        if first_tq is not None:
            self.first_iso = motion_util.Isometry(q=Quaternion(array=first_tq[3:]), t=np.array(first_tq[:3]))
        else: #为啥默认 把首帧坐标系这样设置 zx 取反？ 平移0  绕x轴转-180 其实就是R[:,1:3]=-I[:,1:3]取反 和nice-slam那边一样
            self.first_iso = motion_util.Isometry(q=Quaternion(array=[0.0, -1.0, 0.0, 0.0])) #此函数 为 w i j k 实部在前

        if end_frame == -1: #直到末帧
            end_frame = len(self.color_names)

        self.color_names = self.color_names[start_frame:end_frame]# 截取子序列
        self.depth_names = self.depth_names[start_frame:end_frame]

        if load_gt: #默认不读入  
            gt_traj_path = (list(self.path.glob("*.freiburg")) + list(self.path.glob("groundtruth.txt")))[0]
            self.gt_trajectory = self._parse_traj_file(gt_traj_path) # list 某个坐标系
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame] #截取
            change_iso = self.first_iso.dot(self.gt_trajectory[0].inv()) # self.first_iso.dot以首帧为世界系并 改为新的世界系(参数给出的first pose)
            self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory] # change_iso.dot
            assert len(self.gt_trajectory) == len(self.color_names)
        else:
            self.gt_trajectory = None
        self.depth_noise = w_depth_noise # 是否使用噪声深度
        self.create_noise = create_noise # 是否同时生成噪声深度图
        print('use noise depth: {}, create it: {}'.format(self.depth_noise, self.create_noise))

    def _parse_traj_file(self, traj_path):
        camera_ext = {} # 字典 key 是时间戳(frameid)
        traj_data = np.genfromtxt(traj_path) # N,8 fid tx ty tz i j k w
        cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0)) #旋转 绕z轴180 why？ R=-I[:,0:2] 即x y 取反 [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        for cur_p in traj_data:# 首帧pose不是I 对于ICL-nium
            cur_q = Quaternion(imaginary=cur_p[4:7], real=cur_p[-1]).rotation_matrix # 虚部 实部 转旋转矩阵
            cur_t = cur_p[1:4]
            cur_q[1] = -cur_q[1] # 第二行取反 why
            cur_q[:, 1] = -cur_q[:, 1] # 再对上面的第二列取反 最后等价于[0,1],[2,1],[1,0],[1,2] 都取反
            cur_t[1] = -cur_t[1] #平移y取反
            cur_iso = motion_util.Isometry(q=Quaternion(matrix=cur_q), t=cur_t)
            camera_ext[cur_p[0]] = cano_quat.dot(cur_iso) #cano_quat.dot为何变换？ 要搞清icl的坐标系是啥
        camera_ext[0] = camera_ext[1] # ICL 的首帧id 是1 不是0 所以复制为第0帧 及前两帧一样
        return [camera_ext[t] for t in range(len(camera_ext))]

    def __len__(self):
        return len(self.color_names)

    def __next__(self):
        if self.frame_id >= len(self):
            raise StopIteration

        depth_img_path = self.path / self.depth_names[self.frame_id]
        rgb_img_path = self.path / self.color_names[self.frame_id]

        # Convert depth image into point cloud.
        depth_data = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32)/ self.calib[4] #/ self.calib[4] # 真实尺度
        frame_data = FrameData()
        depth_data = torch.from_numpy(depth_data.astype(np.float32)).cuda() # / self.calib[4]
        
        rgb_data = cv2.imread(str(rgb_img_path))
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.

        
        frame_data.gt_pose = self.gt_trajectory[self.frame_id] if self.gt_trajectory is not None else None
        frame_data.calib = FrameIntrinsic(self.calib[0], self.calib[1], self.calib[2], self.calib[3], self.calib[4])
        # frame_data.depth = depth_data
        frame_data.rgb = rgb_data
        
        if (self.depth_noise and self.create_noise): # 生成并使用
            # 先shuffle (前后没差)
            # depth_data_shuffle = shuffle_rand(depth_data, sigma_s=1.25)
            depth_data = depth_data.cpu().numpy().astype(np.float32) # m -> cm  * 100.
            # 转为视差 bf/z
            bf = self.calib[5] * self.calib[0] # 351.30 试下论文里说的  * 100.
            disp_data = bf / depth_data #depth_data _shuffle
            disp_data_wnoise = model_noise(disp_data, depth_data, bf, sigma_d=(3.0), sigma_s=1.5) # 3. 2.5 1. 1./6
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
            # pass
        elif not(self.depth_noise):
            # 正常用原图
            frame_data.depth = depth_data
            # pass

        self.frame_id += 1
        return frame_data
