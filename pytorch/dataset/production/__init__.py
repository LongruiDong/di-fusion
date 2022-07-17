import numpy as np
import copy, math
from numba import jit

class FrameIntrinsic:
    def __init__(self, fx, fy, cx, cy, dscale):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.dscale = dscale
        # self.baseline = 

    def to_K(self):
        return np.asarray([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ])


class FrameData:
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.noisedepth = None
        self.gt_pose = None
        self.calib = None


class RGBDSequence:
    def __init__(self):
        self.frame_id = 0

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

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