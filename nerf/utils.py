import os
import glob
import tqdm

import random


import numpy as np
import pandas as pd
import time
from skimage import measure
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

# from rich.console import Console
# from torch_ema import ExponentialMovingAverage

from packaging import version as pver

from scipy.spatial import cKDTree as KDTree

import trimesh
import numpy.linalg as LA



import imageio



import numpy as np
import pandas as pd


import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
# from rich.console import Console
# from torch_ema import ExponentialMovingAverage


import numpy as np

import numpy.linalg as LA


from scipy.spatial import cKDTree as KDTree
import torch
from skimage import measure
import trimesh

def get_grid_uniform(resolution,rangee=1.0):
        x = np.linspace(-rangee, rangee, resolution)
        y = x
        z = x

        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

        return {"grid_points": grid_points.cuda(),
                "xyz": [x, y, z]}

def get_surface( model,resolution=100):
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        with torch.no_grad():
            occ=model(pnts).detach().cpu().numpy().squeeze()
            
        z.append(occ)
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                    grid['xyz'][0][2] - grid['xyz'][0][1],
                    grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()


        
        meshexport = trimesh.Trimesh(verts, faces, normals)
        
        return 0,meshexport
    else:
        return 0,0
    return None

def get_distance(target,source):
    kd_tree = KDTree(target)
    #for every source point we need find its closest distance in the target
    one_distances, _ = kd_tree.query(source)
    distance_source_to_target = np.mean(one_distances)
    a=0
    return distance_source_to_target
def calculate_champfer_distance(mesh_est,mesh_gt):
    sample_gt=trimesh.sample.sample_surface(mesh_gt, 100000)[0]
    sample_est = trimesh.sample.sample_surface(mesh_est, 100000)[0]
    d_acc=get_distance(sample_gt, sample_est)

    # for every gt point, what is the closest point in the est
    d_com = get_distance( sample_est,sample_gt)
    print("accuracy %f"%(d_acc))
    print("completion %f" % (d_com))
    print("chamfer %f" % ((d_com+d_acc)/2))
    return ((d_com+d_acc)/2)


def get_oriented_pc(centered_data):
   
    cov = np.cov(centered_data)
    eval, evec = LA.eig(cov)

    aligned_coords = np.matmul(evec.T, centered_data)

    xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])

    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                        [y1, y2, y2, y1, y1, y2, y2, y1],
                                                        [z1, z1, z1, z1, z2, z2, z2, z2]])
    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
    central_point= np.matmul(evec, np.array([[(xmax+xmin)/2],[(ymax+ymin)/2],[(zmin+zmax)/2]]))
    return rrc,central_point


def init_model(opt):
    try:
        gt_mesh=trimesh.load_mesh("data_processeing/redwood_dataset/GT/%s.ply"%opt.object_id_number)
        use_gt_mesh_for_eval=True
    except Exception:
        use_gt_mesh_for_eval=False   
        gt_mesh=0  

    angle_azimuth_depth=opt.angle_azimuth_depth
    
    
    depth_im=imageio.imread(opt.depth_path)
    depth_im_mask=depth_im>0
    K=np.array([[525,0,319.5],[0,525,239.5],[0,0,1]])
    pixels_infra=get_specific_pixels_infra(torch.from_numpy(K).float(),torch.eye(3),width=640,height=480)
    


    normal_to_plane=torch.from_numpy(np.load("data_processing/redwood_dataset/world_planes/%s.npy"%opt.object_id_number))
   
    
    angle_to_normal=(torch.acos(normal_to_plane[2])*180/np.pi).item()
    angle_to_top_view=180-angle_to_normal
    angle_to_horizontal_view=90-angle_to_normal
    angle_to_bottom_view=-angle_to_normal

    
    ys,xs=np.where(depth_im>0)
    z=depth_im[ys,xs]
    
    K_inv=np.linalg.inv(K)
    point_cloud= (K_inv@np.stack((xs,ys,np.ones_like(xs))))*z[np.newaxis,:]
    distances_im=np.zeros_like(depth_im).astype(np.float32)
    distances_im[ys,xs]=np.linalg.norm(point_cloud,axis=0)

    point_cloud=point_cloud.T
    data__p=torch.from_numpy(point_cloud).float()
    mean_vertex=data__p.mean(dim=0)
    data__p-=mean_vertex.unsqueeze(0)

    rrc,central_point=get_oriented_pc(data__p.numpy().T)
    rrc=torch.from_numpy(rrc).float()
    
    if rrc.norm(dim=0).max()/rrc.norm(dim=0).min()>1.7:   
        # for non-isotropic objects we use the oriented bounding box     
        mean_vertex=(mean_vertex.unsqueeze(0)+torch.from_numpy(central_point.T)).squeeze().float()
        data__p-=torch.from_numpy(central_point.T)


    mean_scale= data__p.norm(dim=1).max()
    scale_vertices=0.5/mean_scale
    data__p*=(0.5/mean_scale)
    
    homography_3d=torch.eye(4)
    homography_3d[0,3]=-mean_vertex[0]
    homography_3d[1,3]=-mean_vertex[1]
    homography_3d[2,3]=-mean_vertex[2]


    homography_3d_2=torch.eye(4)
    homography_3d_2[0,0]=0.5/mean_scale
    homography_3d_2[1,1]=0.5/mean_scale
    homography_3d_2[2,2]=0.5/mean_scale

    norm_homography= homography_3d_2@homography_3d

    
    plane_eq=torch.from_numpy(np.load("data_processing/redwood_dataset/world_planes/%s.npy"%opt.object_id_number)).float()
    plane_eq_normalize=(torch.inverse(norm_homography).T)@(plane_eq.unsqueeze(1))
    plane_eq_normalize/=plane_eq_normalize[:3].norm()


    points_on_surface=data__p.cuda().T
    return normal_to_plane,gt_mesh,use_gt_mesh_for_eval,angle_azimuth_depth,depth_im,depth_im_mask,pixels_infra, plane_eq,angle_to_normal,angle_to_top_view,angle_to_horizontal_view,angle_to_bottom_view,distances_im,data__p,mean_vertex,scale_vertices,norm_homography,plane_eq,plane_eq_normalize,points_on_surface
    



def get_specific_pixels_infra(K,R,width=64,height=48):
   
    yy, xx = torch.meshgrid(torch.linspace(0, height-1, steps=height), torch.linspace(0, width-1, steps=width))
    pixels_locations=torch.stack((xx.flatten(),yy.flatten(),torch.ones_like(yy.flatten())))
    pixel_normalized_locations=K.inverse()@pixels_locations

    normalization_factors=pixel_normalized_locations.norm(dim=0,keepdim=True)
    pixel_directions=pixel_normalized_locations/pixel_normalized_locations.norm(dim=0,keepdim=True)

    mapped_rays=(R.float().cuda()@(pixel_directions.view(3,-1).T.cuda()).T).T

    return normalization_factors,pixel_directions,mapped_rays,pixels_locations


def get_dir_index(rendered_angle_to_normal,rendered_azimuth):
    if rendered_angle_to_normal>140:
            dirs=4 #overhead
    elif rendered_angle_to_normal<40:
        dirs=5 #bottom
    elif (rendered_azimuth)>135 and (rendered_azimuth)<225:
        dirs=2 #back
    elif (rendered_azimuth)>45 and (rendered_azimuth)<=135:
        dirs=1 #side
    elif (rendered_azimuth)<315 and (rendered_azimuth)>=225:
        dirs=3 #side
    else:
        dirs=0 #front
    
    return dirs

def render_image(opt,angle_to_top_view,angle_to_bottom_view,angle_azimuth_depth,angle_to_normal,
                 mean_vertex,scale_vertices,normal_to_plane,model,
                 angle_rot_add=-45.0,sds_W=64, sds_H=48,angle_rotx_add=0,white_background=False,return_extrinsics=False,scaling=1.0,shading = 'albedo',ambient_ratio = 1.0):
    #Make sure we see the object in natural pose (not flipped)
    angle_rotx_add=np.minimum(angle_rotx_add,angle_to_top_view)
    angle_rotx_add=np.maximum(angle_rotx_add,angle_to_bottom_view)
    
    #Absolute angles for detemining the direction text
    rendered_azimuth=angle_azimuth_depth-(angle_rot_add)
    rendered_angle_to_normal=angle_to_normal+(angle_rotx_add)
    rendered_azimuth=(rendered_azimuth+360*4)%360



    angle_rot_add*=np.pi/180
    angle_rotx_add*=np.pi/180
    

    if sds_W==sds_H:
        K=torch.from_numpy(np.array([[525,0,319.5],[0,525,239.5+80],[0,0,1]])).float()
    else:
        K=torch.from_numpy(np.array([[525,0,319.5],[0,525,239.5],[0,0,1]])).float()


    B=1
    
    scale_mat=torch.diag(torch.tensor([sds_W/640.0,sds_W/640.0,1]))
    try:
        if white_background==True:
            pixels_infra=get_specific_pixels_infra(scale_mat@K,torch.eye(3),width=sds_W,height=sds_H)
            pixels_rays_d=pixels_infra[2].unsqueeze(0)
            pixels_rays_o=-mean_vertex.unsqueeze(0).unsqueeze(0)*scale_vertices
            pixels_rays_o=pixels_rays_o.tile(1,pixels_rays_d.shape[1],1)
        else: 
            # print("loaded cached")
            pixels_infra=pixels_infra_for_fast
            pixels_rays_d=pixels_infra[2].unsqueeze(0)
            pixels_rays_o=-mean_vertex.unsqueeze(0).unsqueeze(0)*scale_vertices
            pixels_rays_o=pixels_rays_o.tile(1,pixels_rays_d.shape[1],1)
    except Exception:
        pixels_infra_for_fast=get_specific_pixels_infra(scale_mat@K,torch.eye(3),width=sds_W,height=sds_H)
        pixels_infra=pixels_infra_for_fast
        pixels_rays_d=pixels_infra[2].unsqueeze(0)
        pixels_rays_o=-mean_vertex.unsqueeze(0).unsqueeze(0)*scale_vertices
        pixels_rays_o=pixels_rays_o.tile(1,pixels_rays_d.shape[1],1)
        
    #Get random color for background
    bg_color = torch.ones((pixels_rays_o.shape[1] * B, 3), device='cuda')*(torch.rand(3).unsqueeze(0).cuda()) 
    if white_background:
        bg_color*=0
        bg_color+=1.0

    #Rotation matrix around plane's normal
    rotation_ex=torch.from_numpy(cv2.Rodrigues(angle_rot_add*normal_to_plane[:3].T.numpy())[0]).float()
    
    #Rotation around the size direction (elevation rotation)
    side_dir=torch.cross(normal_to_plane[:3].float(),torch.tensor([0,0,1.0]))
    side_dir=side_dir/side_dir.norm()
    rotation_elevation=torch.from_numpy(cv2.Rodrigues(angle_rotx_add*side_dir.numpy())[0]).float()

    #Apply both rotations
    pixels_rays_o2=(rotation_ex@rotation_elevation@pixels_rays_o.squeeze().T).T.unsqueeze(0)
    pixels_rays_d2=(rotation_ex@rotation_elevation@pixels_rays_d.cpu().squeeze().T).T.unsqueeze(0)
    pixels_rays_o2*=scaling
    
    
    outputs_pixels = model.render(pixels_rays_o2.cuda(), pixels_rays_d2.cuda(), staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(opt))
    
    
    pred_rgb = outputs_pixels['image'].reshape(1, sds_H, sds_W, 3).permute(0, 3, 1, 2)
    pred_rgb_depth=outputs_pixels['depth'].reshape(B,  sds_H, sds_W, 1).permute(0, 3, 1, 2)
    pred_mask =outputs_pixels['weights_sum'].reshape(B,  sds_H, sds_W, 1).permute(0, 3, 1, 2) 
    
    return pred_rgb,pred_rgb_depth,pred_mask,pixels_rays_o2,pixels_rays_d2,rendered_azimuth,rendered_angle_to_normal



def get_angles(epoch,angle_to_horizontal_view):
    if epoch<20:
            angle_rot_add=(torch.rand(1).item()-0.5)*0.0
            angle_rotx_add=0
    elif epoch<50:
        angle_rot_add=(torch.rand(1).item()-0.5)*60.0
        rand_factorr=torch.rand(1).item() #in [0,1]
        angle_rotx_add=rand_factorr*0+(1-rand_factorr)*angle_to_horizontal_view
    elif epoch<80:
        angle_rot_add=(torch.rand(1).item()-0.5)*90.0
        rand_factorr=torch.rand(1).item() #in [0,1]
        angle_rotx_add=rand_factorr*0+(1-rand_factorr)*angle_to_horizontal_view
    elif epoch<100:
        angle_rot_add=(torch.rand(1).item()-0.5)*120.0
        rand_factorr=torch.rand(1).item() #in [0,1]
        angle_rotx_add=rand_factorr*0+(1-rand_factorr)*angle_to_horizontal_view
    elif epoch<120:
        angle_rot_add=(torch.rand(1).item()-0.5)*180.0
        rand_factorr=torch.rand(1).item() #in [0,1]
        angle_rotx_add=rand_factorr*0+(1-rand_factorr)*angle_to_horizontal_view
    else:
        angle_rot_add=(torch.rand(1).item()-0.5)*360.0
        rand_factorr=torch.rand(1).item() #in [0,1]
        angle_rotx_add=rand_factorr*0+(1-rand_factorr)*angle_to_horizontal_view
    return angle_rot_add,angle_rotx_add
        

def get_sensor_losses(pixels_infra,mean_vertex,scale_vertices,model,opt,B,ambient_ratio,shading,depth_rgb_supervision_mask,opacity_supervision,depth_im):
    cur_pixels_inds=torch.randint(pixels_infra[0].shape[1],(2000,))
    cur_pixels_rays_d=pixels_infra[2][cur_pixels_inds,:].unsqueeze(0)
    pixels_rays_o=-mean_vertex.unsqueeze(0).unsqueeze(0)*scale_vertices
    
    cur_pixels_rays_o=pixels_rays_o.tile(1,2000,1)
    cur_pixels_rays_o=(cur_pixels_rays_o.squeeze().T).T.unsqueeze(0)
    cur_pixels_rays_d=(cur_pixels_rays_d.cpu().squeeze().T).T.unsqueeze(0)


    bg_color = torch.ones((cur_pixels_rays_o.shape[1] * B, 3), device="cuda")*(torch.rand(3).unsqueeze(0).cuda()) # pixel-wise random        
    outputs_pixels_high_res = model.render(cur_pixels_rays_o.cuda(), cur_pixels_rays_d.cuda(), staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(opt))


    pred_depth_pixels_cur = outputs_pixels_high_res['depth']
    pred_mask_pixels_cur = outputs_pixels_high_res['weights_sum']
    
    j_s=pixels_infra[3][0,cur_pixels_inds].round().long()
    i_s=pixels_infra[3][1,cur_pixels_inds].round().long()
    
    opacity_supervision_batch=opacity_supervision[i_s,j_s]
    depth_rgb_supervision_mask_batch=depth_rgb_supervision_mask[i_s,j_s]
    
    gt_depth_batch=depth_im[i_s,j_s]
    
            
    gt_depth_pixels_batch=torch.from_numpy(gt_depth_batch[depth_rgb_supervision_mask_batch].astype(np.float32)).cuda()*scale_vertices
    depth_loss_batch =  ((pred_depth_pixels_cur[0,depth_rgb_supervision_mask_batch]-gt_depth_pixels_batch)**2).mean()
    
    mask_loss_batch  = (torch.from_numpy(opacity_supervision_batch).cuda()*1.0-pred_mask_pixels_cur.squeeze()).abs().mean()

    return 0,depth_loss_batch,mask_loss_batch


def get_losses(opt,pixels_infra,mean_vertex,scale_vertices,model,points_on_surface,plane_eq_normalize,depth_im_mask,distances_im):
    loss=0
    #The sdf should be zero at the point cloud locations:
    sdfs_point_cloud=model.sdf_net(points_on_surface.T)
    loss_point_cloud= sdfs_point_cloud.abs().mean()
    
    #Eikonal regularization 
    eikonal_loss_surface=(model.sdf_net.gradient(points_on_surface.T).squeeze().norm(dim=1)-1).abs().mean()
    uniform_points=((torch.rand(1000,3)-0.5)*2).cuda()
    eikonal_loss_uniform=(model.sdf_net.gradient(uniform_points).squeeze().norm(dim=1)-1).abs().mean()
    sdf_uniform=model.sdf_net(uniform_points)
    
    #Get plane loss
    inds_inside=(sdf_uniform<0).detach()
    inds_below_plane=((uniform_points@plane_eq_normalize[:3].cuda()+plane_eq_normalize[3].cuda())<0).detach()
    loss_plane=-(sdf_uniform[torch.logical_and(inds_below_plane,inds_inside)]).sum()


    
    #Sensor losses
    _,depth_loss,mask_loss = get_sensor_losses(pixels_infra,mean_vertex,scale_vertices,model,opt,1,1.0,'albedo',depth_im_mask,depth_im_mask,distances_im )
    
    loss+=eikonal_loss_uniform*10000+eikonal_loss_surface*10000+loss_point_cloud*100000
    
    loss+=mask_loss*100000

    loss+=depth_loss*100000
    loss+=loss_plane*100000 

    return loss


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):


        
        self.normal_to_plane,self.gt_mesh,self.use_gt_mesh_for_eval,self.angle_azimuth_depth,self.depth_im,self.depth_im_mask,self.pixels_infra, self.plane_eq,self.angle_to_normal,self.angle_to_top_view,self.angle_to_horizontal_view,self.angle_to_bottom_view,self.distances_im,self.data__p,self.mean_vertex,self.scale_vertices,self.norm_homography,self.plane_eq,self.plane_eq_normalize,self.points_on_surface=init_model(opt)


        self.name = name
        self.opt = opt
        self.mute = mute

        self.no_sds=opt.no_sds
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint

        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = None
    
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # guide model
        self.guidance = guidance

        # text prompt
        if self.guidance is not None:
            
            for p in self.guidance.parameters():
                p.requires_grad = False

            self.prepare_text_embeddings()
        
        else:
            self.text_z = None
    
        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if True:#optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)#, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if True:#lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = None#ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        if not self.opt.dir_text:
            self.text_z = self.guidance.get_text_embeds([self.opt.text], [self.opt.negative])
        else:
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                # construct dir-encoded text
                text = f"{self.opt.text}, {d} view"

                negative_text = f"{self.opt.negative}"

                # explicit negativethe dir-encoded text
                if self.opt.suppress_face:
                    if negative_text != '': negative_text += ', '

                    if d == 'back': negative_text += "face"
                    # elif d == 'front': negative_text += ""
                    elif d == 'side': negative_text += "face"
                    elif d == 'overhead': negative_text += "face"
                    elif d == 'bottom': negative_text += "face"
                
                text_z = self.guidance.get_text_embeds([text], [negative_text])
                self.text_z.append(text_z)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    
    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                pass
                
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        # rays_o = data['rays_o'] # [B, N, 3]
        # rays_d = data['rays_d'] # [B, N, 3]

        # B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # TODO: shading is not working right now...
        if self.global_step < self.opt.albedo_iters:
            shading = 'albedo'
            ambient_ratio = 1.0
        else: 
            rand = random.random()
            if rand > 0.8: 
                shading = 'albedo'
                ambient_ratio = 1.0
            elif rand > 0.4: 
                shading = 'textureless'
                ambient_ratio = 0.1
            else: 
                shading = 'lambertian'
                ambient_ratio = 0.1
        angle_rot_add,angle_rotx_add=get_angles(self.epoch,self.angle_to_horizontal_view)

        
            

        scaling=1
        
        #render image for the SDS loss
        pred_rgb,_,_,_,_,rendered_azimuth,rendered_angle_to_normal=render_image(self.opt,
                                                                                self.angle_to_top_view,
                                                                                self.angle_to_bottom_view,
                                                                                self.angle_azimuth_depth,
                                                                                self.angle_to_normal, self.mean_vertex,self.scale_vertices,self.normal_to_plane,self.model,
                                                                                 angle_rot_add,angle_rotx_add=angle_rotx_add,sds_W=80,sds_H=80,scaling=scaling,shading=shading,ambient_ratio=ambient_ratio)
        dirs=get_dir_index(rendered_angle_to_normal,rendered_azimuth)

        
                
            
        if self.opt.dir_text:
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z
        
        if self.no_sds==False:
            loss = self.guidance.train_step(text_z, pred_rgb)
            torch.cuda.empty_cache() 
        else:
            loss=0


        loss+=get_losses(self.opt,self.pixels_infra,self.mean_vertex,self.scale_vertices,self.model,self.points_on_surface,self.plane_eq_normalize,self.depth_im_mask,self.distances_im)

      
        return pred_rgb, 0, loss

   
    def train(self, train_loader, valid_loader, max_epochs):

        assert self.text_z is not None, 'Training must provide a text prompt!'

      
        start_t = time.time()
        # self.evaluate_one_epoch(valid_loader)
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")



    def evaluate(self, loader, name=None):
       
        self.evaluate_one_epoch(loader, name)


   
    
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
        
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_ws, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                        
                
                
                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                   
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def eval_curr(self,data,pbar,loader,name,input_view=False):
        total_loss = 0
        with torch.cuda.amp.autocast(enabled=self.fp16):
            torch.cuda.empty_cache() 
            
            with torch.no_grad():

                if input_view==False:
                    preds,preds_depth,preds_ws,pixels_rays_o2,pixels_rays_d2,rendered_azimuth,rendered_angle_to_normal=render_image(self.opt,
                                                                                self.angle_to_top_view,
                                                                                self.angle_to_bottom_view,
                                                                                self.angle_azimuth_depth,
                                                                                self.angle_to_normal, self.mean_vertex,self.scale_vertices,self.normal_to_plane,self.model,angle_rot_add=60-30.0*self.local_step,sds_W=250, sds_H=250,angle_rotx_add=0.0,white_background=True,scaling=2.0)
                else:
                    preds,preds_depth,preds_ws,pixels_rays_o2,pixels_rays_d2,rendered_azimuth,rendered_angle_to_normal=render_image(self.opt,
                                                                                self.angle_to_top_view,
                                                                                self.angle_to_bottom_view,
                                                                                self.angle_azimuth_depth,
                                                                                self.angle_to_normal, self.mean_vertex,self.scale_vertices,self.normal_to_plane,self.model,angle_rot_add=0.0,sds_W=250, sds_H=250,angle_rotx_add=0,white_background=True,scaling=1.0)
            torch.cuda.empty_cache() 
            print("-------------------")
            print("im %d"%self.local_step)
            print("angle_to_normal %f"%rendered_angle_to_normal)
            print("azimuth %f"%rendered_azimuth)
            
            print("-------------------")
            data['rays_o']=pixels_rays_o2.cuda()
            data['rays_d']=pixels_rays_d2.cuda()
            preds=preds.permute(0,2,3,1)
            preds_depth=preds_depth[:,0]
            preds_ws=preds_ws[:,0]

                
        if self.world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / self.world_size
            
            preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
            dist.all_gather(preds_list, preds)
            preds = torch.cat(preds_list, dim=0)

            preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
            dist.all_gather(preds_depth_list, preds_depth)
            preds_depth = torch.cat(preds_depth_list, dim=0)
        
        loss_val =0# loss.item()
        total_loss += loss_val

        # only rank = 0 will perform evaluation.
        if self.local_rank == 0:

            # save image
            save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
            save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

            #self.log(f"==> Saving validation image to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            pred = preds[0].detach().cpu().numpy()
            pred = (pred * 255).astype(np.uint8)

            pred_depth = preds_depth[0].detach().cpu().numpy()
            pred_depth = (pred_depth * 255).astype(np.uint8)

            cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_path_depth, pred_depth)

            pbar.update(loader.batch_size)
        return total_loss

   
    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            _,output_mesh=get_surface(  self.model.sdf_net, resolution=100)
            os.makedirs(os.path.join(self.workspace, 'validation'), exist_ok=True)
            output_mesh.vertices=(torch.tensor(output_mesh.vertices).float()/self.scale_vertices+self.mean_vertex.unsqueeze(0)).cpu()
            output_mesh.export(os.path.join(self.workspace, 'validation', f'{name}__surface.obj'))
         
            if self.use_gt_mesh_for_eval:
                chamfer_error=calculate_champfer_distance(output_mesh,self.gt_mesh)
                print("chamfer epoch %d"%self.epoch)
                print(chamfer_error)
                




            for data in loader:    
                if self.local_step==0:
                    total_loss=self.eval_curr(data,pbar,loader,name,input_view=True)
                self.local_step += 1
                total_loss=self.eval_curr(data,pbar,loader,name)
                

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
              
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }


        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")