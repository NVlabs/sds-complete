import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils import seed_everything,Trainer
import numpy as np
import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default="", help="text prompt")

   
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")

    parser.add_argument('--eval_interval', type=int, default=100, help="evaluate on the valid set every interval epochs")
    
    parser.add_argument('--workspace', type=str, default='workspace/')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--depth_path', type=str, default='', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--object_id_number', type=str, default='01184', help='choose from [stable-diffusion, clip]')
    
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=200200, help="training iters")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--max_steps', type=int, default=512, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    # model options
    parser.add_argument('--bg_radius', type=float, default=0, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=10, help="max (center) density for the gaussian density blob")
    parser.add_argument('--blob_radius', type=float, default=0.3, help="control the radius for the gaussian density blob")
    # network backbone
    parser.add_argument('--backbone', type=str, default='grid', choices=['grid', 'vanilla'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam', 'adamw'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.

    
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true', help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    parser.add_argument('--lambda_entropy', type=float, default=1e-4, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=0, help="loss scale for surface smoothness")

    parser.add_argument('--angle_azimuth_depth', type=float, default=0, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_elevation_depth', type=float, default=0, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    



    parser.add_argument('--no_sds', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    
    opt = parser.parse_args()
    opt.depth_path="data_processing/redwood_dataset/depths/%s_segmented.png"%opt.object_id_number

    data_map_txt={"01184":"An outdoor trash can with wheels","06127":"A plant in a large vase","06830":"Children's tricycle with adult's handle" ,"07306":"An office trash can","05452":"An a outside chair","06145":"A one leg square table","05117":"An old chair","06188":"A motorcycle","07136":"A couch","09639":"An executive chair"}
    data_map_azimuth={"01184":-10,"06127":0,"06830":-50 ,"07306":0,"05452":75,"06145":10,"05117":-40,"09639":10,"06188":-40,"07136":90}
    
    
    opt.angle_azimuth_depth=data_map_azimuth[opt.object_id_number]

    opt.text = data_map_txt[opt.object_id_number]
    opt.workspace=opt.workspace+datetime.datetime.utcnow().strftime("%m_%d_%Y__%H_%M_%S_%f")

       

   

    opt.dir_text = True
    opt.backbone = 'vanilla'


    
    from nerf.network import NeRFNetwork
    
    print(opt)

    seed_everything(opt.seed)

    model = NeRFNetwork(opt)

    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
        
    train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()

    from nerf.sd import StableDiffusion
    guidance = StableDiffusion(device, opt.sd_version, opt.hf_key)

    trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, ema_decay=None, fp16=False, lr_scheduler=None, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
    
   
    valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=12).dataloader()

    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    trainer.train(train_loader, valid_loader, max_epoch)