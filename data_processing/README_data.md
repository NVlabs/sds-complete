# Point-Cloud Completion with Pretrained Text-to-image Diffusion Models
### [Project Page](https://sds-complete.github.io/) | [Paper](https://arxiv.org/pdf/2306.10533.pdf) 


This repository contains supplementary data for the <a href="http://redwood-data.org/3dscan/">Redwood Dataset </a>. The supplementary data was used for evaluating SDS-Complete on 10 redwood scans. The data contains:
* Object masks for the GT
* Object masks for the input point cloud 
* Transformations for aligning each GT data into the input point cloud's coordinates system. 
* World plane normals for each input point cloud data


## Downloading Redwood scans
Suplementary data is provided for following scans:
* "09639" 
* "05117"       
* "05452"
* "06127"
* "06188"
* "07136"
* "07306" 
* "01184"  
* "06145"   
* "06830"

For each scan it is required to first download the depth (point cloud) data (e.g. for scan "09639"):
* Enter http://redwood-data.org/3dscan/models.html?i=09639
* Download "RGB-D SEQUENCE"
* Alternatively, go into  data_processing/frames_data_full/09639/README.md and download the provided link.
* Extract 09639.zip into data_processing/frames_data_full/09639/
* You should have a folder data_processing/frames_data_full/09639/depth



## Prepare the input and GT data

```
from pathlib import Path
import imageio
import numpy as np
import trimesh


from pathlib import Path
import imageio
import numpy as np
import trimesh
import os
data_map={
    "09639":"0000174-000005772880.png",
    "05117": "0000699-000023291737.png",
    "05452":"0002573-000085825711.png",
    "06127":"0000004-000000100108.png",
    "06188":"0000036-000001167924.png",
    "07136":"0000052-000001701832.png",
    "07306":"0000081-000002669540.png",
    "01184":"0000394-000013114115.png",
    "06145":"0000715-000023825644.png",
    "06830":"0000262-000008709374.png"
}

Path("redwood_dataset/point_clouds").mkdir(parents=True, exist_ok=True)
Path("redwood_dataset/depths").mkdir(parents=True, exist_ok=True)

os.system("cp -r for_inputs/world_planes redwood_dataset/")
#For each scan apply the provided mask on the depth data and extract the point cloud: 
for scan_name in data_map.keys():
    depth_data=imageio.imread("frames_data_full/%s/depth/%s"%(scan_name,data_map[scan_name]))

    mask=np.load("for_inputs/masks/%s.npy"%scan_name)
    
    imageio.imwrite("redwood_dataset/depths/%s.png"%scan_name, depth_data)
    depth_data[mask==False]=0

    imageio.imwrite("redwood_dataset/depths/%s_segmented.png"%scan_name, depth_data)

    ys,xs=np.where(depth_data>0)
    z=depth_data[ys,xs]
    K=np.array([[525,0,319.5],[0,525,239.5],[0,0,1]])
    K_inv=np.linalg.inv(K)
    point_cloud= (K_inv@np.stack((xs,ys,np.ones_like(xs))))*z[np.newaxis,:]
    point_cloud=point_cloud.T
    
    trimesh.PointCloud(point_cloud).export("redwood_dataset/point_clouds/%s.ply"%scan_name)





#For each scan, download the GT data and apply the provided transformation to align it with the input point cloud. Then mask it with the provided mask:
scans=["09639",   "05117",       "05452",       "06127",       "06188",       "07136",       "07306" ,     "01184"  ,     "06145",    "06830",   ]


for scan_name in scans:
    os.system("wget  -O temp.ply http://redwood-data.org/3dscan/data/mesh/%s.ply"%(scan_name))
    trimesh_mesh=trimesh.load("temp.ply", process=False)

    transformation=np.load("for_GT/transformations/%s_transformation.npy"%scan_name)
    trimesh_mesh.vertices=(transformation[:3,:3]@(trimesh_mesh.vertices.T)+transformation[:3,3][:,np.newaxis]).T

    trimesh_mesh.vertices*=1000

    indices=np.load("for_GT/masks/mask_%s.npy"%scan_name)

    
    maskk=np.zeros(trimesh_mesh.vertices.shape[0])
    maskk[indices]=1


    vertices_inds=np.cumsum(maskk>0.5)-1
    trimesh_mesh.faces= trimesh_mesh.faces[maskk[trimesh_mesh.faces].sum(axis=1)==3]
    trimesh_mesh.vertices=trimesh_mesh.vertices[maskk>0.5]
    trimesh_mesh.faces=vertices_inds[trimesh_mesh.faces]
    


    from pathlib import Path
    Path("redwood_dataset/GT").mkdir(parents=True, exist_ok=True)
    
    trimesh_mesh.export("redwood_dataset/GT/%s.ply"%scan_name)

```

If you want to run on your own point cloud files you need to find the world plane. An example code for finding the world pland with RANSAC is provided in find_plane.py. 

You also need to provide the object mask for the input point cloud.




