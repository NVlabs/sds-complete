

import imageio
import numpy as np
import trimesh

def get_plane(point_cloud):
    max_inliers=0
    inlier_threshold=20.0
    for i in range(1000):
        inds_cur=np.random.permutation(point_cloud.shape[0])[:3]
        cur_sample_points = point_cloud[inds_cur,:]
        cur_sample_points = np.concatenate((cur_sample_points, np.ones((3, 1))), axis=1)
        plane=(np.linalg.svd(cur_sample_points)[2][-1,:])
        plane = plane / np.linalg.norm(plane[:3])
        inliers=np.sum(np.abs(point_cloud @ plane[:3] + plane[3]) < inlier_threshold)
        
        if inliers>max_inliers:
            best_inlier_inds=np.abs(point_cloud @ plane[:3] + plane[3]) < inlier_threshold
            bestplane=plane
            max_inliers=inliers
            print(max_inliers)
    return best_inlier_inds,bestplane

def main():

    data_map={
        "09639":"0000174-000005772880.png",
    }

    scan_name="09639"

    depth_data=imageio.imread("frames_data_full/%s/depth/%s"%(scan_name,data_map[scan_name]))


    ys,xs=np.where(depth_data>0)
    z=depth_data[ys,xs]
    K=np.array([[525,0,319.5],[0,525,239.5],[0,0,1]])
    K_inv=np.linalg.inv(K)
    point_cloud= (K_inv@np.stack((xs,ys,np.ones_like(xs))))*z[np.newaxis,:]
    point_cloud=point_cloud.T

    trimesh.PointCloud(point_cloud).export("point_cloud_full.obj")
    best_inlier_inds,bestplane=get_plane(point_cloud)
    trimesh.PointCloud(point_cloud[best_inlier_inds]).export("segmented_plane.obj")
    np.save("plane.npy",bestplane)
   
if __name__ == "__main__":
    main()
