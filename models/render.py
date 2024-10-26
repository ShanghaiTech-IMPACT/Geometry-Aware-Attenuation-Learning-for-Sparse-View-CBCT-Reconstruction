import numpy as np
import torch.nn.functional as F
import torch

def angle2vec(PrimaryAngle, SecondaryAngle, isocenter, sid, sad, proj_spacing_x, proj_spacing_y):
    # input: PrimaryAngle, SecondaryAngle in rad
    # output: vec [12]

    cam_x = isocenter[0] + sad * np.cos(SecondaryAngle) * np.cos(PrimaryAngle)
    cam_y = isocenter[1] + sad * np.cos(SecondaryAngle) * np.sin(PrimaryAngle)
    cam_z = isocenter[2] + sad * np.sin(SecondaryAngle)
    cam = np.array([cam_x, cam_y, cam_z])

    det_x = isocenter[0] - (sid-sad) * np.cos(SecondaryAngle) * np.cos(PrimaryAngle)
    det_y = isocenter[1] - (sid-sad) * np.cos(SecondaryAngle) * np.sin(PrimaryAngle)
    det_z = isocenter[2] - (sid-sad) * np.sin(SecondaryAngle)
    det = np.array([det_x, det_y, det_z])

    u_x = -proj_spacing_x * np.sin(PrimaryAngle)
    u_y = proj_spacing_x  * np.cos(PrimaryAngle)
    u_z = 0
    
    v_x = proj_spacing_y * np.sin(SecondaryAngle) * np.cos(PrimaryAngle)
    v_y = proj_spacing_y * np.sin(SecondaryAngle) * np.sin(PrimaryAngle)
    v_z = -proj_spacing_y * np.cos(SecondaryAngle)  # opposite direction for display purpose, not necessary

    u_vector = np.array([u_x, u_y, u_z])
    v_vector = np.array([v_x, v_y, v_z])

    vec = np.concatenate([cam, det, u_vector, v_vector])
    return vec

def get_pixel00_center(detectors, uvectors, vvectors, H, W):
    ## detectors, uvectors, vectors, [N, 3]
    ## pixel00_center, [N, 3]
    device = detectors.device
    float_H = torch.tensor(H).to(device)
    float_W = torch.tensor(W).to(device)
    v_offset = torch.floor(float_H/2) + torch.floor((float_H+1)/2) - (float_H+1)/2
    u_offset = torch.floor(float_W/2) + torch.floor((float_W+1)/2) - (float_W+1)/2
    pixel00_center = detectors - u_offset * uvectors - v_offset * vvectors
    return pixel00_center

def get_rays(vecs, H, W):
    '''
    :param vecs: [N, 12]
    :return rays: [N, H, W, 6]
    '''
    # we only support cone-beam ray on flat panel now.
    device = vecs.device
    N = vecs.shape[0]
    sources, detectors, uvectors, vvectors = vecs[:, :3], vecs[:, 3:6], vecs[:, 6:9], vecs[:, 9:]    # (N, 3)

    pixel00_center = get_pixel00_center(detectors, uvectors, vvectors, H, W)   # (N, 3)
    row_indices, col_indices = torch.meshgrid(torch.arange(H, device=device),    # [0, H - 1], [0, W - 1]
                                              torch.arange(W, device=device),
                                              indexing='ij')
    row_indices = row_indices.expand(N, -1, -1).unsqueeze(-1)
    col_indices = col_indices.expand(N, -1, -1).unsqueeze(-1)
    pix_coords = pixel00_center[:, None, None, :] + col_indices * uvectors[:, None, None, :] + row_indices * vvectors[:, None, None, :]

    rays_origin = sources.view(N, 1, 1, 3).expand(-1, H, W, -1)
    rays_dirs = pix_coords - rays_origin
    rays_dirs = rays_dirs / torch.linalg.norm(rays_dirs, dim=-1, keepdim=True)
    rays = torch.cat((rays_origin, rays_dirs), dim=3) 
    return rays
    
def sample_volume_interval(rays, volume_origin, volume_phy, render_step_size,):
    # sample interval (flatten)
    device = rays.device
    near, far = ray_AABB(rays, volume_origin, volume_phy)

    dis = far - near
    _, index = torch.sort(dis, dim=0)
    near_ = near[index[-1]]
    far_ = far[index[-1]]
    max_dis = far_ - near_
    N_sample = int(max_dis / render_step_size)

    t_step_start = torch.linspace(0, 1-1/N_sample, N_sample, device=device)
    t_step_end = torch.linspace(1/N_sample, 1, N_sample, device=device)
    
    t_start = near + max_dis * t_step_start
    t_end = near + max_dis * t_step_end
    mask = outer_mask((t_start + t_end)/2, near, far)
    t_start = t_start[mask==1]
    t_end = t_end[mask==1]
    ray_indices = torch.where(mask==1)[0]
    return t_start, t_end, ray_indices   

def ray_AABB(rays, volume_origin, volume_phy,):
    device = rays.device
    cam_centers = rays[:, 0:3]
    cam_raydir = rays[:, 3:6]
    xyz_max = volume_phy + volume_origin
    xyz_min = volume_origin
    
    eps = torch.tensor(1e-6, dtype=torch.float32, device=device)
    vx = torch.where(cam_raydir[:,0]==0, eps, cam_raydir[:,0])
    vy = torch.where(cam_raydir[:,1]==0, eps, cam_raydir[:,1])
    vz = torch.where(cam_raydir[:,2]==0, eps, cam_raydir[:,2])
    
    ax = (xyz_max[0] - cam_centers[:,0]) / vx
    ay = (xyz_max[1] - cam_centers[:,1]) / vy
    az = (xyz_max[2] - cam_centers[:,2]) / vz
    bx = (xyz_min[0] - cam_centers[:,0]) / vx
    by = (xyz_min[1] - cam_centers[:,1]) / vy
    bz = (xyz_min[2] - cam_centers[:,2]) / vz

    t_min = torch.max(torch.max(torch.min(ax, bx), torch.min(ay, by)), torch.min(az, bz))
    t_max = torch.min(torch.min(torch.max(ax, bx), torch.max(ay, by)), torch.max(az, bz))
    
    return t_min.unsqueeze(1), t_max.unsqueeze(1)

def outer_mask(z_samp,near,far):
    zero_mask_1 = (z_samp>=near)
    zero_mask_2 = (z_samp<=far)
    zero_mask = zero_mask_1 * zero_mask_2 + 0
    return zero_mask

def mu2ct(pix):
    mu_water = 0.022
    ct = (pix/mu_water-1)*1000
    return ct

def ct2mu(pix):
    mu_water = 0.022
    mu = (pix / 1000 + 1) * mu_water
    return mu

def volume_sampling(xyz, volume, volume_origin, volume_phy):
    xyz = ((xyz - volume_origin) / volume_phy) * 2 - 1 # normalize
    volume = volume.unsqueeze(0).unsqueeze(0).to(torch.float32)
    xyz = xyz.unsqueeze(0)
    xyz = xyz.unsqueeze(2)
    xyz = xyz.unsqueeze(2)
    samples = F.grid_sample(
        volume,
        xyz,
        align_corners=True,
        padding_mode="zeros",
    )
    return samples[0,0,...]

def if_intersect(rays, volume_origin, volume_phy):
    device = rays.device
    near, far = ray_AABB(rays, volume_origin, volume_phy)
    dis = far - near
    _, index = torch.sort(dis, dim=0)
    near_ = near[index[-1]]
    far_ = far[index[-1]]
    max_dis = far_ - near_
    if max_dis <= 0:
        return 0
    else:
        return 1

def composite(rays, volume, volume_origin, volume_phy, render_step_size, chunksize=65536):
    split_rays = torch.split(rays, chunksize)
    proj_batch = None
    pred_proj = []
    for ray_batch in split_rays:
        # sample interval and composite
        if_inter = if_intersect(ray_batch, volume_origin, volume_phy)
        if if_inter: 
            proj_batch = composite_batch(ray_batch, volume, volume_origin, volume_phy, render_step_size)
            pred_proj.append(proj_batch)
        else:
            zero_tensor = torch.zeros_like(ray_batch[:, :1])
            pred_proj.append(zero_tensor)
    pred_proj = torch.cat(pred_proj, dim=0)
    return pred_proj

def composite_batch(rays, volume, volume_origin, volume_phy, render_step_size):
    t_start, t_end, ray_indices = sample_volume_interval(rays, volume_origin, volume_phy, render_step_size)
    rays_origins = rays[:, :3]
    rays_dirs = rays[:, 3:6]
    t_origins = rays_origins[ray_indices]  
    t_dirs = rays_dirs[ray_indices]
    points = t_origins + t_dirs * (t_start + t_end)[:, None] / 2.0  # [n_samples, 3]
    split_points = torch.split(points, 100000)
    val_all = []
    for pnts in split_points:
       #  volume sampling
       val_all.append(volume_sampling(pnts, volume, volume_origin, volume_phy))
    att = torch.cat(val_all,dim=0).squeeze(-1)  # [n_samples, 1]  sample values
    delta = t_end - t_start   # [n_samples,]  weight for each sample value 
    proj = volumetric_rendering_along_rays(delta, att, ray_indices, rays_origins.shape[0])  # [n_rays, 1]
    return proj

def volumetric_rendering_along_rays(weights, values, ray_indices, n_rays):
    if values is None:
        src = weights[..., None]
    else:
        src = weights[..., None] * values  # [n_samples, 1]
    outputs = torch.zeros((n_rays, src.shape[-1]), device=src.device, dtype=src.dtype)  # [n_rays, 1]
    outputs.index_add_(0, ray_indices, src)
    return outputs

def make_coords(volume_resolution, volume_phy, volume_origin, device):
    n1, n2, n3 = volume_resolution
    s1, s2, s3 = volume_phy
    o1, o2, o3 = volume_origin
    
    x = torch.linspace(o1, s1 + o1, n1, device=device)
    y = torch.linspace(o2, s2 + o2, n2, device=device)
    z = torch.linspace(o3, s3 + o3, n3, device=device)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).to(torch.float32)
    return xyz

def predict_3d_volume(model,volume_resolution,volume_origin,volume_phy,scale,device):
    xyz = make_coords(volume_resolution.tolist(), volume_phy.tolist(), volume_origin.tolist(), device)
    xyz_sample = xyz[::scale, ::scale, ::scale, :]
    volume_predict = model(xyz_sample)
    return volume_predict
