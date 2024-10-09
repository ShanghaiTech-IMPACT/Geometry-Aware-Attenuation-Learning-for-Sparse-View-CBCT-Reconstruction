import numpy as np
import SimpleITK as sitk
import os
import json
from models.render import ct2mu, angle2vec, get_rays, composite
from tqdm import tqdm
import argparse
import torch

def GeometryProduction(args, niipath, projpath):
    """
    Projection Geometry Configuration File Production
    """

    start, end, num = args.start, args.end, args.num
    sad, sid = args.sad, args.sid

    # default projection resolution
    proj_resolution = [256, 256]

    os.makedirs(projpath,exist_ok=True)
    path_list = os.listdir(niipath)

    for file_cur in path_list:
        image_path = os.path.join(niipath,file_cur)
        file_name = file_cur[0:-7]
        output_path = os.path.join(projpath,file_name)

        image = sitk.ReadImage(image_path)
        volume_resolution = np.asarray(image.GetSize())
        volume_spacing = np.asarray(image.GetSpacing())
        volume_phy = volume_spacing * (volume_resolution)
        isocenter = np.asarray([0, 0, 0])
        volume_origin = isocenter - volume_phy / 2

        proj_phy = volume_phy * sid / sad
        proj_phy = proj_phy[-2:]

        proj_spacing = volume_spacing * sid / sad
        proj_spacing = proj_spacing[-2:]

        step = (end - start) / num
        angles = np.arange(start, end, step)

        params = {
            'obj_index': file_name,
            'start': start,
            'end': end,
            'angle_per_view': step,
            'N_views': num,
            'sad': sad,
            'sid': sid,
            'volume_resolution': volume_resolution.tolist(),
            'volume_spacing': volume_spacing.tolist(),
            'volume_origin': volume_origin.tolist(),
            'volume_phy': volume_phy.tolist(),
            'proj_resolution': proj_resolution,
            'proj_spacing': proj_spacing.tolist(),
            'proj_phy': proj_phy.tolist(),
        }

        frames = []

        cnt = 0
        for angle in tqdm(angles, desc='Projection Geometry Production'):
            angle *= np.pi / 180  # degree to radian
            vec = angle2vec(angle, isocenter, sid, sad, proj_spacing[0], proj_spacing[1], 'cone_vec')

            frame = {
                'file': str(cnt).zfill(4),
                'vec': vec.tolist(),
            }
            cnt = cnt + 1
            frames.append(frame)

        params['frames'] = frames

        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'transforms.json'), 'w') as f:
            json.dump(params, f, indent=4)
        
        gt_image = sitk.GetArrayFromImage(image)
        gt_image = ct2mu(gt_image)
        gt_image = np.clip(gt_image, 0, gt_image.max())
        gt_image = sitk.GetImageFromArray(gt_image)

        sitk.WriteImage(gt_image, os.path.join(output_path, 'gt_volume.nii.gz'))

        print('Finish geometry production for', file_name)

def ProjectionGeneration(args, projpath):
    """
    DRR Projection Production
    """

    device = 'cuda:0'
    path_list = os.listdir(projpath)
    factor = 0.5 # uniform sampling factor
    chunksize = 65536

    for file_cur in path_list:
        output_path = os.path.join(projpath, file_cur)
        data_path = os.path.join(output_path, 'gt_volume.nii.gz')
        with open(os.path.join(output_path, 'transforms.json')) as f:
            camera_paras = json.load(f)
        W, H = camera_paras['proj_resolution']
        Nframes = camera_paras['N_views']

        volume_phy = torch.tensor(camera_paras['volume_phy']).to(device)
        volume_origin = torch.tensor(camera_paras['volume_origin']).to(device)
        volume_spacing = torch.min(torch.tensor(camera_paras['volume_spacing'])).to(device).to(torch.float32)
        render_step_size = volume_spacing * factor
        volume = sitk.ReadImage(data_path)
        volume_array = sitk.GetArrayFromImage(volume)
        volume_tensor = torch.tensor(volume_array).to(device)
        vecs = []
        for i in range(Nframes):
            frame = camera_paras['frames'][i]
            vec = torch.tensor(frame['vec']).to(device)
            vecs.append(vec)
        vecs = torch.stack(vecs).to(device)
        cam_rays = get_rays(vecs, H, W)

        projs = []
        for i in tqdm(range(Nframes), desc='Projection Generation'):
            frame = camera_paras['frames'][i]
            rays = cam_rays[i, ...]
            rays = rays.reshape(-1, rays.shape[-1])
            projection = composite(rays, volume_tensor, volume_origin, volume_phy, render_step_size, chunksize=chunksize)
            projection = projection.reshape(H, W)
            projs.append(projection)
        projs = torch.stack(projs)
        projs = projs.cpu().detach().numpy()
        projs = sitk.GetImageFromArray(projs)
        sitk.WriteImage(projs, os.path.join(output_path, 'proj.nii.gz'))

        print('Finish projection generation for', file_cur)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Projection Geometry Configuration File Production')

    parser.add_argument('--start', type=int, default=0, help='Start angle')
    parser.add_argument('--end', type=int, default=360, help='End angle')
    parser.add_argument('--num', type=int, default=360, help='Number of angles')
    parser.add_argument('--sad', type=float, default=500, help='Source-to-axis distance (SAD) | 500 for dental, 1000 for spine')
    parser.add_argument('--sid', type=float, default=700, help='Source-to-image distance (SID) | 700 for dental, 1500 for spine')
    parser.add_argument('--datapath', type=str, default='./dataset/dental', help='Path to input NIfTI files')

    args = parser.parse_args()
    niipath = os.path.join(args.datapath, 'raw_volume')
    projpath = os.path.join(args.datapath, 'syn_data')

    # geometry files production
    GeometryProduction(args, niipath, projpath)

    # projection simulation
    ProjectionGeneration(args, projpath)
