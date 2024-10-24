import json
import os
import torch
import SimpleITK as sitk
import numpy as np
from models.render import angle2vec, get_rays, composite
from tqdm import tqdm

class CBCTDataset(torch.utils.data.Dataset):
    """
    Dataset from CBCT projection
    """
    def __init__(
        self, args, stage="train"
    ):
        """
        :param args
        :param stage train | val | test | visual
        """
        super().__init__()
        self.args = args
        self.stage = stage
        self.angle_sampling = args.angle_sampling
        self.device = args.device
        self.datadir = args.datadir
        dataset_split_json = os.path.join('./data/dataset_split', args.datatype+'_split.json')
        with open(dataset_split_json, 'r') as file:
            json_data = json.load(file)
        dataset_split = json_data[stage]
        self.dataset_split = dataset_split
        print("Loading CBCT dataset", self.datadir, "stage:",self.stage)    
    
    def __len__(self):
        return len(self.dataset_split)
    
    def __getitem__(self, index):

        # load paras
        paras_json = os.path.join(self.datadir, self.dataset_split[index],'transforms.json')
        with open(paras_json, 'r') as file:
            paras = json.load(file)

        # load volume
        volume_path = os.path.join(self.datadir, self.dataset_split[index], "gt_volume.nii.gz")
        volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))
        volume = np.clip(volume, 0, volume.max())
        volume_tensor = torch.tensor(volume, dtype=torch.float32, device=self.device)

        # basic information
        start, end, nviews = self.args.start, self.args.end, self.args.nviews

        if self.angle_sampling == "uniform":
            # load images
            img_path = os.path.join(self.datadir, self.dataset_split[index], 'proj.nii.gz')
            proj = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            proj = np.clip(proj, 0, proj.max())
            angle_per_view = paras['angle_per_view']
            start_index = int(np.round(start/angle_per_view))
            end_index = int(np.round(end/angle_per_view))
            indices = np.linspace(start_index, end_index, nviews, endpoint=False, dtype=int)
            all_imgs = torch.tensor(proj[indices], dtype=torch.float32, device=self.device).unsqueeze(1)

            # load poses
            vecs = []
            for i in indices:
                frame = paras['frames'][i]
                vec = torch.tensor(frame['vec'], dtype=torch.float32, device=self.device)
                vecs.append(vec)
            vecs = torch.stack(vecs)

        elif self.angle_sampling == "random":  
            # we only recommend random sampling during evaluation because X-ray simulation is really slow, 
            # only for dental/spine dataset
            # basic information
            isocenter = [0, 0, 0]
            sad = paras['sad']
            sid = paras['sid']
            proj_spacing = paras['proj_spacing']
            W, H = paras['proj_resolution']
            factor = 0.5
            chunksize = 65536
            volume_phy = torch.tensor(paras['volume_phy']).to(self.device)
            volume_origin = torch.tensor(paras['volume_origin']).to(self.device)
            volume_spacing = torch.min(torch.tensor(paras['volume_spacing'])).to(self.device).to(torch.float32)
            render_step_size = volume_spacing * factor

            # pose generation
            angles = np.random.uniform(start, end, nviews)
            vecs = []
            for angle in tqdm(angles, desc='Projection Geometry Production'):
                angle *= np.pi / 180
                vec = angle2vec(angle, 0, isocenter, sid, sad, proj_spacing[0], proj_spacing[1])
                vec = torch.tensor(vec, dtype=torch.float32, device=self.device)
                vecs.append(vec)
            vecs = torch.stack(vecs)
            cam_rays = get_rays(vecs, H, W)

            # projection generation
            all_imgs = []
            for i in tqdm(range(nviews), desc='Projection Generation'):
                rays = cam_rays[i, ...]
                rays = rays.reshape(-1, rays.shape[-1])
                projection = composite(rays, volume_tensor, volume_origin, volume_phy, render_step_size, chunksize=chunksize)
                projection = projection.reshape(H, W)
                all_imgs.append(projection)
            all_imgs = torch.stack(all_imgs).unsqueeze(1)

        result = {
            "paras": paras,
            "3Dvolume": volume_tensor,
            "images": all_imgs,
            "poses": vecs, 
            "obj_index": paras['obj_index']
        }

        return result
