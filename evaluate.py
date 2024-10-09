import os
from data.Dataset import CBCTDataset
from util.evaluate_args import parse_args
import datetime
from models.model import model
import torch.utils.data
from models.render import *
from util.util_func import *

def test_step():
    # data loading
    src_images = data["images"].to(device=device).squeeze(0)
    if args.expnorm:
        src_images = torch.exp(-src_images/divide) 
    src_poses = data["poses"].to(device=device).squeeze(0)

    # basic information
    volume_phy = torch.tensor(data['paras']['volume_phy']).to(device).to(torch.float32)
    volume_origin = torch.tensor(data['paras']['volume_origin']).to(device).to(torch.float32)
    volume_gt = data['3Dvolume'].to(device=device).squeeze(0).to(torch.float32)  # GT volume of target object
    volume_gt = torch.clamp(volume_gt, clamp_min, clamp_max)
    volume_resolution = torch.tensor(data['paras']['volume_resolution']).to(device).to(torch.int64)

    start_time = datetime.datetime.now()
    # 2d projection encoding
    G_render.encoder(src_images, src_poses)
    # 3d volume decoding
    volume_predict = predict_3d_volume(model=G_render, volume_resolution=volume_resolution,
                                        volume_origin=volume_origin, volume_phy=volume_phy,
                                        scale=args.eval_scale, device=device)
    volume_predict_clamp = torch.clamp(volume_predict, clamp_min, clamp_max)
    end_time = datetime.datetime.now()
    elapse_time = (end_time - start_time).total_seconds()

    # record the reconstructed volume
    volume_gt_nii = visual_path + '/' + obj_index + '/volume/volume_gt.nii.gz'
    volume_predict_nii = visual_path + '/' + obj_index + '/volume/volume_predict.nii.gz'
    volume_gt_hu = mu2ct(volume_gt)  # convert mu to ct number
    volume_predict_hu = mu2ct(volume_predict)
    tensor2nii(volume_gt_hu, volume_gt_nii)
    tensor2nii(volume_predict_hu, volume_predict_nii) # record original volume rather than clamped volume for analysis convinience

    loss_dict = {'obj_index':obj_index}
    loss_dict['elapse_time'] = elapse_time
    # calculate metrics with clamped volume for more accurate evaluation
    loss_dict['psnr_3d_clamp'] = round(get_psnr(data_norm(volume_predict_clamp), data_norm(volume_gt)), 8)
    loss_dict['ssim_3d_clamp'] = round(get_ssim_3d(data_norm(volume_predict_clamp), data_norm(volume_gt), data_range=1), 8)
    return loss_dict

def fmt_loss_str(losses):
    return (" " + " ".join(k + ":" + str(losses[k]) for k in losses))

if __name__ == '__main__':
    args, conf = parse_args()
    device = args.device

    # logs
    # you can change uniform/random angle sampling, angle samping range, input views, scale (reconstruction resolution)
    prefix = args.angle_sampling + '_start_' + str(args.start) + '_end_' + str(args.end) + '_nviews_' + str(args.nviews) + '_scale_' + str(args.eval_scale)
    logs_path = os.path.join(args.logs_path, args.name, prefix)
    os.makedirs(logs_path, exist_ok=True)
    visual_path = os.path.join(args.visual_path, args.name, prefix)
    os.makedirs(visual_path, exist_ok=True)
    checkpoints_path = os.path.join(args.checkpoints_path, args.name)
    os.makedirs(checkpoints_path, exist_ok=True)

    # model
    G_render = model(model_conf=conf['model'], device=device)
    if args.resume_name is not None:
        model_path = os.path.join(checkpoints_path, 'ckpt_history', 'ckpt_'+args.resume_name)
        data = torch.load(model_path, map_location=device)
        G_render.load_state_dict(data['G_render'])

    # dataset & dataloader
    evaluate_dataset = CBCTDataset(args, args.dataname)
    evaluate_data_loader = torch.utils.data.DataLoader(
        evaluate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # clamp
    if args.datatype == 'dental':
        clamp_min = conf.get_float('data.dental.clamp_min')
        clamp_max = conf.get_float('data.dental.clamp_max')
        divide = 1
    if args.datatype == 'spine':
        clamp_min = conf.get_float('data.spine.clamp_min')
        clamp_max = conf.get_float('data.spine.clamp_max')
        divide = 10
    if args.datatype == 'Walnuts':
        clamp_min = conf.get_float('data.Walnuts.clamp_min')
        clamp_max = conf.get_float('data.Walnuts.clamp_max')
        divide = 1

    # evaluate
    G_render.eval()

    psnr_3d_clamp_list = []
    ssim_3d_clamp_list = []
    with torch.no_grad():
        for data in evaluate_data_loader:
            obj_index = data["obj_index"][0]
            os.makedirs(os.path.join(visual_path, obj_index + '/volume'), exist_ok=True)
            test_losses = test_step()
            test_loss_str = fmt_loss_str(test_losses)
            now = datetime.datetime.now()
            print("*** Evaluate:", now.strftime('%Y-%m-%d %H:%M:%S'), test_loss_str,)
            # batch logs
            psnr_3d_clamp_list.append(test_losses['psnr_3d_clamp'])
            ssim_3d_clamp_list.append(test_losses['ssim_3d_clamp'])          
            f_test_psnr_batch = open(logs_path + '/metric_batch.txt', mode='a')
            f_test_psnr_batch.write(now.strftime('%Y-%m-%d %H:%M:%S') + test_loss_str + '\n')
            f_test_psnr_batch.close()
        # avg logs
        avg_dict = {}
        avg_dict['psnr_3d_clamp_mean'] = np.mean(psnr_3d_clamp_list)
        avg_dict['psnr_3d_clamp_std'] = np.std(psnr_3d_clamp_list)
        avg_dict['ssim_3d_clamp_mean'] = np.mean(ssim_3d_clamp_list)
        avg_dict['ssim_3d_clamp_std'] = np.std(ssim_3d_clamp_list)
        avg_dict_str = fmt_loss_str(avg_dict)

        now = datetime.datetime.now()
        f_test_psnr = open(logs_path + '/logs_avg.txt', mode='a')
        f_test_psnr.write(now.strftime('%Y-%m-%d %H:%M:%S') +  avg_dict_str + '\n')
        f_test_psnr.close()
        print("*** Evaluate:", now.strftime('%Y-%m-%d %H:%M:%S'), avg_dict_str,)
