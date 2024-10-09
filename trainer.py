import os.path
import torch.utils.data
from models.render import *
from util.util_func import *
import datetime
from models.loss import *

class trainer():
    def __init__(self, G_render, train_data_loader, val_data_loader, test_data_loader, visual_data_loader, args,
                 conf, device=None):
        self.G_render = G_render
        self.args = args
        self.conf = conf
        self.device = device

        # dataloader
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.visual_data_loader = visual_data_loader

        # interval
        self.vis_interval = conf.get_int('train.print.vis_interval')
        self.save_interval = conf.get_int('train.print.save_interval')
        self.val_interval = conf.get_int('train.print.val_interval')
        self.test_interval = conf.get_int('train.print.test_interval')

        # loss lambda
        self.mse_lambda_2d = conf.get_float('train.G_loss.mse_lambda_2d')
        self.mse_lambda_3d = conf.get_float('train.G_loss.mse_lambda_3d')
        self.gd1_lambda = conf.get_float('train.G_loss.gd1_lambda')

        # epoch
        self.is_train = args.is_train
        self.num_epochs = args.epochs
        self.resume_name = args.resume_name  # specify the resume epoch
        if not self.is_train:
            self.num_epochs = self.num_epochs + 1

        # render 
        self.ray_batch_size = conf.get_int('render.ray_batch_size')
        self.factor = conf.get_float('render.factor')
        self.chunksize = conf.get_int('render.chunksize')
        
        # others
        self.expnorm = args.expnorm
        # We highly recommend to use expnorm, which results in similar projection intensity range between [0, 1].
        # It is beneficial for encoder to extract features, which usually leads to better performance and faster convergence.
        if args.datatype == 'dental':
            self.clamp_min = conf.get_float('data.dental.clamp_min')
            self.clamp_max = conf.get_float('data.dental.clamp_max')
            self.divide = 1
        if args.datatype == 'spine':
            self.clamp_min = conf.get_float('data.spine.clamp_min')
            self.clamp_max = conf.get_float('data.spine.clamp_max')
            self.divide = 10
        if args.datatype == 'Walnuts':
            self.clamp_min = conf.get_float('data.Walnuts.clamp_min')
            self.clamp_max = conf.get_float('data.Walnuts.clamp_max')
            self.divide = 1
            
        # logs
        self.logs_path = os.path.join(args.logs_path, args.name)
        os.makedirs(self.logs_path, exist_ok=True)
        self.visual_path = os.path.join(args.visual_path, args.name)
        os.makedirs(self.visual_path, exist_ok=True)
        self.checkpoints_path = os.path.join(args.checkpoints_path, args.name)
        os.makedirs(self.checkpoints_path, exist_ok=True)

        # lr scheduler & optimizer
        init_lr = conf.get_float('lr_sche.init_lr')
        step_size = conf.get_float('lr_sche.step_size')
        gamma = conf.get_float('lr_sche.gamma')
        self.G_optim = torch.optim.Adam(self.G_render.parameters(), lr=init_lr, )
        self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.G_optim, step_size=step_size,
                                                              gamma=gamma)

        # loss
        self.mse_loss = torch.nn.L1Loss(reduction='mean')

        # load weights & optimizer & iterator
        self.begin_epochs = 0
        os.makedirs("%s/ckpt_history" % (self.checkpoints_path,), exist_ok=True)
        self.latest_model_path = "%s/ckpt_latest" % (self.checkpoints_path,)
        self.history_model_path = "%s/ckpt_history/ckpt_" % (self.checkpoints_path,)
        if args.resume:
            self.load_ckpt(self.resume_name)
        
    def save_ckpt(self, epoch):
        data = {
            'iter': epoch + 1,
            'G_render': self.G_render.state_dict(),
            'G_optim': self.G_optim.state_dict(),
            'G_lr_scheduler': self.G_lr_scheduler.state_dict(),
        }
        torch.save(data, self.latest_model_path)
        if (epoch % self.save_interval == 0) or epoch == self.num_epochs - 1:
            torch.save(data, self.history_model_path + str(epoch))

    def load_ckpt(self, resume_name=None):
        data = None
        if resume_name is None:
            if os.path.exists(self.latest_model_path):
                data = torch.load(self.latest_model_path, map_location=self.device)
        else:
            if os.path.exists(os.path.join(self.history_model_path, resume_name)):
                data = torch.load(os.path.join(self.history_model_path, resume_name), map_location=self.device)
        if data is not None:
            if 'G_render' in data: self.G_render.load_state_dict(data['G_render'])
            if 'iter' in data: self.begin_epochs = data['iter']
            if 'G_optim' in data: self.G_optim.load_state_dict(data['G_optim'])
            if 'G_lr_scheduler' in data: self.G_lr_scheduler.load_state_dict(data['G_lr_scheduler'])

    def train_step(self, data,):
        device = self.device
        # data loading
        src_images = data["images"].to(device=device).squeeze(0)
        if self.expnorm:
            src_images = torch.exp(-src_images/self.divide)
        src_poses = data["poses"].to(device=device).squeeze(0)
        
        # basic information
        _, _, H, W = src_images.shape
        volume_phy = torch.tensor(data['paras']['volume_phy']).to(device).to(torch.float32)
        volume_origin = torch.tensor(data['paras']['volume_origin']).to(device).to(torch.float32)
        volume_spacing = torch.min(torch.tensor(data['paras']['volume_spacing'])).to(device).to(torch.float32)
        render_step_size = volume_spacing * self.factor
        volume_gt = data['3Dvolume'].to(device=device).squeeze(0).to(torch.float32)
        volume_gt = torch.clamp(volume_gt, self.clamp_min, self.clamp_max)
        volume_resolution = torch.tensor(data['paras']['volume_resolution']).to(device).to(torch.int64)
        
        loss_dict = {}
        
        # 2d projection encoding
        self.G_render.encoder(src_images, src_poses)
        # 3d volume decoding
        volume_predict = predict_3d_volume(model=self.G_render, volume_resolution=volume_resolution,
                                           volume_origin=volume_origin, volume_phy=volume_phy,
                                           scale=self.G_render.decoder.scale, device=device)

        self.G_optim.zero_grad()
        # 3d loss
        mse_loss_3d = self.mse_loss(volume_predict, volume_gt) * self.mse_lambda_3d
        loss_dict['mse_loss_3d'] = round(mse_loss_3d.item(), 8)
        G_loss = mse_loss_3d

        # gd loss
        if self.gd1_lambda > 0:
            gd1_loss = gradient1_loss(volume_gt=volume_gt, volume_predict=volume_predict, loss_func=self.mse_loss) * self.gd1_lambda
            loss_dict['gd1_loss'] = round(gd1_loss.item(), 8)
            G_loss += gd1_loss

        # 2d ray batch loss
        if self.mse_lambda_2d > 0:
            pix_inds = torch.randint(0, self.args.nviews * H * W, (self.ray_batch_size,))
            images_gt_all = src_images.reshape(-1, 1)
            proj_gt = images_gt_all[pix_inds]
            src_rays = get_rays(src_poses, H, W)
            proj_rays = src_rays.view(-1, src_rays.shape[-1])[pix_inds].to(device=device)
            proj_predict = composite(rays=proj_rays, volume=volume_predict, volume_origin=volume_origin,
                                        volume_phy=volume_phy, render_step_size=render_step_size, 
                                        chunksize=self.chunksize).reshape(proj_gt.shape)
            if self.expnorm:
                proj_predict = torch.exp(-proj_predict/self.divide)
            mse_loss_2d = self.mse_loss(proj_predict, proj_gt) * self.mse_lambda_2d
            loss_dict['mse_loss_2d'] = round(mse_loss_2d.item(), 8)
            G_loss += mse_loss_2d

        loss_dict['G_loss'] = round(G_loss.item(), 8)

        # update model
        G_loss.backward()
        self.G_optim.step()

        # first set G_render to eval state, calculate the PSNR, and turn it back to train state
        self.G_render.eval()
        with torch.no_grad():
            self.G_render.encoder(src_images, src_poses)
            volume_predict = predict_3d_volume(model=self.G_render, volume_resolution=volume_resolution,
                                               volume_origin=volume_origin, volume_phy=volume_phy,
                                               scale=self.G_render.decoder.scale, device=device)
            volume_predict_clamp = torch.clamp(volume_predict, self.clamp_min, self.clamp_max)
            # 3d ssim calculation is too slow, so we only calculate psnr
            loss_dict['psnr_3d_clamp'] = round(get_psnr(data_norm(volume_predict_clamp), data_norm(volume_gt)), 8)
        self.G_render.train()
        return loss_dict

    def test_step(self, data):
        device = self.device
        # data loading
        src_images = data["images"].to(device=device).squeeze(0)
        if self.expnorm:
            src_images = torch.exp(-src_images/self.divide)
        src_poses = data["poses"].to(device=device).squeeze(0)
        obj_index = data["obj_index"][0]

        # basic information
        volume_phy = torch.tensor(data['paras']['volume_phy']).to(device).to(torch.float32)
        volume_origin = torch.tensor(data['paras']['volume_origin']).to(device).to(torch.float32)
        volume_gt = data['3Dvolume'].to(device=device).squeeze(0).to(torch.float32)
        volume_gt = torch.clamp(volume_gt, self.clamp_min, self.clamp_max)
        volume_resolution = torch.tensor(data['paras']['volume_resolution']).to(device).to(torch.int64)

        loss_dict = {
            'obj_index': obj_index,
        }

        # 2d projection encoding
        self.G_render.encoder(src_images, src_poses)
        # 3d volume decoding
        volume_predict = predict_3d_volume(model=self.G_render, volume_resolution=volume_resolution,
                                           volume_origin=volume_origin, volume_phy=volume_phy,
                                           scale=self.G_render.decoder.scale, device=device)
        
        # metrics calculation
        volume_predict_clamp = torch.clamp(volume_predict, self.clamp_min, self.clamp_max)
        loss_dict = {}
        loss_dict['psnr_3d_clamp'] = round(get_psnr(data_norm(volume_predict_clamp), data_norm(volume_gt)), 8)
        loss_dict['ssim_3d_clamp'] = round(get_ssim_3d(data_norm(volume_predict_clamp), data_norm(volume_gt), data_range=1), 8)

        return loss_dict

    def vis_step(self, data, epoch=0, ):
        device = self.device
        # data loading
        src_images = data["images"].to(device=device).squeeze(0)
        if self.expnorm:
            src_images = torch.exp(-src_images/self.divide)
        src_poses = data["poses"].to(device=device).squeeze(0)
        obj_index = data["obj_index"][0]

        # basic information
        volume_phy = torch.tensor(data['paras']['volume_phy']).to(device).to(torch.float32)
        volume_origin = torch.tensor(data['paras']['volume_origin']).to(device).to(torch.float32)
        volume_gt = data['3Dvolume'].to(device=device).squeeze(0).to(torch.float32)
        volume_gt = torch.clamp(volume_gt, self.clamp_min, self.clamp_max)
        volume_resolution = torch.tensor(data['paras']['volume_resolution']).to(device).to(torch.int64)

        loss_dict = {
            'obj_index': obj_index,
        }

        # 2d projection encoding
        self.G_render.encoder(src_images, src_poses)
        # 3d volume decoding
        volume_predict = predict_3d_volume(model=self.G_render, volume_resolution=volume_resolution,
                                            volume_origin=volume_origin, volume_phy=volume_phy,
                                            scale=self.G_render.decoder.scale, device=device)

        # alculate metrics with clamped volume for more accurate evaluation
        volume_predict_clamp = torch.clamp(volume_predict, self.clamp_min, self.clamp_max)
        loss_dict['psnr_3d_clamp'] = round(get_psnr(data_norm(volume_predict_clamp), data_norm(volume_gt)), 8)
        loss_dict['ssim_3d_clamp'] = round(get_ssim_3d(data_norm(volume_predict_clamp), data_norm(volume_gt), data_range=1), 8)

        # save the volume prediction
        os.makedirs(os.path.join(self.visual_path, obj_index + '/volume'), exist_ok=True)
        volume_gt_nii = self.visual_path + '/' + obj_index + '/volume/volume_gt.nii.gz'
        volume_predict_nii = self.visual_path + '/' + obj_index + '/volume/volume_' + str(epoch) + '.nii.gz'
        volume_gt_hu = mu2ct(volume_gt)  # convert mu to ct number
        volume_predict_hu = mu2ct(volume_predict)
        tensor2nii(volume_gt_hu, volume_gt_nii)
        tensor2nii(volume_predict_hu, volume_predict_nii) # record original volume rather than clamped volume for analysis convinience
        return loss_dict

    def start(self):
        
        if self.is_train:
            for epoch in range(self.begin_epochs, self.num_epochs):

                now = datetime.datetime.now()
                f_train_lr = open(self.logs_path + '/train_lr.txt', mode='a')
                f_train_lr.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' G_lr:' + str(
                    self.G_optim.param_groups[0]["lr"]) + '\n')
                f_train_lr.close()

                # train with the train dataset
                print('Network Training')
                train_batch = 0
                train_psnr_3d_clamp = 0 
                for train_data in self.train_data_loader:
                    train_losses = self.train_step(train_data)
                    train_loss_str = fmt_loss_str(train_losses)
                    now = datetime.datetime.now()
                    f_train_ls = open(self.logs_path + '/train_ls.txt', mode='a')
                    f_train_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' Batch:' + str(
                        train_batch) + train_loss_str
                                    + " G_lr:" + str(self.G_optim.param_groups[0]["lr"])+'\n')
                    f_train_ls.close()
                    print("*** train:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, "Batch:", train_batch,
                        train_loss_str, "G_lr:", str(self.G_optim.param_groups[0]["lr"]),)
                    train_batch = train_batch + 1

                    # batch psnr
                    train_psnr_3d_clamp = train_psnr_3d_clamp + train_losses['psnr_3d_clamp']

                # epoch psnr
                train_psnr_3d_clamp = train_psnr_3d_clamp / train_batch
                now = datetime.datetime.now()
                f_train_psnr = open(self.logs_path + '/train_metric.txt', mode='a')
                f_train_psnr.write(
                    now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' train_psnr_3d_clamp:' + str(train_psnr_3d_clamp) + '\n')
                f_train_psnr.close()
                print("*** train:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, 'train_psnr_3d_clamp:', str(train_psnr_3d_clamp))

                # network saving
                print("saving network & optimizer")
                self.save_ckpt(epoch)
                
                # validate with the val dataset
                if ((epoch % self.val_interval == 0) and (epoch > 0)) or epoch == self.num_epochs - 1:
                    print('Network validating')
                    val_batch = 0
                    val_psnr_3d_clamp = 0
                    val_ssim_3d_clamp = 0
                    for val_data in self.val_data_loader:
                        self.G_render.eval()
                        with torch.no_grad():
                            val_losses = self.test_step(val_data)
                        self.G_render.train()
                        val_loss_str = fmt_loss_str(val_losses)
                        now = datetime.datetime.now()
                        f_val_ls = open(self.logs_path + '/val_ls.txt', mode='a')
                        f_val_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' Batch:' + str(
                            val_batch) + val_loss_str + '\n')
                        f_val_ls.close()
                        print("*** validate:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, "Batch:", val_batch, val_loss_str,)
                        val_batch = val_batch + 1

                        # batch psnr
                        val_psnr_3d_clamp = val_psnr_3d_clamp + val_losses['psnr_3d_clamp']
                        
                        # batch ssim
                        val_ssim_3d_clamp = val_ssim_3d_clamp + val_losses['ssim_3d_clamp']

                    # epoch psnr
                    val_psnr_3d_clamp = val_psnr_3d_clamp / val_batch
                    # epoch ssim
                    val_ssim_3d_clamp = val_ssim_3d_clamp / val_batch

                    now = datetime.datetime.now()
                    f_val_psnr = open(self.logs_path + '/val_metric.txt', mode='a')
                    f_val_psnr.write(
                        now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' val_psnr_3d_clamp:' + str(val_psnr_3d_clamp) + 
                        ' val_ssim_3d_clamp:' + str(val_ssim_3d_clamp) +  '\n')
                    f_val_psnr.close()
                    print("*** validate:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, 'val_psnr_3d_clamp:', str(val_psnr_3d_clamp), 
                          'val_ssim_3d_clamp:'+ str(val_ssim_3d_clamp) + '\n') 

                # test with the test dataset
                if ((epoch % self.test_interval == 0) and (epoch > 0)) or epoch == self.num_epochs - 1:
                    print('Network Testing')
                    test_batch = 0
                    test_psnr_3d_clamp = 0
                    test_ssim_3d_clamp = 0
                    for test_data in self.test_data_loader:
                        self.G_render.eval()
                        with torch.no_grad():
                            test_losses = self.test_step(test_data)
                        self.G_render.train()
                        test_loss_str = fmt_loss_str(test_losses)
                        now = datetime.datetime.now()
                        f_test_ls = open(self.logs_path + '/test_ls.txt', mode='a')
                        f_test_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' Batch:' + str(
                            test_batch) + test_loss_str + '\n')
                        f_test_ls.close()
                        print("*** test:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, "Batch:", test_batch, test_loss_str)
                        test_batch = test_batch + 1

                        # batch psnr
                        test_psnr_3d_clamp = test_psnr_3d_clamp + test_losses['psnr_3d_clamp']

                        # batch ssim
                        test_ssim_3d_clamp = test_ssim_3d_clamp + test_losses['ssim_3d_clamp']

                    # epoch psnr
                    test_psnr_3d_clamp = test_psnr_3d_clamp / test_batch
                    # epoch ssim
                    test_ssim_3d_clamp = test_ssim_3d_clamp / test_batch

                    now = datetime.datetime.now()
                    f_test_psnr = open(self.logs_path + '/test_metric.txt', mode='a')
                    f_test_psnr.write(
                        now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' test_psnr_3d_clamp:' + str(test_psnr_3d_clamp) + 
                        ' test_ssim_3d_clamp:'+ str(test_ssim_3d_clamp) + '\n')
                    f_test_psnr.close()
                    print("*** test:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, 'test_psnr_3d_clamp:', str(test_psnr_3d_clamp), 
                    'test_ssim_3d_clamp:', str(test_ssim_3d_clamp), '\n')

                # lr schedule
                self.G_lr_scheduler.step()

                # visualization with the visual dataset during training when meet the epoch condition
                if ((epoch % self.vis_interval == 0) and (epoch > 0)) or epoch == self.num_epochs - 1:
                    for vis_data in self.visual_data_loader:
                        print("Generating visualization")
                        self.G_render.eval()
                        with torch.no_grad():
                            vis_losses = self.vis_step(vis_data, epoch=epoch, )
                        self.G_render.train()
                        vis_loss_str = fmt_loss_str(vis_losses)
                        now = datetime.datetime.now()
                        f_vis_psnr = open(self.logs_path + '/visual_metric.txt', mode='a')
                        f_vis_psnr.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + vis_loss_str + '\n')
                        f_vis_psnr.close()
                        print("*** visual:", now.strftime('%Y-%m-%d %H:%M:%S'), " Epoch:", epoch, vis_loss_str)
        
        # visualization when not training (must resume some trained net)
        else:
            epoch = self.begin_epochs
            for vis_data in self.visual_data_loader:
                print("Generating visualization")
                self.G_render.eval()
                with torch.no_grad():
                    vis_losses = self.vis_step(vis_data, epoch=epoch)
                self.G_render.train()
                vis_loss_str = fmt_loss_str(vis_losses)
                now = datetime.datetime.now()
                f_vis_psnr = open(self.logs_path + '/visual_metric.txt', mode='a')
                f_vis_psnr.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' visualization:' + vis_loss_str + '\n')
                f_vis_psnr.close()
                print("*** visual:", now.strftime('%Y-%m-%d %H:%M:%S'), vis_loss_str)
