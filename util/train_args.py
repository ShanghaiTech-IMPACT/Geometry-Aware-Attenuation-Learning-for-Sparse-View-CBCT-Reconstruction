import os
import argparse
from pyhocon import ConfigFactory
import datetime
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-B", type=int, default=1, help="Object batch size | right now we only support 1 batch")
    parser.add_argument("--start", type=int, default=0, help="start scanning angle")  # it is recommended to use integer angle
    parser.add_argument("--end", type=int, default=360, help="end scanning angle")
    parser.add_argument("--nviews", "-V", type=int, default=20, help="Number of selected views",)
    parser.add_argument("--angle_sampling", type=str, default="uniform", help="angle sampling strategy | uniform | random")
    parser.add_argument("--expnorm", action="store_false", help="Whether to use exponential projection normalization") 
    parser.add_argument("--train_scale", type=int, default=4, help="set downsampling scale manually during training stage")
    parser.add_argument("--fusion", type=str, default='ada', help="multi-view feature fusing strategy")
    parser.add_argument("--name", "-n", type=str, default='SVCT_train', help="experiment name")
    parser.add_argument("--logs_path", type=str, default="train/logs", help="logs output directory",)
    parser.add_argument("--checkpoints_path",type=str,default="train/checkpoints",help="checkpoints output directory",)
    parser.add_argument("--visual_path",type=str,default="train/visuals",help="visualization output directory",)
    parser.add_argument("--epochs",type=int,default=500,help="number of epochs to train",)
    parser.add_argument("--datadir", "-D", type=str, default='dataset/dental/syn_data', help="Dataset directory")
    parser.add_argument("--conf", "-c", type=str, default='conf/train.conf', help='Config file')
    parser.add_argument("--device", type=str, default='cuda', help='compute device')
    parser.add_argument("--is_train", action="store_true", help="Training or visualization")
    parser.add_argument("--resume", "-r", action="store_true", help="continue training")
    parser.add_argument("--resume_name", type=str, default=None, help='resume which trained net for continue training')
    parser.add_argument("--datatype", type=str, default="dental", help="data type dental | spine | Walnuts")
    parser.add_argument("--gd1_lambda", type=float, default=1.0, help='weight for gradient loss')
    parser.add_argument("--mse_lambda_2d", type=float, default=0.01, help='weight for projection loss')  

    args = parser.parse_args()

    conf = ConfigFactory.parse_file(args.conf)
    if args.train_scale!=0:
        conf.put("model.SRGAN.generator.scale", args.train_scale)
    conf.put("model.fusion", args.fusion)
    conf.put("train.G_loss.gd1_lambda", args.gd1_lambda)
    conf.put("train.G_loss.mse_lambda_2d", args.mse_lambda_2d)

    now = datetime.datetime.now()
    exp_state_list = [now.strftime('%Y-%m-%d %H:%M:%S'), '\n'
                     'Exp name: ' , args.name , '\n' ,
                     'Training or not: ' , "yes" if args.is_train else "no" , '\n' ,
                     'Resume: ' , "yes" if args.resume else "no" , '\n' ,
                     'resume name: ', str(args.resume_name), '\n',
                     'config file: ' , args.conf , '\n' ,
                     'Dataset: ' , args.datadir , '\n' ,
                     'datatype: ', args.datatype, '\n' ,
                     'start scanning angle: ', str(args.start), '\n',
                     'end scanning angle: ', str(args.end), '\n',
                     'input views: ' , str(args.nviews) , '\n',
                     'angle sampling: ', args.angle_sampling, '\n',
                     'expnorm: ', "yes" if args.expnorm else "no", '\n',
                     'ray_batch_size: ', str(conf['render.ray_batch_size']), '\n',
                     'factor: ', str(conf['render.factor']), '\n' ,
                     'scale: ' , str(conf['model.SRGAN.generator.scale']) , '\n',
                     'fusion: ', str(conf['model.fusion']), '\n',
                     'inplanes: ' , str(conf['model.SRGAN.generator.inplanes']) , '\n' ,
                     'mse_lambda_2d: ', str(conf['train.G_loss.mse_lambda_2d']), '\n',
                     'mse_lambda_3d: ', str(conf['train.G_loss.mse_lambda_3d']), '\n',
                     'gd1_lambda: ', str(conf['train.G_loss.gd1_lambda']), '\n']

    exp_state = ''.join(exp_state_list)
    print(exp_state)
    logs_path = os.path.join(args.logs_path, args.name)
    os.makedirs(logs_path, exist_ok=True)
    f_exp = open(logs_path + '/exp_state.txt', mode='a')
    f_exp.write(exp_state)
    f_exp.close()

    return args,conf
