import torch
from models.model import model
from data.Dataset import CBCTDataset
from util.train_args import parse_args
from trainer import trainer

if __name__ == '__main__':
    args,conf = parse_args()
    device = args.device
    ## dataset
    train_dataset = CBCTDataset(args, stage="train")
    val_dataset = CBCTDataset(args, stage="val")
    test_dataset = CBCTDataset(args, stage="test")
    visual_dataset = CBCTDataset(args, stage="visual")

    ## dataloader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle = True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle = True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle = True,
    )
    visual_data_loader = torch.utils.data.DataLoader(
        visual_dataset,
        batch_size=args.batch_size,
        shuffle = False,
    )

    ## model
    G_render = model(model_conf=conf['model'], device=device,)
    net_trainer = trainer(G_render,train_data_loader,val_data_loader,
    test_data_loader,visual_data_loader,args,conf,device)
    net_trainer.start()
