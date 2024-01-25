import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.dataloader import SampleDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from backbone.model import simpleMLP,DummyModel
from loss.loss import mse
import time


def save_model(model_save_pth,model, epoch,train_mse,val_mse):
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filename = 'model_{}_epoch_{}_train_mse_{:0.2e}_val_mse_{:0.2e}.pt'.format(current_time,
                                                                                     epoch,
                                                                                     train_mse,
                                                                                     val_mse,)

    filename = os.path.join(model_save_pth,filename)
    torch.save(model.state_dict(), filename)


def train():
    
    debug = False
    use_pretrain = False

    # split train and val sets
    train_ratio = 0.8
    dataset = SampleDataset(root_dir='data',file_name='Admission_Predict_Ver1.1.csv')
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(len(train_dataset)*0.3),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

    #backbone
    backbone = simpleMLP(in_channels=8,
                        hidden_channels=[8,16,32, 64, 128, 64, 8, 1],
                        # norm_layer=nn.BatchNorm1d,
                        dropout=0, inplace=False).cuda()

    # backbone = DummyModel(in_channels=8,out_channels=1).cuda()

    # try read pre-train model
    if use_pretrain:
        weights_pth = 'final.pt'
        try:
            backbone.load_state_dict(torch.load(weights_pth))
        except:
            print(f'No {weights_pth}')

    # set lr,#epoch, optimizer and scheduler
    lr = 1e-3
    optimizer = optim.Adam(
        backbone.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=True)

    num_epoch = 20000
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=5e-5)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    model_save_pth = os.path.join('model_saved', current_time)
    os.mkdir(model_save_pth)
    writer = SummaryWriter(model_save_pth)

    # start training
    backbone.train()
    for epoch in range(num_epoch):
        loss_list = []
        for step, batch in enumerate(train_loader):

            backbone.zero_grad()
            sample, target = batch
            sample, target = sample.cuda(), target.cuda()

            output = backbone(sample)

            loss = mse(output, target)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        scheduler.step()

        if epoch % 100 == 0:
            # print(f'\r Epoch:{epoch} MSE loss = {np.mean(loss_list)} ,lr = {optimizer.param_groups[0]["lr"]}     ', end = ' ')
            writer.add_scalar('Training MSE Loss', np.mean(loss_list), epoch)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]["lr"], epoch)

        # valing and save
        if epoch % 1000 == 0:
            print('Valing.....')
            val_loss_list = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_sample, val_target = val_batch
                    val_sample, val_target = val_sample.cuda(), val_target.cuda()

                    output = backbone(val_sample)
                    val_wmse = mse(output, val_target)

                    val_loss_list.append(val_wmse.item())

            if debug:
                # give some valing example, only for test
                print('############ sample val ##################')
                for i in range(5):
                    print('Pred:',output[i])
                    print('Target:',val_target[i])
                    print()
                print('################################################')

            Train_mse = np.mean(loss_list)
            val_mse = np.mean(val_loss_list)

            writer.add_scalar('Validation MSE', val_wmse, epoch + 1)
            print(f'VAL Epoch:{epoch} Train wmse = {Train_mse}, '
                  f'val wmse = {val_wmse}, ')
            print()
            save_model(model_save_pth,backbone, epoch, Train_mse, val_mse)

    torch.save(backbone.state_dict(), os.path.join(model_save_pth,'final.pt'))
    dummy_input = torch.randn(1, 8, requires_grad=True).cuda()
    torch.onnx.export(backbone,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      os.path.join(model_save_pth,'final.onnx'),  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    writer.flush()
    writer.close()

if __name__ == '__main__':
    train()