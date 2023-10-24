import torch
import torch.nn as nn
import torch.optim as optim
from Model.MIRFuse import MIRFuse
from Loss.MIR_Loss import MIRLoss
# from matplotlib import pyplot as plt
import logging

def MIRFuse_run(
        epochs:int,
        lr:float,
        dataloader_VIS, dataloader_IR, Iter_per_epoch, device_ids,
        device_prime, dir, plt_dir
):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s : %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    model = MIRFuse(channels=1, filter=32, shared_dim=64, exclusive_dim=64)
    device = torch.device(device_prime)
    model.to(device)

    Loss = MIRLoss(local_mutual_loss_coeff=1.0, global_mutual_loss_coeff=0.5, shared_loss_coeff=0.1,
                    window_size=11, gamma=100)

    model.encoder = nn.DataParallel(model.encoder, device_ids=device_ids)
    # model.encoder_ir = nn.DataParallel(model.encoder_ir, device_ids=device_ids)
    model.decoder = nn.DataParallel(model.decoder, device_ids=device_ids)
    model.local_stat_VI = nn.DataParallel(model.local_stat_VI, device_ids=device_ids)
    model.local_stat_IR = nn.DataParallel(model.local_stat_IR, device_ids=device_ids)
    model.global_stat_VI = nn.DataParallel(model.global_stat_VI, device_ids=device_ids)
    model.global_stat_IR = nn.DataParallel(model.global_stat_IR, device_ids=device_ids)
    # model.discriminator_x = nn.DataParallel(model.discriminator_x, device_ids=device_ids)
    # model.discriminator_y = nn.DataParallel(model.discriminator_y, device_ids=device_ids)
    optimizer1 = optim.Adam(model.encoder.parameters(), lr=lr)
    # optimizer2 = optim.Adam(model.encoder_ir.parameters(), lr=lr)
    optimizer3 = optim.Adam(model.decoder.parameters(), lr=lr)
    optimizer4 = optim.Adam(model.local_stat_VI.parameters(), lr=lr)
    optimizer5 = optim.Adam(model.local_stat_IR.parameters(), lr=lr)
    optimizer6 = optim.Adam(model.global_stat_VI.parameters(), lr=lr)
    optimizer7 = optim.Adam(model.global_stat_IR.parameters(), lr=lr)
    # optimizer8 = optim.Adam(model.discriminator_x.parameters(), lr=lr)
    # optimizer9 = optim.Adam(model.discriminator_y.parameters(), lr=lr)

    epochs_list = []
    loss_list = []
    recons_loss_list = []

    for iteration in range(epochs):

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        model.encoder.train()
        # model.encoder_ir.train()
        model.decoder.train()
        model.local_stat_VI.train()
        model.local_stat_IR.train()
        model.global_stat_IR.train()
        model.global_stat_VI.train()
        # model.discriminator_x.train()
        # model.discriminator_y.train()

        data_iter_VIS = iter(dataloader_VIS)
        data_iter_IR = iter(dataloader_IR)

        for step in range(Iter_per_epoch):
            data_VIS= (next(data_iter_VIS))[0]
            data_IR= (next(data_iter_IR))[0]

            data_VIS = data_VIS.to(device)
            data_IR = data_IR.to(device)

            output = model(data_VIS, data_IR)

            loss = Loss(data_VIS, data_IR, output).total_loss
            mim_loss = Loss(data_VIS, data_IR, output).mim_loss
            shared_loss = Loss(data_VIS, data_IR, output).shared_loss
            # mse_loss = Loss(data_VIS, data_IR, output).mse_loss
            het_loss = Loss(data_VIS, data_IR, output).het_loss

            a = a + loss.mean().item()
            b = b + mim_loss.mean().item()
            c = c + shared_loss.mean().item()
            # d = d + mse_loss.mean().item()
            e = e + het_loss.mean().item()


            optimizer1.zero_grad()
            # optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            optimizer5.zero_grad()
            optimizer6.zero_grad()
            optimizer7.zero_grad()
            # optimizer8.zero_grad()
            # optimizer9.zero_grad()

            loss.backward()

            optimizer1.step()
            # optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            optimizer6.step()
            optimizer7.step()
            # optimizer8.step()
            # optimizer9.step()

        logger.info('epochs = %d', iteration + 1)
        logger.info('loss = %.8f', a)
        logger.info('mutual_loss = %.8f', b)
        logger.info('shared_loss = %.8f', c)
        logger.info('reconstruction_loss = %.8f', d)
        logger.info('het_loss = %.8f', e)
        # logger.info('discriminator_loss = %.8f', f)
        logger.info('-----------------------------------')

        epochs_list.append(iteration + 1)
        loss_list.append(a)
        recons_loss_list.append(d)

        torch.save(model.encoder, dir + '/encoder.pth')
        # torch.save(model.encoder_ir, dir + '/encoder_ir.pth')
        torch.save(model.decoder, dir + '/decoder.pth')