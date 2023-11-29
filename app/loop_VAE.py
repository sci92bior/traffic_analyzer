import torch
import torch.distributions as td
import torch.nn.functional as F
import torch.optim.lr_scheduler as scheduler
from torch import optim
from torch.utils.data import DataLoader

from app.models import VAE
from app.utils.data import PacketDataSet
from app.utils.other import FindThreshold, Logger


def calculate_loss(data, reconstructed, mu, logvar, z, alfa, beta, device):
    alfa = alfa.unsqueeze(1).expand(-1, data.shape[1])
    std_enc = torch.exp(0.5 * logvar)

    # dist z|x
    q_zx = td.Normal(loc=mu, scale=std_enc)

    # Find q(z|x)
    log_QzGx = q_zx.log_prob(z)

    # Find p(z)
    mu_prior = torch.zeros(z.shape).to(device)
    std_prior = torch.ones(z.shape).to(device)
    p_z = td.Normal(loc=mu_prior, scale=std_prior)
    log_Pz = p_z.log_prob(z)

    # Find p(x|z)
    recons_loss = F.mse_loss(reconstructed[alfa != 0], data[alfa != 0], reduction="sum")

    kl = log_QzGx - log_Pz * beta
    kl = torch.mean(kl.sum(-1))

    loss = kl + recons_loss
    return loss, recons_loss, kl


def calculate_test_loss(data, reconstructede):
    recons_loss = F.mse_loss(reconstructede, data, reduction="none")

    return recons_loss.mean(-1)


def train_epoch(args,
                encoder: 'Encoder model',
                decoder: 'Decoder model',
                opt_ae: optim,
                scheduler_opt_ae: scheduler,
                train_loader: DataLoader,
                epoch: int,
                scaler: torch.cuda.amp.GradScaler,
                logger,
                device: torch.device):
    encoder.train()
    decoder.train()
    data_to_log = {"Graph": {}, "Print": {}, "Table": {}, "Conf": {}}
    for iteration, data in enumerate(train_loader):
        b,c,g = data.shape
        data[:,:, -1][data[:,:, -1] > 0] = 1  # 0 - normal, anormal as 1
        y = 1 - data[:,4, -1].to(device)  # 1 normal, anormal as 0
        beta = torch.mean(y)
        data = data[:,:, :-1].reshape(b,-1).to(device)

        opt_ae.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            mu, logvar, z = encoder(data)
            reconstructed = decoder(z)
            loss, recon, kl = calculate_loss(data, reconstructed, mu, logvar, z, y, beta, device)

        scaler.scale(loss).backward()
        scaler.step(opt_ae)
        scaler.update()

        if scheduler_opt_ae:
            scheduler_opt_ae.step()
            data_to_log["Print"]["Learning_rate"] = scheduler_opt_ae.get_last_lr()
        data_to_log["Print"]["Train_loss"] = loss.item()
        data_to_log["Graph"]["Train loss"] = loss.item()
        data_to_log["Print"]["Train_recon_loss"] = recon.item()
        data_to_log["Graph"]["Train Train_recon_loss"] = recon.item()
        data_to_log["Print"]["Train_kl_loss"] = kl.item()
        data_to_log["Graph"]["Train_kl_loss"] = kl.item()
        logger.step(data_to_log, epoch, iteration, len(train_loader))


def test_epoch(args,
               encoder: 'Encoder model',
               decoder: 'Decoder model',
               test_loader: DataLoader,
               epoch: int,
               logger,
               device: torch.device):
    encoder.eval()
    decoder.eval()
    data_to_log = {"Graph": {}, "Print": {}, "Table": {}, "Conf": {}}
    total_loss_normal = 0
    iter_normal = 0
    total_loss_anormal = 0
    iter_anormal = 0
    ThresholdCompute = FindThreshold(args, device)
    with torch.no_grad():
        for iteration, data in enumerate(test_loader):
            b, c, g = data.shape
            #data[:, :, -1][data[:, :, -1] > 0] = 1  # 0 - normal, anormal as 1
            #y = 1 - data[:, 2, -1].to(device)  # 1 normal, anormal as 0
            #beta = torch.mean(y)
            #data = data[:, :, :-1].reshape(b, -1).to(device)

            b, c, g = data.shape
            label = data[:,4, -1].to(device)
            data = data[:,:, :-1].reshape(b, -1).to(device)

            mu, logvar, z = encoder(data)
            reconstructed = decoder(z)
            rcloss = calculate_test_loss(data, reconstructed)
            #print(rcloss)
            loss_normal = rcloss[label[:] == 0]
            iter_normal += len(loss_normal)
            total_loss_normal += loss_normal.sum()
            loss_anormal = rcloss[label[:] != 0]
            iter_anormal += len(loss_anormal)
            total_loss_anormal += loss_anormal.sum()

            ThresholdCompute.append_data(label, rcloss)

        threshold = ThresholdCompute.Find_Optimal_Cutoff()
        y_true, preds, conf = ThresholdCompute.Confusion_Matrix(threshold)
        labels = ["Normal - Negative", "Anormal - Positive"]
        data_to_log["Conf"][f"Conf_Matrix_{epoch}"] = [y_true, preds, labels]
        sensitivity = conf[1][1] / (conf[1][1] + conf[1][0])
        specifity = conf[0][0] / (conf[0][0] + conf[0][1])
        data_to_log["Print"]["Sensitivity"] = sensitivity
        data_to_log["Graph"]["Sensitivity"] = sensitivity
        data_to_log["Print"]["Specificity"] = specifity
        data_to_log["Graph"]["Specificity"] = specifity
        data_to_log["Graph"]["Best Threshold"] = threshold[0]

    logger.step(data_to_log, epoch, iteration, len(test_loader))
    return (sensitivity + specifity) / 2


def train_init_VAE(args, config, device):
    batch_size = config["Training"]["batch_size"]
    max_epoch = config["Training"]["max_epoch"]
    encoder = VAE.Encoder(**config["VAE"]["Model"])
    decoder = VAE.Decoder(**config["VAE"]["Model"])
    opt_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(config["Training"]["lr"]),
                        betas=(0.9, 0.999), weight_decay=1e-4)
    if args.scheduler:
        scheduler_opt_ae = scheduler.StepLR(opt_ae, step_size=config["Training"]["scheduler_step"],
                                            gamma=config["Training"]["scheduler_gamma"])
    else:
        scheduler_opt_ae = None
    if args.pretrained:
        try:
            encoder.load_state_dict(
                torch.load(config["VAE"]["Pretrained_Enc_Path"], map_location=torch.device('cpu')), strict=True)
            decoder.load_state_dict(
                torch.load(config["VAE"]["Pretrained_Dec_Path"], map_location=torch.device('cpu')), strict=True)

        except FileNotFoundError as e:
            print("Not found pretrained weights. Continue without any pretrained weights.")

    encoder.to(device)
    decoder.to(device)

    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    encoder = DataParallel(encoder)
    #    decoder = DataParallel(decoder)

    train_set = PacketDataSet([args.train_normal_dataset_path, args.train_anormal_dataset_path], train = True)
    test_set = PacketDataSet([args.train_normal_dataset_path, args.train_anormal_dataset_path], train = False)
    #test_set = PacketDataSet([args.test_normal_dataset_path, args.test_anormal_dataset_path])
    logger = Logger(args, train_set.get_columns())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0,
                             drop_last=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    temp_performance = 0
    last_save = 0
    for epoch in range(0, max_epoch):
        train_epoch(args,
                    encoder,
                    decoder,
                    opt_ae,
                    scheduler_opt_ae,
                    train_loader,
                    epoch,
                    scaler,
                    logger,
                    device)
        performance = test_epoch(args,
                                 encoder,
                                 decoder,
                                 test_loader,
                                 epoch,
                                 logger,
                                 device)
        if performance > temp_performance:
            temp_performance = performance
            last_save = epoch
            torch.save(encoder.state_dict(),
                       f'./current_models_{args.run_name}/Encoder_' + str(epoch) + '_' + f"{epoch:06}" + '.pth')
            torch.save(encoder.state_dict(), f'./saved_models_{args.run_name}/Encoder.pth')

            torch.save(decoder.state_dict(),
                       f'./current_models_{args.run_name}/Decoder' + str(epoch) + '_' + f"{epoch:06}" + '.pth')
            torch.save(decoder.state_dict(), f'./saved_models_{args.run_name}/Decoder.pth')
        if epoch > last_save + 100:
            print("No progress since 10 epochs - stopped training on epoch " + str(epoch))
            print("Best (sensitivity + specifity) / 2 = " + str(temp_performance))
            break
