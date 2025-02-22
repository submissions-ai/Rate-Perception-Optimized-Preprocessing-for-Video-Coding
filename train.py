import os, sys, time, random
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import TrainDataset, TrainDataset_real, TrainDataset_real_paired, ValDataset, ValDataset_real
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from model.RFDN import RFDN
from model.preprocess_arch import UNetSN, UNet, UNetPS
from model.preprocess_rfdn import RFLite_v2, RFLite_v3
from model.loss import RFDNLoss
from model.loss_ssim import SSIMLoss
from model.loss_dct import LowRankLoss, LowRankLoss4, LowRankLoss16, UpperBandEntropyLoss, UpperBandEntropyLoss16
from utils import utils_image, degradations, diffjpeg
from utils.misc import *
from config import load_config, DEFAULT_CONFIG, DEGRADATION_CONFIG
from torch.utils.tensorboard import SummaryWriter

### python -m torch.distributed.launch --nproc_per_node=2 --master_port=45678 train_pre.py -c configs/default.yaml

class Trainer:
    def __init__(self, rank, world_size, config, degradation_config):
        self.config = config
        self.degradation_config = degradation_config

        self.init_distributed(rank, world_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_datasets()
        # self.init_writer()
        self.train(self.rank, self.world_size)
        self.cleanup()
    
    # def init_writer(self):
    #     if self.rank == 0:
    #         # self.log('Initializing writer')
    #         self.writer = SummaryWriter(self.config["output_dir"])

    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        torch.cuda.set_device(self.rank % self.world_size)
        torch.distributed.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
    
    def cleanup():
        torch.dist.destroy_process_group()

    def init_datasets(self):
        
        if self.config['degradation_type'] == 'real':
            print(".......... real degradation ..............")
            self.train_set = TrainDataset_real(self.config, self.degradation_config)
            self.val_set = ValDataset_real(self.config, self.degradation_config)
        elif self.config['degradation_type'] == 'paired':
            print(".......... paired degradation ..............")
            self.train_set = TrainDataset_real_paired(self.config, self.degradation_config)
            self.val_set = ValDataset_real(self.config, self.degradation_config)
        else:
            print(".......... norm degradation ..............")
            self.train_set = TrainDataset(self.config, self.degradation_config)
            self.val_set = ValDataset(self.config)
        # train_set = TrainDataset_real(config, degradation_config)
        if self.config["dist"]:
            self.train_sampler = DistributedSampler(self.train_set, shuffle=True, drop_last=True)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                batch_size=self.config["batch_size"]//len(self.config["gpus"]), 
                shuffle=False, 
                num_workers=self.config["num_workers"]//len(self.config["gpus"]), 
                drop_last=False, 
                pin_memory=True,
                sampler=self.train_sampler)
        else:
            self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                batch_size=self.config["batch_size"], 
                shuffle=True, 
                num_workers=self.config["num_workers"], 
                drop_last=True, 
                pin_memory=True)
        
        self.val_loader = torch.utils.data.DataLoader(self.val_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            drop_last=False, 
            pin_memory=True
            )
    
    def output_log(self, log_str):
        log_file_path = os.path.join(self.config["output_dir"], "train_log.txt")
        with open(log_file_path, "a") as f:
            f.write(log_str)
    
    def init_model(self):
        # self.model = RFLite_v2()
        self.model = RFLite_v3()

    def init_loss_and_optimizer(self):
        
        self.l1_criterion = torch.nn.L1Loss().to(self.device)
        self.ssim_criterion = SSIMLoss().to(self.device)
        self.lowrank_criterion = LowRankLoss().to(self.device)
        self.lowrank16_criterion = LowRankLoss16().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=0)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config["scheduler_milestones"], gamma=self.config["scheduler_gamma"])

        self.jpeger = diffjpeg.DiffJPEG(differentiable=False).cuda()

    @torch.no_grad()
    def real_degradations(self, batch):

        opt = self.degradation_config
        L = batch["L"].to(self.device)  # low-quality image
        H = batch["H"].to(self.device)
        kernel = batch["kernel"].to(self.device)
        kernel2 = batch["kernel2"].to(self.device)
        sinc_kernel = batch["sinc_kernel"].to(self.device)

        ori_h, ori_w = L.size()[2:4]
        out = L

        if np.random.uniform() > opt['skip_degradation_prob']:
        # ----------------------- The first degradation process ----------------------- #
        # blur
            if opt["do_first_blur"]:
                out = utils_image.filter2D(out, kernel)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = opt['gray_noise_prob']
            if np.random.uniform() < opt['gaussian_noise_prob']:
                out = degradations.random_add_gaussian_noise_pt(
                    out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = degradations.random_add_poisson_noise_pt(
                    out,
                    scale_range=opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        # if np.random.uniform() < opt['second_blur_prob']:
        #     out = utils_image.filter2D(out, kernel2)
        # # random resize
        # updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
        # if updown_type == 'up':
        #     scale = np.random.uniform(1, opt['resize_range2'][1])
        # elif updown_type == 'down':
        #     scale = np.random.uniform(opt['resize_range2'][0], 1)
        # else:
        #     scale = 1
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # out = F.interpolate(
        #     out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode)
        # # add noise
        # gray_noise_prob = opt['gray_noise_prob2']
        # if np.random.uniform() < opt['gaussian_noise_prob2']:
        #     out = degradations.random_add_gaussian_noise_pt(
        #         out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        # else:
        #     out = degradations.random_add_poisson_noise_pt(
        #         out,
        #         scale_range=opt['poisson_scale_range2'],
        #         gray_prob=gray_noise_prob,
        #         clip=True,
        #         rounds=False)
        
            if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
                out = utils_image.filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
                out = utils_image.filter2D(out, sinc_kernel)

        # clamp and round
        L = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return L, H
    

    def train_loss(self, epoch, step):
        self.model.train()

        E = self.model(self.L)

        l1_loss = self.l1_criterion(E, self.H)
        # ssim
        ssim_loss = self.ssim_criterion(E, self.H)
        ssim_weight = -0.1
        ssim_loss = ssim_weight * ssim_loss

        # Low rank loss
        lowrank_loss = self.lowrank_criterion(E)
        loss1_weight = 8
        loss1 = loss1_weight * lowrank_loss
        lowrank16_loss = self.lowrank16_criterion(E)
        loss2_weight = 8
        loss2 = loss2_weight * lowrank16_loss

        loss = l1_loss + ssim_loss + loss1 + loss2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.rank == 0:
            # sys.stdout.write(f"\r[{epoch},{step}] loss {loss.item():.4f} l1 loss {l1_loss.item():.4f} other loss {other_loss.item():.4f}")
            sys.stdout.write(f"\r[{epoch},{step}] loss {loss.item():.4f} l1 loss {l1_loss.item():.4f} other loss {loss1.item():.4f} loss2 {loss2.item():.4f}")
            # sys.stdout.write(f"\r[{epoch},{step}] loss {loss.item():.4f} l1 loss {l1_loss.item():.4f} other loss {other_loss.item():.4f} loss2 {loss2.item():.4f} vif loss {vif_loss.item():.4f}")
            sys.stdout.flush()
        if "tb_writer" in self.config and self.config["tb_writer"] is not None:
            self.config["tb_writer"].add_scalar("loss/train", loss, step)
            self.config["tb_writer"].add_scalar("L1 loss/train", l1_loss, step)
            self.config["tb_writer"].add_scalar("SSIM loss/train", other_loss, step)
            self.config["tb_writer"].add_scalar("LowRankLoss loss/train", loss1, step)
            self.config["tb_writer"].add_scalar("LowBandEntropyLoss loss/train", loss2, step)

    def train(self, rank, world_size):

        epoch_start = 0
        step_start = 0

        # Random seed
        seed = self.config["manual_seed"]
        if seed is None:
            seed = random.randint(1, 10000)
        if self.rank == 0:
            print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # init model
        self.init_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        if self.config["dist"]:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)

        # loss and optimizer
        self.init_loss_and_optimizer()
        print("start learning rate: ", self.config["learning_rate"])
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"], weight_decay=0)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config["scheduler_milestones"], gamma=self.config["scheduler_gamma"])
        
        output_dir = self.config["output_dir"]
        if not os.path.isdir(output_dir) and self.rank == 0:
            os.makedirs(output_dir)

        # DDP init model parameter
        init_snapshot_filename = "model_init.pth"
        init_snapshot_path = os.path.join(output_dir, init_snapshot_filename)
        if self.config["dist"] and not self.config["network"]["pretrained"]:
            if self.rank == 0:
                save_checkpoint(init_snapshot_path, self.model, self.optimizer, self.scheduler, epoch_start, step_start, float("inf"))
        torch.distributed.barrier()

        # load pretrained weights if needed
        if self.config["network"]["pretrained"] is not None:
            snapshot_path = self.config["network"]["pretrained"]
        else:
            snapshot_path = init_snapshot_path
            print("Train from scratch")
        epoch_start, step_start, _ = load_checkpoint(snapshot_path, self.model, self.device)
        print("Loaded pretrained model.")
        
        # init
        if self.rank == 0:
            log_str = "*************************************************\n"
            log_str += f"Start training from step {step_start} at {time.strftime('%X %x')}\n"
            print(log_str)
            self.output_log(log_str)

            tb_dir=os.path.join(output_dir, "tb")
            self.config["tb_writer"] = SummaryWriter(log_dir=tb_dir, flush_secs=60)

        epochs_max = 1000
        current_step = step_start
        for epoch in range(epoch_start, epochs_max):
            for i, batch in enumerate(self.train_loader):
                current_step += 1
                if self.config["degradation_type"] == 'real' or self.config["degradation_type"] == 'paired':
                    self.L, self.H = self.real_degradations(batch)
                    self.train_loss(epoch, current_step)
                else:
                    self.L = batch["L"].to(device)  # low-quality image
                    self.H = batch["H"].to(device)
                    self.train_loss(epoch, current_step)

                if current_step % self.config["steps_val"] == 0 and self.rank == 0:
                    snapshot_filename = "step_{}.pth".format(current_step)
                    snapshot_path = os.path.join(output_dir, snapshot_filename)
                    print("current learning rate: ", self.scheduler.get_lr())
                    if self.config["degradation_type"] == 'real'or self.config["degradation_type"] == 'paired':
                        self.valildate_real(epoch, current_step)
                    else:
                        self.valildate(epoch, current_step)

                    # save_checkpoint(snapshot_path, model, optimizer, scheduler, epoch, current_step, float("inf"))
            self.scheduler.step()



    @torch.no_grad()
    def valildate(self, epoch, step):
        self.model.eval()

        all_loss = []
        all_psnr = []
        all_ssim = []
        print("------------validating--------------")
        start_time = time.time()
        for i, batch in enumerate(self.val_loader):
            L = batch["L"].to(self.device)  # low-quality image
            H = batch["H"].to(self.device)  # high-quality image

            E = model(L)

            l1_loss = self.l1_criterion(E, H).item()
            other_loss = self.other_criterion(E, H).item()
        
            E = utils_image.tensor2single(E)
            H = utils_image.tensor2single(H)
            E = utils_image.single2uint(E)
            H = utils_image.single2uint(H)
            psnr = utils_image.calculate_psnr(E, H, border=4)
            ssim = utils_image.calculate_ssim(E, H, border=4)

            all_loss.append(l1_loss)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            sys.stdout.write(f"\rval [{i + 1}/{len(self.val_loader)}] loss {l1_loss:.4f} psnr {psnr:.4f} ssim {ssim:.4f}")
            sys.stdout.flush()
        end_time = time.time()

        avg_loss = np.mean(all_loss)
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        duration = end_time - start_time
        print(f"\r[{epoch},{step}] val loss: {avg_loss:.4f}, psnr {avg_psnr:.4f}, ssim {avg_ssim:.4f}, duration {duration:.2f}")
        print("===============================================================")

        if "tb_writer" in self.config and self.config["tb_writer"] is not None:
            self.config["tb_writer"].add_scalar("Loss/val", avg_loss, step)
            self.config["tb_writer"].add_scalar("PSNR/val", avg_psnr, step)
            self.config["tb_writer"].add_scalar("SSIM/val", avg_ssim, step)

        # log
        output_dir = self.config["output_dir"]
        log_str = f"epoch {epoch} step {step} val loss: {avg_loss:.4f}, psnr: {avg_psnr:.4f}, ssim: {avg_ssim:.4f}\n"
        self.output_log(log_str)

        # save weights
        snapshot_filename = "step_{}.pth".format(step)
        snapshot_path = os.path.join(output_dir, snapshot_filename)
        save_checkpoint(snapshot_path, self.model, self.optimizer, self.scheduler, epoch, step, l1_loss)
        # torch.save(model.state_dict(), snapshot_path, f'epoch-{epoch}.pth')
        # save best weights
        if "best_psnr" not in self.config or avg_psnr >= self.config["best_psnr"]:
            best_psnr_snapshot_path = os.path.join(output_dir, "model_best_psnr.pth")
            # torch.save(model.state_dict(), best_psnr_snapshot_path, f'epoch-{epoch}.pth')
            save_checkpoint(best_psnr_snapshot_path, self.model, self.optimizer, self.scheduler, epoch, step, l1_loss)
            self.config["best_psnr_step"] = step
            self.config["best_psnr"] = avg_psnr
    
    @torch.no_grad()
    def valildate_real(self, epoch, step):
        self.model.eval()

        all_loss = []
        all_psnr = []
        all_ssim = []
        print("------------validating--------------")
        start_time = time.time()
        for i, batch in enumerate(self.val_loader):
            L, H = self.real_degradations(batch)

            E = self.model(L)

            l1_loss = self.l1_criterion(E, H).item()

        
            E = utils_image.tensor2single(E)
            H = utils_image.tensor2single(H)
            L = utils_image.tensor2single(L)

            E = utils_image.single2uint(E)
            H = utils_image.single2uint(H)
            L = utils_image.single2uint(L)

            res = E.astype(np.float32) - L.astype(np.float32)
            res = np.abs(res) * 10
            res = res.astype(np.uint8)

            output_dir = os.path.join(self.config["output_dir"], str(epoch))
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, str(i) + ".png")
            save_image = np.concatenate([L, E, H, res], axis=1)
            utils_image.img_save(save_image, output_path)

            psnr = utils_image.calculate_psnr(E, H, border=4)
            ssim = utils_image.calculate_ssim(E, H, border=4)

            all_loss.append(l1_loss)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            sys.stdout.write(f"\rval [{i + 1}/{len(self.val_loader)}] loss {l1_loss:.4f} psnr {psnr:.4f} ssim {ssim:.4f}")
            sys.stdout.flush()
        end_time = time.time()

        avg_loss = np.mean(all_loss)
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        duration = end_time - start_time
        print(f"\r[{epoch},{step}] val loss: {avg_loss:.4f}, psnr {avg_psnr:.4f}, ssim {avg_ssim:.4f}, duration {duration:.2f}")
        print("===============================================================")

        if "tb_writer" in self.config and self.config["tb_writer"] is not None:
            self.config["tb_writer"].add_scalar("Loss/val", avg_loss, step)
            self.config["tb_writer"].add_scalar("PSNR/val", avg_psnr, step)
            self.config["tb_writer"].add_scalar("SSIM/val", avg_ssim, step)

        # log
        output_dir = self.config["output_dir"]
        log_str = f"epoch {epoch} step {step} val loss: {avg_loss:.4f}, psnr: {avg_psnr:.4f}, ssim: {avg_ssim:.4f}\n"
        self.output_log(log_str)

        # save weights
        snapshot_filename = "step_{}.pth".format(step)
        snapshot_path = os.path.join(output_dir, snapshot_filename)
        save_checkpoint(snapshot_path, self.model, self.optimizer, self.scheduler, epoch, step, l1_loss)
        # torch.save(model.state_dict(), snapshot_path, f'epoch-{epoch}.pth')
        # save best weights
        if "best_psnr" not in self.config or avg_psnr >= self.config["best_psnr"]:
            best_psnr_snapshot_path = os.path.join(output_dir, "model_best_psnr.pth")
            save_checkpoint(best_psnr_snapshot_path, self.model, self.optimizer, self.scheduler, epoch, step, l1_loss)
            self.config["best_psnr_step"] = step
            self.config["best_psnr"] = avg_psnr


if __name__ == "__main__":

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '45678'
    gpu_list = ",".join(str(x) for x in DEFAULT_CONFIG["gpus"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    world_size = torch.cuda.device_count()
    mp.spawn(
        Trainer,
        nprocs=world_size,
        args=(world_size, DEFAULT_CONFIG, DEGRADATION_CONFIG,),
        join=True)




