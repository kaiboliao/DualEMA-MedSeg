from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, test_3d_patch
# from dataloaders.dataset import *
from dataloaders.LADataset import LAHeart
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables
from utils.LA_utils import to_cuda
from utils.BCP_utils import *
from utils.losses import *

from networks.VNet import VNet
from utils.losses import mask_DiceLoss

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Datasets/la/data', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='LAtrain', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=3000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=10000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=1e-3, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1345, help='random seed')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
args = parser.parse_args()


def global_feature_l1_loss(feat_s, feat_t, reduction='mean'):
    loss = torch.abs(feat_s - feat_t)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def flatten_feature(feat):
    B, C, D, H, W = feat.shape
    return feat.view(B, C, -1)

def compute_self_correlation(feat):
    feat = F.normalize(feat, dim=1)  # (B, C, N)
    M = torch.bmm(feat.transpose(1, 2), feat)

    return M

def local_relation_loss(feat_s, feat_t):

    # ---- flatten to regional vectors ----
    v_s = flatten_feature(feat_s)   # (B, C, N)
    v_t = flatten_feature(feat_t)   # (B, C, N)

    # ---- compute correlation matrices ----
    M_s = compute_self_correlation(v_s)
    M_t = compute_self_correlation(v_t).detach()

    # ---- L2 distance between relations ----
    loss = F.mse_loss(M_s, M_t)

    return loss

def create_Vnet(ema=False):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def pre_train(args, snapshot_path):
    model = create_Vnet()

    c_batch_size = 2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    max_epoch = 81
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch_num in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))

        if epoch_num % 5 == 0:
            model.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(snapshot_path, 'best_model.pth'.format(args.model))
                save_net_opt(model, optimizer, save_mode_path, epoch_num)
                save_net_opt(model, optimizer, save_best_path, epoch_num)
                logging.info("save best model to {}".format(save_mode_path))

            model.train()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = create_Vnet()
    ema_model = create_Vnet(ema=True).cuda()
    ema_model_s = create_Vnet(ema=True).cuda()

    c_batch_size = 2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_unlab', logging=logging)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_unlab', reverse=True, logging=logging)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)


    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    pretrained_model = os.path.join(pre_snapshot_path, 'best_model.pth')

    load_net_opt(model, optimizer, pretrained_model)
    load_net(ema_model, pretrained_model)
    load_net(ema_model_s, pretrained_model)


    model.train()
    ema_model.train()#（global structure）
    ema_model_s.train()#(sharp boundaries）

    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    best_dice = 0
    max_epoch = 276
    iterator = tqdm(range(1, max_epoch), ncols=70)

    high_confidence_queue = []
    queue_size = 6

    for epoch in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(
                zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda(
                [img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])

            with torch.no_grad():
                unoutput_a_l = ema_model(unimg_a, pred_type="pred")
                unoutput_b_l = ema_model(unimg_b, pred_type="pred")
                unoutput_a_s = ema_model_s(unimg_a, pred_type="pred")
                unoutput_b_s = ema_model_s(unimg_b, pred_type="pred")
                unoutput_a_fusion = 0.5 * unoutput_a_s + 0.5 * unoutput_a_l
                unoutput_b_fusion = 0.5 * unoutput_b_s + 0.5 * unoutput_b_l


                conf_a_l = torch.softmax(unoutput_a_l, dim=1).max(dim=1)[0].mean()
                conf_b_l = torch.softmax(unoutput_b_l, dim=1).max(dim=1)[0].mean()
                conf_a_s = torch.softmax(unoutput_a_s, dim=1).max(dim=1)[0].mean()
                conf_b_s = torch.softmax(unoutput_b_s, dim=1).max(dim=1)[0].mean()
                conf_a = 0.5 * conf_a_l + 0.5 * conf_a_s
                conf_b = 0.5 * conf_b_l + 0.5 * conf_b_s

                plab_a = get_cut_mask(unoutput_a_fusion, nms=1)
                plab_b = get_cut_mask(unoutput_b_fusion, nms=1)


                if conf_a > conf_b:
                    high_confidence_data = (unimg_a.clone(), plab_a.clone())
                else:
                    high_confidence_data = (unimg_b.clone(), plab_b.clone())

                if len(high_confidence_queue) >= queue_size:
                    high_confidence_queue.pop(0)
                high_confidence_queue.append(high_confidence_data)

                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)


            mixl_img = unimg_a * img_mask + img_b * (1 - img_mask)
            mixu_img = img_a * img_mask + unimg_b * (1 - img_mask)
            _, feat_global_u_t = ema_model(mixu_img, pred_type="teacher_l")
            _, feat_local_u_t = ema_model_s(mixu_img, pred_type="teacher_s")
            outputs_l, _, _ = model(mixl_img, pred_type="student")
            outputs_u, feat_local_u, feat_global_u = model(mixu_img, pred_type="student")

            if len(high_confidence_queue) >= 3:
                available_samples = high_confidence_queue[:-1]
                num_available = len(available_samples)
                if num_available >= 2:
                    indices = random.sample(range(num_available), 2)
                    hc_img1, hc_plab1 = available_samples[indices[0]]
                    hc_img2, hc_plab2 = available_samples[indices[1]]

                mixhl_img = unimg_a * img_mask + hc_img1 * (1 - img_mask)
                mixhu_img = hc_img2 * img_mask + unimg_b * (1 - img_mask)

                _, feat_global_hu_t = ema_model(mixhu_img, pred_type="teacher_l")
                _, feat_local_hu_t = ema_model_s(mixhu_img, pred_type="teacher_s")

                outputs_hl, _, _ = model(mixhl_img, pred_type="student")
                outputs_hu, feat_local_hu, feat_global_hu = model(mixhu_img, pred_type="student")

                loss_l = mix_loss(outputs_l, plab_a.long(), lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
                loss_u = mix_loss(outputs_u, lab_a, plab_b.long(), loss_mask, u_weight=args.u_weight)
                loss_hl = mix_loss(outputs_hl,plab_a.long(),hc_plab1.long(),loss_mask,u_weight=args.u_weight, unlab=True)
                loss_hu = mix_loss(outputs_hu,hc_plab2.long(),plab_b.long(),loss_mask,u_weight=args.u_weight)

                loss_loc_u = local_relation_loss(feat_local_u, feat_local_u_t)
                loss_loc_hu = local_relation_loss(feat_local_hu, feat_local_hu_t)
                loss_glb_u = global_feature_l1_loss(feat_global_u, feat_global_u_t)
                loss_glb_hu = global_feature_l1_loss(feat_global_hu, feat_global_hu_t)
                loss_LG = loss_l + loss_u + 0.5 * (loss_loc_u + loss_glb_u)
                loss_RQG = 0.7 * (loss_hl + loss_hu) + 0.5 * (loss_loc_hu + loss_glb_hu)
                loss = loss_LG + loss_RQG

                logging.info(
                    'epoch %d iteration %d : loss: %03f, loss_LG: %03f, loss_RQG: %03f, loss_l: %03f, loss_u: %03f, loss_hl: %03f, loss_hu: %03f'
                    % (epoch, iter_num, loss, loss_LG, loss_RQG, loss_l, loss_u, loss_hl, loss_hu))
                logging.info(
                    'loss_loc_u: %03f, loss_glb_u: %03f, loss_loc_hu: %03f, loss_glb_hu: %03f'
                    % (loss_loc_u, loss_glb_u, loss_loc_hu, loss_glb_hu))
                writer.add_scalar('self/loss', loss.item(), iter_num)
                writer.add_scalar('self/loss_l', loss_l.item(), iter_num)
                writer.add_scalar('self/loss_u', loss_u.item(), iter_num)
                writer.add_scalar('self/loss_hl', loss_hl.item(), iter_num)
                writer.add_scalar('self/loss_hu', loss_hu.item(), iter_num)
                writer.add_scalar('self/loss_loc_u', loss_loc_u.item(), iter_num)
                writer.add_scalar('self/loss_glb_u', loss_glb_u.item(), iter_num)
                writer.add_scalar('self/loss_loc_hu', loss_loc_hu.item(), iter_num)
                writer.add_scalar('self/loss_glb_hu', loss_glb_hu.item(), iter_num)
                writer.add_scalar('self/loss_LG', loss_LG.item(), iter_num)
                writer.add_scalar('self/loss_RQG', loss_RQG.item(), iter_num)
            else:
                _, feat_global_u_t = ema_model(mixu_img, pred_type="teacher_l")
                _, feat_local_u_t = ema_model_s(mixu_img, pred_type="teacher_s")

                loss_l = mix_loss(outputs_l, plab_a.long(), lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
                loss_u = mix_loss(outputs_u, lab_a, plab_b.long(), loss_mask, u_weight=args.u_weight)

                loss_loc_u = local_relation_loss(feat_local_u, feat_local_u_t)
                loss_glb_u = global_feature_l1_loss(feat_global_u, feat_global_u_t)

                loss = loss_l + loss_u + 0.5 *(loss_loc_u + loss_glb_u)
                logging.info('epoch %d iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f'
                             % (epoch, iter_num, loss, loss_l, loss_u))
                logging.info(
                    'loss_loc_u: %03f, loss_glb_u: %03f'
                    % (loss_loc_u, loss_glb_u))
                writer.add_scalar('self/loss', loss.item(), iter_num)
                writer.add_scalar('self/loss_l', loss_l.item(), iter_num)
                writer.add_scalar('self/loss_u', loss_u.item(), iter_num)
                writer.add_scalar('self/loss_loc_u', loss_loc_u.item(), iter_num)
                writer.add_scalar('self/loss_glb_u', loss_glb_u.item(), iter_num)
            writer.add_scalar('self/conf_a_l', conf_a_l, iter_num)
            writer.add_scalar('self/conf_a_s', conf_a_s, iter_num)
            writer.add_scalar('self/conf_b_l', conf_b_l, iter_num)
            writer.add_scalar('self/conf_b_s', conf_b_s, iter_num)
            logging.info('epoch %d iteration %d : conf_a_l: %03f, conf_a_s: %03f, conf_b_l: %03f, conf_b_s: %03f'
                         % (epoch, iter_num, conf_a_l, conf_a_s, conf_b_l, conf_b_s))

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, 0.99)
            update_ema_variables(model, ema_model_s, 0.9)

        if epoch % 5 == 0:
            model.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(self_snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best_path)
                logging.info("save best model to {}".format(save_mode_path))
            logging.info("cur dice %.4f, max dice %.4f" % (dice_sample, best_dice))
            writer.add_scalar('var/dice', dice_sample, iter_num)
            writer.add_scalar('var/max_dice', best_dice, iter_num)

            model.train()

if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./model/LAtrain/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/LAtrain/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Starting new training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/train_LA.py', self_snapshot_path)
    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    self_train(args, pre_snapshot_path, self_snapshot_path)

