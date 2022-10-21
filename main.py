#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
from pickle import FALSE, TRUE, load
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
import numpy as np
from torch.utils.data import DataLoader
from model import DIT


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('models'):
        os.makedirs('models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, net, train_loader, test_loader):
    """
    Function: train and test the model
    Param:
        net:   DIT
        train_loader:  consist of pcd and Transform dictionary
        test_loader:  consist of pcd and Transform dictionary
    """
    print("Use Adam")
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    epoch_factor = args.epochs / 100.0

    scheduler = MultiStepLR(opt,
                            milestones=[int(30*epoch_factor), int(60*epoch_factor), int(80*epoch_factor)],
                            gamma=0.1)

    info_test_best = None
             
    if args.load_model:
        print('load param from',args.model_path)
        net.load_state_dict(torch.load(args.model_path)) 
        net.eval()

    if args.eval == True:
        print('Testing begins! q(^_^)p ~~~')
    else:
        print('Training begins! q(^_^)p ~~~')

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()
        print(f" Epoch: {epoch}, LR: {lr}")
        if args.eval == False:
            info_train = net._train_one_epoch(epoch=epoch, train_loader=train_loader, opt=opt, args=args)
            gc.collect()
        info_test = net._test_one_epoch(epoch=epoch, test_loader=test_loader,vis=args.vis)
        if args.eval == False:
            scheduler.step()
        if info_test_best is None or info_test_best['loss'] < info_test['loss']:
            info_test_best = info_test
            info_test_best['stage'] = 'best_test'
            if args.eval == False:
                torch.save(net.state_dict(),'models/model.best.pth')
        if args.eval == False:
            net.logger.write(info_test_best)
            torch.save(net.state_dict(),'models/model.%d.pth' % (epoch))
        else:
            break
        gc.collect()
    if args.eval == True:
        print('Testing completed! /(^o^)/~~~')
    else:
        print('Training completed! /(^o^)/~~~')


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp3', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--n_emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=12, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--n_iters', type=int, default=3, metavar='N',
                        help='Num of iters to run inference')
    parser.add_argument('--discount_factor', type=float, default=0.9, metavar='N',
                        help='Discount factor to compute the loss')
    parser.add_argument('--n_ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--temp_factor', type=float, default=100, metavar='N',
                        help='Factor to control the softmax precision')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--lr', type=float, default=0.00003, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=4327, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualize the registration process')
    parser.add_argument('--cycle_consistency_loss', type=float, default=0.1, metavar='N',
                        help='cycle consistency loss')
    parser.add_argument('--discrimination_loss', type=float, default=0.5, metavar='N',
                        help='discrimination loss')
    parser.add_argument('--gaussian_noise', type=float, default=0.05, metavar='N',
                        help='The clip of gaussian noise')
    parser.add_argument('--unseen', type=bool, default=True, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--rot_factor', type=float, default=4, metavar='N',
                        help='Divided factor of rotation')
    parser.add_argument('--model_path', type=str, default='models/model.Low_Noise_Partial.pth', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch when training)')
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch when testing)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--n_points', type=int, default=1000, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--n_subsampled_points', type=int, default=800, metavar='N',
                        help='Num of subsampled points to use')
    parser.add_argument('--corres_mode', action='store_true', default=False,
                        help='decide whether use GMCCE or not')
    parser.add_argument('--GMCCE_Sharp', type=float, default=30, metavar='N',
                        help='The Sharp of GMCCE module')
    parser.add_argument('--GMCCE_Thres', type=float, default=0.6, metavar='N',
                        help='The Threshold of GMCCE module')                
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='decide whether load model or not')    
    parser.add_argument('--token_dim', default=64, type=int, metavar='PCT',
                        help='the token dim')

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    _init_(args)

    train_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                            num_subsampled_points=args.n_subsampled_points,
                                            partition='train', gaussian_noise=args.gaussian_noise,
                                            unseen=args.unseen, rot_factor=args.rot_factor),
                                batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)
    test_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                        num_subsampled_points=args.n_subsampled_points,
                                        partition='test', gaussian_noise=args.gaussian_noise,
                                        unseen=args.unseen, rot_factor=args.rot_factor),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=0)

    net = DIT(args).cuda()
    if args.load_model:
        if args.model_path is '':
            model_path = 'models/model.best.pth'
        else:
            model_path = args.model_path
        if not os.path.exists(model_path):
            print("can't find pretrained model")
            return

    train(args, net, train_loader, test_loader)
    print('FINISH')


if __name__ == '__main__':
    main()
