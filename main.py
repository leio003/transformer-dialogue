# -*- coding: utf-8 -*-
import os
import time
import math
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from dataset import dailyDataset, collate_func
from transformer_based import transformer_base
from datetime import datetime
import argparse
import random
import numpy as np
import logging
from gensim.models import KeyedVectors
import re
from collections import defaultdict
from os.path import join, exists
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.utils.tensorboard import SummaryWriter
import json


SENT_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
logger = None
pad_id = 1


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', default=False, action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--train_path', default='data/dialogues_train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--test_path', default='data/dialogues_test.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--log_path', default='log/training_{}.log'.format(datetime.now().strftime('%Y-%m-%d')), type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--epochs', default=1, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练batch size')
    parser.add_argument('--hidden_size', default=300, type=int, required=False, help='隐藏层大小')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--log_step', default=25, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=42, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--embedding_path', type=str, default='embedding/glove.txt', help="加载glove向量")
    parser.add_argument('--num_words', type=int, default=50000, help="embedding的词个数")
    parser.add_argument('--dropout', type=int, default=0.2, help="dropout的大小")
    parser.add_argument('--word2idx', type=str, default='embedding/word2idx.json', help="word2idx文件保存位置")
    return parser.parse_args()

def set_random_seed(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def train(model, word2idx, args):
    gen_train_dataset = dailyDataset(args.train_path, word2idx)
    gen_train_dataloader = DataLoader(gen_train_dataset, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=collate_func)
    total_steps = int(gen_train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    logger.info('starting training')
    # 用于统计每次梯度累计的loss
    running_loss = 0
    # 统计一共训练了多少个step
    overall_step = 0
    # 记录tensorboardX
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    # 开始训练
    for epoch in tqdm(range(args.epochs)):
        epoch_start_time = datetime.now()
        for batch_idx, batch in enumerate(tqdm(gen_train_dataloader)):

            src_batch, tgt_batch, src_pad_batch, tgt_pad_batch, tgt_mask_batch = batch['src_ids'], \
                                                                                 batch['tgt_ids'], batch[
                                                                                     'src_pad_mask'], batch[
                                                                                     'tgt_pad_mask'], batch['tgt_mask']

            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            src_pad_batch = src_pad_batch.to(args.device)
            tgt_pad_batch = tgt_pad_batch.to(args.device)
            tgt_mask_batch = tgt_mask_batch.to(args.device)

            outputs = model(src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch)

            output = outputs[:, :-1, :].contiguous().view(-1, outputs.size(2))
            label = tgt_batch[:, 1:].contiguous().view((-1))
            loss = loss_fn(output, label)
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
            loss.backward()
            # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                running_loss += loss.item()
                # 更新参数
                optimizer.step()
                # 清空梯度信息
                optimizer.zero_grad()
                overall_step += 1
                # 更新日志与tnesorboardX信息
                if (overall_step + 1) % args.log_step == 0:
                    logger.info(
                        "batch {} of epoch {}, loss {}".format(batch_idx + 1, epoch + 1, loss))
                    tb_writer.add_scalar('loss', loss.item(), overall_step)
        if ((epoch + 1) % 5 == 0):
            eval_loss = evaluate(model, word2idx, args)
            logger.info('eval loss:{}'.format(eval_loss))
            logger.info('saving model for epoch {}_loss_{}'.format((epoch + 1), eval_loss))
            model_path = join(args.dialogue_model_output_path,
                            'model_epoch{}.pt'.format(epoch + 1))
            torch.save(model, model_path)

        logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    logger.info('saving model finally')
# 当前训练对话模型
    model_path = join(args.dialogue_model_output_path, 'model_finally.pt')
    torch.save(model, model_path)

    logger.info('training finished')

def evaluate(model, word2idx, args):
    test_dataset = dailyDataset(args.test_path, word2idx, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_func)
    model.eval()
    logger.info('starting evaluating')
    loss_ls = []
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            src_batch, tgt_batch, src_pad_batch, tgt_pad_batch, tgt_mask_batch = batch['src_ids'], \
                                                                                 batch['tgt_ids'], batch[
                                                                                     'src_pad_mask'], batch[
                                                                                     'tgt_pad_mask'], batch['tgt_mask']

            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            src_pad_batch = src_pad_batch.to(args.device)
            tgt_pad_batch = tgt_pad_batch.to(args.device)
            tgt_mask_batch = tgt_mask_batch.to(args.device)

            outputs = model(src_batch, tgt_batch, src_pad_batch, tgt_pad_batch, tgt_mask_batch)

            output = outputs[:, :-1, :].contiguous().view(-1, outputs.size(2))
            label = tgt_batch[:, 1:].contiguous().view((-1))
            loss = loss_fn(output, label)
            loss_ls.append(loss.item())
        logger.info("finishing evaluating")
    return np.mean(loss_ls)

def main():
    args = setup_train_args()
    global logger
    logger = create_logger(args)
    logger.info('args config:\n{}'.format(args.__dict__))
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(args.device))
    # 设置随机数种子
    if args.seed:
        set_random_seed(args)
    # 加载预训练模型 可用glove
    # emb, word2idx = load_word_embeddings(args)
    word2idx = defaultdict(lambda: 1)
    with open(args.word2idx, 'r') as load_f:
        wo = json.load(load_f)
    for k, v in wo.items():
        word2idx[k] = v

    model = transformer_base(args.num_words + 4, )
    generator = model.to(args.device)
    # 对数据进行处理
    # 创建模型存放文件夹
    if not os.path.exists(args.dialogue_model_output_path):
        os.mkdir(args.dialogue_model_output_path)

    # 开始训练
    train(generator, word2idx, args)
    # 测试模型
    eval_loss = evaluate(model, word2idx, args)
    logger.info('eval loss:{}'.format(eval_loss))


if __name__ == "__main__":
    main()