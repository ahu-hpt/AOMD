from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from metric.utils import pdist
import numpy as np
import collections

__all__ = ['L1Triplet', 'L2Triplet', 'ContrastiveLoss','Similarity','HardDarkRank','AOMD']


class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p==2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                     margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)


class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_loss = (F.pairwise_distance(anchor_embed, positive_embed, p=2)).pow(2)
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed, negative_embed, p=2)).clamp(min=0).pow(2)

        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()


class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class FitNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, student, teacher):
        if student.dim() == 2:
            student = student.unsqueeze(2).unsqueeze(3)
            teacher = teacher.unsqueeze(2).unsqueeze(3)

        return (self.transform(student) - teacher).pow(2).mean()


class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()

class MapMatrix(nn.Module):
    def __init__(self, s_shapes, t_shapes):
        super(MapMatrix, self).__init__()
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        s_c = [s[1] for s in s_shapes]
        t_c = [t[1] for t in t_shapes]
        if np.any(np.asarray(s_c) != np.asarray(t_c)):
            raise ValueError('There is a dimensional mismatch in the first dimension (error in MapMatrix)')

    def forward(self, g_s, g_t):
        s_mm = self.compute_mm(g_s)
        t_mm = self.compute_mm(g_t)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_mm, t_mm)]
        return loss_group

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_mm(g):
        mm_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)
            mm = bot * top
            mm = (mm).mean(-2)  # mode2
            # mm = (mm).mean(-1)  # mode1
            mm_list.append(mm)
        return mm_list


class AOMD(nn.Module):
    def __init__(self,start_weight_2=1,end_weight_2=1):
        super(AOMD, self).__init__()
        # self.stu_layers_feats_adap = nn.ModuleList(
        #     [nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0), nn.ReLU()),
        #      nn.Sequential(nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0), nn.ReLU()),
        #      nn.Sequential(nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0), nn.ReLU()),
        #      nn.Sequential(nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0), nn.ReLU())]
        # ).cuda()
        self.start_weight_2 = start_weight_2
        self.end_weight_2 = end_weight_2

    def forward(self, t_feats_list, s_feats_list):
        # t_feats_list : [L S N C H W]  [layers scales bzs channel height width]
        # or
        # t_feats_list : [L N C H W]  [layers scales bzs channel height width]  in this code

        map_kdloss = 0
        global_map_loss = 0
        local_map_loss = 0

        '''extract features from inputs of backbone, fpn, ssh'''
        new_t_feats_list = []
        new_s_feats_list = []

        ''' backbone datatype is OrderDict , others is list
        new_t_feats_list,new_s_feats_list
        [ [ [N C H W],
            [N C H W],
            [N C H W] ]  ,  
          [ [N C H W],
            [N C H W],
            [N C H W] ]  ,
          [ [N C H W],
            [N C H W],
            [N C H W] ]    ]
        '''
        for t_feats, s_feats in zip(t_feats_list, s_feats_list):
            if isinstance(t_feats, collections.OrderedDict):
                new_t_feats_list.append(list(t_feats.values()))
                new_s_feats_list.append(list(s_feats.values()))
            else:
                new_t_feats_list.append(t_feats)
                new_s_feats_list.append(s_feats)



        ''' we need to kd the mapping relation from teacher,
        two transfer relationships from backbone to fpn and fpn to ssh.
        t_layers_channel_feats , s_layers_channel_feats : is the channel features from three corresponding layers  
        [ [ [N C 1 1],
            [N C 1 1],
            [N C 1 1] ]  ,  
          [ [N C 1 1],
            [N C 1 1],
            [N C 1 1] ]  ,
          [ [N C 1 1],
            [N C 1 1],
            [N C 1 1] ]    ]  --> the another different key is every list's feature is the corresponding feature from each layer
        '''

        t_layers_global_feats = []
        s_layers_global_feats = []

        for t_feats, s_feats in zip(new_t_feats_list, new_s_feats_list):
            # t_feats , s_feats  [ N, C, H, W]
            t_f = torch.mean(t_feats, [2, 3], keepdim=True)
            s_f = torch.mean(s_feats, [2, 3], keepdim=True)

            t_layers_global_feats.append(t_f)
            s_layers_global_feats.append(s_f)

        """ ensure the same channel between student and teacher"""
        # s_layers_global_feats_adap = []
        # for i, s_layer_channel_feats in zip(range(len(s_layers_global_feats)), s_layers_global_feats):
        #     s_layer_channel_feats_adap = []
        #     for j, s_feats in zip(range(len(s_layer_channel_feats)), s_layer_channel_feats):
        #         s_layer_channel_feats_adap.append(self.stu_layers_feats_adap[i * 3 + j](s_feats))
        #     s_layers_global_feats_adap.append(s_layer_channel_feats_adap)
        s_layers_global_feats_adap = s_layers_global_feats
        # from models.FSP import FSP
        map_loss_list = []

        t_layer_feats_shapes = [t_layer_feat.shape for t_layer_feat in t_layers_global_feats]
        s_layer_feats_shapes = [s_layer_feat.shape for s_layer_feat in s_layers_global_feats_adap]
        Map_loss = MapMatrix(t_layer_feats_shapes, s_layer_feats_shapes)
        map_loss_list += Map_loss(t_layers_global_feats, s_layers_global_feats_adap)
        global_map_loss = sum(map_loss_list)

        t_layers_local_feats = []
        s_layers_local_feats = []

        for t_feats, s_feats in zip(new_t_feats_list, new_s_feats_list):
            # t_feats , s_feats  [ N, C, H, W]
            t_f = torch.mean(t_feats, [1], keepdim=True)
            s_f = torch.mean(s_feats, [1], keepdim=True)
            N, C, H, W = t_f.shape
            t_f = t_f.view(N, H * W, 1, 1)
            t_layers_local_feats.append(t_f)
            N, C, H, W = s_f.shape
            s_f = s_f.view(N, H * W, 1, 1)
            s_layers_local_feats.append(s_f)


        # from models.FSP import FSP
        map_loss_list = []

        t_layer_feats_shapes = [t_layer_feat.shape for t_layer_feat in t_layers_local_feats]
        s_layer_feats_shapes = [s_layer_feat.shape for s_layer_feat in s_layers_local_feats]
        map_loss = MapMatrix(t_layer_feats_shapes, s_layer_feats_shapes)
        map_loss_list += map_loss(t_layers_local_feats, s_layers_local_feats)
        local_map_loss = sum(map_loss_list)

        map_kdloss = 1.0*global_map_loss + 1.0*local_map_loss

        """ dunamic weights """
        # start_weigth = 1e2
        # end_weigth = 1e1
        # 73.37 resnet32x4_kd_resnet8x4
        # 70.61 resnet56_kd_resnet20
        # resnet110_kd_resnet20
        # start_weigth_2 = 2
        # end_weigth_2 = 1

        # resnet110_kd_resnet32
        # start_weigth_2 = 1.5
        # end_weigth_2 = 0.75

        '''
        start_weight_2 = self.start_weight_2
        end_weight_2 = self.end_weight_2


        imitation_loss_weigth_2 = start_weight_2 + (end_weight_2 - start_weight_2) * (float(epoch) / max_epoch)
        map_kdloss *= imitation_loss_weigth_2
        '''

        # return map_kdloss+ar_kdloss
        return map_kdloss