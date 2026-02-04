import torch
import numpy as np
import torch.nn.functional as F
from types import SimpleNamespace
from .nlrd import NLRDLoss

class KDCriterion:
    def __init__(self, **kwargs) -> None:
        args = SimpleNamespace(**kwargs)
        self.args = args
        self.criterion_aligned_img_kd = args.img_criterion
        self.criterion_nlp_kd = args.nlp_criterion
        self.temperature = args.temperature #2
        logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad = False)
        self.logit_scale = logit_scale.exp()
        
        # CoOp配置：如果使用CoOp特征，不需要class_num缩放
        self.use_coop = getattr(args, 'use_coop', False)
        
        # NLRD配置
        self.use_nlrd = getattr(args, 'use_nlrd', False)
        if self.use_nlrd:
            nlrd_k = getattr(args, 'nlrd_k', 1)
            nlrd_lambda = getattr(args, 'nlrd_lambda', 1.0)
            self.nlrd_weight = getattr(args, 'nlrd_weight', 0.5)
            self.nlrd_criterion = NLRDLoss(k=nlrd_k, lambda1=nlrd_lambda)

    def __call__(self, inputs):
        hidden_features, out, clip_img_features, clip_nlp_features, aligned_img, aligned_nlp = inputs
        
        # ===== High层: L1点对点对齐 =====
        img_loss = self.criterion_aligned_img_kd(hidden_features, aligned_img)

        student_nlp_logits = self.logit_scale * hidden_features @ aligned_nlp.T / self.temperature
        teacher_nlp_logits = self.logit_scale * clip_img_features @ clip_nlp_features.T / self.temperature
        kd_loss = self.criterion_nlp_kd(F.log_softmax(student_nlp_logits, dim=1),
                             F.softmax(teacher_nlp_logits, dim=1)) * (self.temperature * self.temperature)
        
        # VL2Lite原版需要class_num缩放，但使用CoOp时不需要
        if not self.use_coop:
            kd_loss = kd_loss * self.args.class_num / 2
        
        # ===== NLRD: 邻域Logits关系蒸馏 =====
        if self.use_nlrd:
            # 计算teacher logits: 图像特征与文本特征的相似度
            teacher_logits = self.logit_scale * clip_img_features @ clip_nlp_features.T
            # 学生logits直接使用分类输出
            student_logits = out
            
            # 计算NLRD损失
            nlrd_loss = self.nlrd_criterion(student_logits, teacher_logits)
            
            return img_loss, kd_loss, nlrd_loss
        
        return img_loss, kd_loss