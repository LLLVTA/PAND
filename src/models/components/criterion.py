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
        
        # CoOp configuration: no class_num scaling needed when using CoOp features
        self.use_coop = getattr(args, 'use_coop', False)
        
        # NLRD configuration
        self.use_nlrd = getattr(args, 'use_nlrd', False)
        if self.use_nlrd:
            nlrd_k = getattr(args, 'nlrd_k', 1)
            nlrd_lambda = getattr(args, 'nlrd_lambda', 1.0)
            self.nlrd_weight = getattr(args, 'nlrd_weight', 0.5)
            self.nlrd_criterion = NLRDLoss(k=nlrd_k, lambda1=nlrd_lambda)

    def __call__(self, inputs):
        hidden_features, out, clip_img_features, clip_nlp_features, aligned_img, aligned_nlp = inputs
        
        # ===== High-level: L1 point-to-point alignment =====
        img_loss = self.criterion_aligned_img_kd(hidden_features, aligned_img)

        student_nlp_logits = self.logit_scale * hidden_features @ aligned_nlp.T / self.temperature
        teacher_nlp_logits = self.logit_scale * clip_img_features @ clip_nlp_features.T / self.temperature
        kd_loss = self.criterion_nlp_kd(F.log_softmax(student_nlp_logits, dim=1),
                             F.softmax(teacher_nlp_logits, dim=1)) * (self.temperature * self.temperature)
        
        # VL2Lite original requires class_num scaling, but not needed when using CoOp
        if not self.use_coop:
            kd_loss = kd_loss * self.args.class_num / 2
        
        # ===== NLRD: Neighborhood Logits Relation Distillation =====
        if self.use_nlrd:
            # Compute teacher logits: similarity between image and text features
            teacher_logits = self.logit_scale * clip_img_features @ clip_nlp_features.T
            # Student logits use classification output directly
            student_logits = out
            
            # Compute NLRD loss
            nlrd_loss = self.nlrd_criterion(student_logits, teacher_logits)
            
            return img_loss, kd_loss, nlrd_loss
        
        return img_loss, kd_loss