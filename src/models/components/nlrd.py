# coding=utf-8
"""
Neighborhood Logits Relation Distillation (NLRD)
Adapted from NRKD paper: "Neighborhood relation-based knowledge distillation for image classification"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NLRDLoss(nn.Module):
    """
    Neighborhood Logits Relation Distillation Loss
    Only includes neighborhood response relation distillation, no feature distillation
    
    Args:
        k (int): Number of neighbors, default is 1
        lambda1 (float): Weight coefficient for NLRD loss, default is 1.0
    """
    def __init__(self, k=1, lambda1=1.0):
        super(NLRDLoss, self).__init__()
        self.k = k
        self.lambda1 = lambda1

    def angle(self, t):
        """
        Select K nearest neighbors based on cosine similarity
        
        Args:
            t: logits tensor [batch_size, num_classes]
            
        Returns:
            sort: Sorted similarities [batch_size, batch_size]
            idx: Sorted indices [batch_size, batch_size]
        """
        t = F.normalize(t, dim=1)  # L2 normalization
        cosine = torch.mm(t, t.T)  # Compute cosine similarity matrix [B, B]
        sort, idx = torch.sort(cosine, descending=True)  # Sort in descending order
        return sort, idx

    def compute_relation_loss(self, logits_s, logits_t, b, nebor_idx):
        """
        Compute neighborhood response relation loss
        
        Args:
            logits_s: Student network logits [batch_size, num_classes]
            logits_t: Teacher network logits [batch_size, num_classes]
            b: batch size
            nebor_idx: Neighbor indices [batch_size, k]
            
        Returns:
            loss: Neighborhood relation loss
        """
        # If batch is too small, number of neighbors may be less than k, adapt dynamically
        actual_k = nebor_idx.size(1)

        # Extract first neighbor's logits
        idx = nebor_idx[:, 0]
        nebor_s_p = torch.index_select(logits_s, 0, idx)
        nebor_t_p = torch.index_select(logits_t, 0, idx)
        
        # Concatenate logits from all actual_k neighbors
        for i in range(1, actual_k):
            idx = nebor_idx[:, i]
            n_s_p = torch.index_select(logits_s, 0, idx)
            n_t_p = torch.index_select(logits_t, 0, idx)
            nebor_s_p = torch.cat((nebor_s_p, n_s_p), 1)
            nebor_t_p = torch.cat((nebor_t_p, n_t_p), 1)
        
        # Reshape to [batch_size, actual_k, num_classes]
        nebor_s_p = nebor_s_p.view(b, actual_k, -1)
        nebor_t_p = nebor_t_p.view(b, actual_k, -1)
        
        # Expand dimensions for difference computation
        l_s = torch.unsqueeze(logits_s, 1)  # [B, 1, M]
        l_t = torch.unsqueeze(logits_t, 1)  # [B, 1, M]
        
        # Compute logits differences between samples and neighbors
        res_s_diff = l_s - nebor_s_p  # [B, K, M]
        res_t_diff = l_t - nebor_t_p  # [B, K, M]
        
        # Compute loss using JS divergence
        loss_nebor_res = self.js_div(res_s_diff, res_t_diff) / (b * actual_k)
        
        return loss_nebor_res

    def js_div(self, q, p):
        """
        Jensen-Shannon divergence
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = (P+Q)/2
        
        Args:
            q: Student response differences [B, K, M]
            p: Teacher response differences [B, K, M]
            
        Returns:
            loss: JS divergence loss
        """
        # Perform softmax on the last dimension (class dimension)
        q = F.softmax(q, dim=2)
        p = F.softmax(p, dim=2)
        
        # Compute intermediate distribution M
        log_mean = ((q + p) / 2).log()
        
        # JS divergence = 0.5 * [KL(q||M) + KL(p||M)]
        loss = (F.kl_div(log_mean, q, reduction='sum') + 
                F.kl_div(log_mean, p, reduction='sum')) / 2
        
        return loss

    def forward(self, logits_s, logits_t):
        """
        Forward pass to compute NLRD loss
        
        Args:
            logits_s: Student network logits [batch_size, num_classes]
            logits_t: Teacher network logits [batch_size, num_classes]
            
        Returns:
            loss: NLRD loss value
        """
        b = logits_t.size(0)
        
        # Step 1: Select K nearest neighbors based on teacher logits
        sort, idx = self.angle(logits_t)
        nebor_idx = idx[:, 1:self.k+1]  # Exclude self (index 0), select next k neighbors
        
        # Step 2: Compute neighborhood response relation loss
        loss = self.lambda1 * self.compute_relation_loss(logits_s, logits_t, b, nebor_idx)
        
        return loss
