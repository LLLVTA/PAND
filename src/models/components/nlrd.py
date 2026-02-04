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
    只包含邻域响应关系蒸馏，不包含特征蒸馏
    
    Args:
        k (int): 邻居数量，默认为1
        lambda1 (float): NLRD损失的权重系数，默认为1.0
    """
    def __init__(self, k=1, lambda1=1.0):
        super(NLRDLoss, self).__init__()
        self.k = k
        self.lambda1 = lambda1

    def angle(self, t):
        """
        基于余弦相似度选择K近邻
        
        Args:
            t: logits张量 [batch_size, num_classes]
            
        Returns:
            sort: 排序后的相似度 [batch_size, batch_size]
            idx: 排序后的索引 [batch_size, batch_size]
        """
        t = F.normalize(t, dim=1)  # L2归一化
        cosine = torch.mm(t, t.T)  # 计算余弦相似度矩阵 [B, B]
        sort, idx = torch.sort(cosine, descending=True)  # 降序排序
        return sort, idx

    def compute_relation_loss(self, logits_s, logits_t, b, nebor_idx):
        """
        计算邻域响应关系损失
        
        Args:
            logits_s: 学生网络的logits [batch_size, num_classes]
            logits_t: 教师网络的logits [batch_size, num_classes]
            b: batch size
            nebor_idx: 邻居索引 [batch_size, k]
            
        Returns:
            loss: 邻域关系损失
        """
        # 若batch过小，邻居数可能少于k，这里动态适配实际邻居数
        actual_k = nebor_idx.size(1)

        # 提取第一个邻居的logits
        idx = nebor_idx[:, 0]
        nebor_s_p = torch.index_select(logits_s, 0, idx)
        nebor_t_p = torch.index_select(logits_t, 0, idx)
        
        # 拼接所有actual_k个邻居的logits
        for i in range(1, actual_k):
            idx = nebor_idx[:, i]
            n_s_p = torch.index_select(logits_s, 0, idx)
            n_t_p = torch.index_select(logits_t, 0, idx)
            nebor_s_p = torch.cat((nebor_s_p, n_s_p), 1)
            nebor_t_p = torch.cat((nebor_t_p, n_t_p), 1)
        
        # 重塑为 [batch_size, actual_k, num_classes]
        nebor_s_p = nebor_s_p.view(b, actual_k, -1)
        nebor_t_p = nebor_t_p.view(b, actual_k, -1)
        
        # 扩展维度以便计算差异
        l_s = torch.unsqueeze(logits_s, 1)  # [B, 1, M]
        l_t = torch.unsqueeze(logits_t, 1)  # [B, 1, M]
        
        # 计算样本与邻居的logits差异
        res_s_diff = l_s - nebor_s_p  # [B, K, M]
        res_t_diff = l_t - nebor_t_p  # [B, K, M]
        
        # 使用JS散度计算损失
        loss_nebor_res = self.js_div(res_s_diff, res_t_diff) / (b * actual_k)
        
        return loss_nebor_res

    def js_div(self, q, p):
        """
        Jensen-Shannon散度
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), 其中 M = (P+Q)/2
        
        Args:
            q: 学生的响应差异 [B, K, M]
            p: 教师的响应差异 [B, K, M]
            
        Returns:
            loss: JS散度损失
        """
        # 在最后一维上进行softmax（类别维度）
        q = F.softmax(q, dim=2)
        p = F.softmax(p, dim=2)
        
        # 计算中间分布M
        log_mean = ((q + p) / 2).log()
        
        # JS散度 = 0.5 * [KL(q||M) + KL(p||M)]
        loss = (F.kl_div(log_mean, q, reduction='sum') + 
                F.kl_div(log_mean, p, reduction='sum')) / 2
        
        return loss

    def forward(self, logits_s, logits_t):
        """
        前向传播计算NLRD损失
        
        Args:
            logits_s: 学生网络的logits [batch_size, num_classes]
            logits_t: 教师网络的logits [batch_size, num_classes]
            
        Returns:
            loss: NLRD损失值
        """
        b = logits_t.size(0)
        
        # Step 1: 基于教师logits选择K近邻
        sort, idx = self.angle(logits_t)
        nebor_idx = idx[:, 1:self.k+1]  # 排除自己（索引0），选择接下来的k个邻居
        
        # Step 2: 计算邻域响应关系损失
        loss = self.lambda1 * self.compute_relation_loss(logits_s, logits_t, b, nebor_idx)
        
        return loss
