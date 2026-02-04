#!/usr/bin/env python
"""测试多层蒸馏的forward流程"""

import torch
from src.models.components.campus import TeacherStudent, ModifiedResNet
from types import SimpleNamespace
import torchvision.models as models

def test_modified_resnet():
    """测试ModifiedResNet是否正确输出layer3特征"""
    print("=" * 50)
    print("测试 ModifiedResNet")
    print("=" * 50)
    
    # 创建ResNet18
    resnet = models.resnet18(pretrained=False)
    modified_resnet = ModifiedResNet(resnet, classnum=200)
    
    # 测试forward
    x = torch.randn(2, 3, 224, 224)
    hidden_features, out, layer3_features = modified_resnet(x)
    
    print(f"输入形状: {x.shape}")
    print(f"hidden_features形状: {hidden_features.shape}  (期望: [2, 512])")
    print(f"logits形状: {out.shape}  (期望: [2, 200])")
    print(f"layer3_features形状: {layer3_features.shape}  (期望: [2, 512])")
    
    assert hidden_features.shape == (2, 512), f"hidden_features形状错误: {hidden_features.shape}"
    assert out.shape == (2, 200), f"logits形状错误: {out.shape}"
    assert layer3_features.shape == (2, 512), f"layer3_features形状错误: {layer3_features.shape}"
    
    print("✅ ModifiedResNet测试通过!\n")

def test_teacher_student():
    """测试TeacherStudent完整流程"""
    print("=" * 50)
    print("测试 TeacherStudent (不加载teacher)")
    print("=" * 50)
    
    # 创建简化的配置
    teacher_config = SimpleNamespace(arch='convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg_soup')
    student_config = SimpleNamespace(arch='resnet18')
    
    # 模拟attributes
    attributes = SimpleNamespace(
        class_num=200,
        prompt_tmpl="a photo of a {}",
        classes={i: f"class_{i}" for i in range(200)}
    )
    
    # 创建模型(不加载teacher避免下载)
    model = TeacherStudent(
        teacher=teacher_config,
        student=student_config,
        data_attributes=attributes,
        use_teacher=False  # 暂时不加载teacher
    )
    
    # 测试forward
    x = torch.randn(2, 3, 224, 224)
    
    # 不使用teacher时应该返回out
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print("✅ TeacherStudent测试通过!\n")

if __name__ == "__main__":
    test_modified_resnet()
    test_teacher_student()
    print("=" * 50)
    print("所有测试通过! 🎉")
    print("=" * 50)
