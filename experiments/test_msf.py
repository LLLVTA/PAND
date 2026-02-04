import torch
from src.models.components.campus import ModifiedResNet
import torchvision.models as models

print("=" * 60)
print("测试多尺度特征融合(MSF)模块")
print("=" * 60)

# 创建ResNet18
origin_model = models.resnet18(pretrained=False)

# 测试1: Baseline模式
print("\n[测试1] Baseline模式 (use_multiscale=False)")
model_baseline = ModifiedResNet(origin_model, classnum=37, use_multiscale=False)
x = torch.randn(4, 3, 224, 224)
hidden, logits = model_baseline(x)
print(f"✅ 输入shape: {x.shape}")
print(f"✅ hidden_features shape: {hidden.shape}")
print(f"✅ logits shape: {logits.shape}")
assert hidden.shape == (4, 512), f"❌ Baseline hidden shape错误: {hidden.shape}"
assert logits.shape == (4, 37), f"❌ Baseline logits shape错误: {logits.shape}"

# 测试2: MSF模式
print("\n[测试2] MSF模式 (use_multiscale=True)")
origin_model2 = models.resnet18(pretrained=False)
model_msf = ModifiedResNet(origin_model2, classnum=37, use_multiscale=True)
hidden_msf, logits_msf = model_msf(x)
print(f"✅ 输入shape: {x.shape}")
print(f"✅ hidden_features shape: {hidden_msf.shape}")
print(f"✅ logits shape: {logits_msf.shape}")
assert hidden_msf.shape == (4, 512), f"❌ MSF hidden shape错误: {hidden_msf.shape}"
assert logits_msf.shape == (4, 37), f"❌ MSF logits shape错误: {logits_msf.shape}"

# 测试3: 参数量对比
params_baseline = sum(p.numel() for p in model_baseline.parameters())
params_msf = sum(p.numel() for p in model_msf.parameters())
print(f"\n[测试3] 参数量对比")
print(f"✅ Baseline参数量: {params_baseline/1e6:.2f}M")
print(f"✅ MSF参数量: {params_msf/1e6:.2f}M")
print(f"✅ 增加: {(params_msf-params_baseline)/1e6:.2f}M ({(params_msf/params_baseline-1)*100:.1f}%)")

# 测试4: Hook是否正确捕获
print(f"\n[测试4] Hook捕获验证")
print(f"✅ 捕获的层: {list(model_msf.intermediate_features.keys())}")
print(f"✅ layer2 shape: {model_msf.intermediate_features['layer2'].shape}")
print(f"✅ layer3 shape: {model_msf.intermediate_features['layer3'].shape}")
print(f"✅ layer4 shape: {model_msf.intermediate_features['layer4'].shape}")

print("\n" + "=" * 60)
print("✅ 所有测试通过!")
print("=" * 60)
