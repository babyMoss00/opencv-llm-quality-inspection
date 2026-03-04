"""
Quick Start: 工业缺陷检测的小样本学习
=====================================

这个脚本展示了最实用的工作流:
1. 用预训练ResNet50提取特征
2. 只需几张OK/NG样本
3. 原型网络 或 KNN 快速分类

适用场景:
  - 新产品线的缺陷检测 (只有少量样本)
  - 胶水检测、异物检测、划痕检测等
  - 需要快速部署的质检场景

使用方法:
  1. 准备数据: 按类别放入文件夹
     data/
     ├── OK/          (正常样本, 3-10张)
     ├── NG_scratch/  (划痕, 3-10张)
     └── NG_dent/     (凹陷, 3-10张)
  
  2. 运行:
     python quick_start_industrial.py --data_dir ./your_data
     
  3. 部署: 使用IndustrialFewShotDetector类集成到PyQt5应用中
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time
import numpy as np


class IndustrialFewShotDetector:
    """
    工业级小样本缺陷检测器
    
    设计原则:
      - 开箱即用: 只需提供几张参考图就能工作
      - 可增量更新: 随时添加新的参考样本
      - 推理快速: 特征提取 + 最近邻, 适合产线实时检测
      - 可导出: 支持保存/加载特征库
    
    Example:
        detector = IndustrialFewShotDetector()
        
        # 注册参考样本
        detector.register_class("OK", ["ok_1.jpg", "ok_2.jpg", "ok_3.jpg"])
        detector.register_class("NG_scratch", ["ng_1.jpg", "ng_2.jpg"])
        
        # 检测
        result = detector.detect("test_image.jpg")
        print(f"结果: {result['class']} (置信度: {result['confidence']:.1%})")
    """
    
    def __init__(self, backbone: str = "resnet50", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_name = backbone
        
        # 加载预训练模型
        print(f"🔧 加载预训练模型: {backbone}...")
        self.model, self.feature_dim = self._load_backbone(backbone)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        
        # 特征库
        self.feature_bank: Dict[str, torch.Tensor] = {}  # 类别 → [N, D]特征矩阵
        self.prototypes: Dict[str, torch.Tensor] = {}     # 类别 → [D]原型向量
        
        print(f"   ✅ 就绪 (特征维度: {self.feature_dim}, 设备: {self.device})")
    
    def _load_backbone(self, backbone: str):
        """加载预训练backbone"""
        if backbone == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = 2048
        elif backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 去掉最后的FC层
        modules = list(model.children())[:-1]
        model = torch.nn.Sequential(*modules)
        return model, feature_dim
    
    @torch.no_grad()
    def _extract_feature(self, image: Image.Image) -> torch.Tensor:
        """提取单张图片的特征向量"""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        feature = self.model(tensor).view(1, -1)
        feature = F.normalize(feature, p=2, dim=-1)
        return feature.squeeze(0)  # [D]
    
    @torch.no_grad()
    def _extract_features_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """批量提取特征"""
        tensors = torch.stack([self.transform(img.convert("RGB")) for img in images])
        tensors = tensors.to(self.device)
        features = self.model(tensors).view(len(images), -1)
        features = F.normalize(features, p=2, dim=-1)
        return features  # [N, D]
    
    def register_class(self, class_name: str, 
                       image_sources: List, 
                       append: bool = True):
        """
        注册一个类别的参考样本
        
        Args:
            class_name: 类别名称 (如 "OK", "NG_scratch")
            image_sources: PIL Image列表, 或图片路径列表
            append: True=追加到已有样本, False=替换
        """
        images = []
        for src in image_sources:
            if isinstance(src, (str, Path)):
                images.append(Image.open(src).convert("RGB"))
            elif isinstance(src, Image.Image):
                images.append(src)
            else:
                raise ValueError(f"Unsupported image source type: {type(src)}")
        
        features = self._extract_features_batch(images)
        
        if append and class_name in self.feature_bank:
            self.feature_bank[class_name] = torch.cat([
                self.feature_bank[class_name], features
            ], dim=0)
        else:
            self.feature_bank[class_name] = features
        
        # 更新原型
        self.prototypes[class_name] = self.feature_bank[class_name].mean(dim=0)
        self.prototypes[class_name] = F.normalize(self.prototypes[class_name], p=2, dim=0)
        
        n = self.feature_bank[class_name].size(0)
        print(f"   📌 已注册类别 '{class_name}': {n} 个样本")
    
    def register_from_directory(self, data_dir: str):
        """
        从目录结构自动注册所有类别
        
        目录结构:
            data_dir/
            ├── OK/
            │   ├── img1.jpg
            │   └── img2.jpg
            └── NG_scratch/
                ├── img1.jpg
                └── img2.jpg
        """
        data_dir = Path(data_dir)
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        
        for cls_dir in sorted(data_dir.iterdir()):
            if cls_dir.is_dir() and not cls_dir.name.startswith("."):
                images = [
                    str(f) for f in sorted(cls_dir.iterdir())
                    if f.suffix.lower() in valid_ext
                ]
                if images:
                    self.register_class(cls_dir.name, images, append=False)
        
        print(f"\n✅ 共注册 {len(self.feature_bank)} 个类别")
    
    def detect(self, image_source, method: str = "prototype") -> dict:
        """
        检测单张图片
        
        Args:
            image_source: PIL Image 或 图片路径
            method: "prototype" (推荐) 或 "knn"
            
        Returns:
            {
                "class": "OK",
                "confidence": 0.95,
                "all_scores": {"OK": 0.95, "NG_scratch": 0.03, "NG_dent": 0.02},
                "inference_time_ms": 12.3
            }
        """
        if not self.feature_bank:
            raise RuntimeError("请先注册参考样本! 调用 register_class() 或 register_from_directory()")
        
        start_time = time.time()
        
        # 加载图片
        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source).convert("RGB")
        else:
            image = image_source
        
        # 提取特征
        query_feature = self._extract_feature(image)  # [D]
        
        # 分类
        if method == "prototype":
            scores = self._classify_prototype(query_feature)
        elif method == "knn":
            scores = self._classify_knn(query_feature, k=3)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 整理结果
        pred_class = max(scores, key=scores.get)
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "class": pred_class,
            "confidence": scores[pred_class],
            "all_scores": scores,
            "inference_time_ms": round(inference_time, 2),
        }
    
    def _classify_prototype(self, query_feature: torch.Tensor) -> Dict[str, float]:
        """原型网络分类"""
        similarities = {}
        for cls_name, prototype in self.prototypes.items():
            sim = torch.dot(query_feature, prototype).item()
            similarities[cls_name] = sim
        
        # Softmax归一化
        values = list(similarities.values())
        exp_values = [np.exp(v * 10) for v in values]  # temperature=10
        total = sum(exp_values)
        
        return {
            cls: exp_v / total 
            for cls, exp_v in zip(similarities.keys(), exp_values)
        }
    
    def _classify_knn(self, query_feature: torch.Tensor, k: int = 3) -> Dict[str, float]:
        """KNN分类"""
        all_sims = []
        all_labels = []
        
        for cls_name, features in self.feature_bank.items():
            sims = torch.mv(features, query_feature)  # [N]
            for sim in sims:
                all_sims.append(sim.item())
                all_labels.append(cls_name)
        
        # Top-K
        indices = np.argsort(all_sims)[-k:]
        votes = {}
        for idx in indices:
            cls = all_labels[idx]
            votes[cls] = votes.get(cls, 0) + all_sims[idx]
        
        # 归一化
        total = sum(votes.values())
        return {cls: v / total for cls, v in votes.items()}
    
    def save_feature_bank(self, path: str):
        """保存特征库 (用于部署)"""
        data = {
            cls: features.cpu().numpy().tolist()
            for cls, features in self.feature_bank.items()
        }
        with open(path, "w") as f:
            json.dump({
                "backbone": self.backbone_name,
                "feature_dim": self.feature_dim,
                "classes": data,
            }, f)
        print(f"💾 特征库已保存: {path}")
    
    def load_feature_bank(self, path: str):
        """加载特征库"""
        with open(path, "r") as f:
            data = json.load(f)
        
        for cls, features in data["classes"].items():
            self.feature_bank[cls] = torch.tensor(features, device=self.device)
            self.prototypes[cls] = F.normalize(
                self.feature_bank[cls].mean(dim=0), p=2, dim=0
            )
        
        print(f"📂 特征库已加载: {len(self.feature_bank)} 个类别")
    
    def status(self):
        """打印当前状态"""
        print(f"\n{'='*50}")
        print(f"🏭 工业小样本检测器状态")
        print(f"{'='*50}")
        print(f"  模型: {self.backbone_name} ({self.feature_dim}D)")
        print(f"  设备: {self.device}")
        print(f"  注册类别: {len(self.feature_bank)}")
        for cls, features in self.feature_bank.items():
            print(f"    - {cls}: {features.size(0)} 个参考样本")
        print(f"{'='*50}\n")


# ============================================================
# PyQt5 集成示例 (可选)
# ============================================================

PYQT5_INTEGRATION_EXAMPLE = """
# =============================================
# PyQt5 集成示例 (适合你的工作场景)
# =============================================

from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal

class DetectionWorker(QThread):
    '''后台检测线程'''
    result_ready = pyqtSignal(dict)
    
    def __init__(self, detector, image_path):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
    
    def run(self):
        result = self.detector.detect(self.image_path)
        self.result_ready.emit(result)

class InspectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = IndustrialFewShotDetector(backbone="resnet50")
        
        # 注册参考样本 (只需几张!)
        self.detector.register_from_directory("./reference_images")
        
        # ... 设置UI ...
    
    def on_capture(self, image):
        '''相机采集回调'''
        result = self.detector.detect(image)
        
        if result["class"] == "OK":
            self.status_label.setText(f"✅ OK ({result['confidence']:.0%})")
            self.status_label.setStyleSheet("color: green")
        else:
            self.status_label.setText(
                f"❌ {result['class']} ({result['confidence']:.0%})"
            )
            self.status_label.setStyleSheet("color: red")
"""


# ============================================================
# Demo
# ============================================================

def demo():
    """快速Demo: 用生成的图形模拟工业检测"""
    import random
    
    print("=" * 60)
    print("🏭 工业缺陷检测 - 小样本学习 Quick Start")
    print("=" * 60)
    
    # 1. 创建模拟数据
    print("\n📦 创建模拟检测数据...")
    demo_dir = Path("./demo_industrial")
    
    categories = {
        "OK": (200, 200, 200),          # 灰色 = 正常
        "NG_dark_spot": (50, 50, 50),    # 黑点 = 缺陷
        "NG_bright_spot": (255, 255, 200) # 亮斑 = 缺陷
    }
    
    for cls_name, base_color in categories.items():
        cls_dir = demo_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(8):
            img = Image.new("RGB", (128, 128))
            pixels = img.load()
            for x in range(128):
                for y in range(128):
                    if cls_name == "OK":
                        noise = random.randint(-20, 20)
                        c = tuple(max(0, min(255, base_color[j] + noise)) for j in range(3))
                    elif cls_name == "NG_dark_spot":
                        cx, cy = 64 + random.randint(-20, 20), 64 + random.randint(-20, 20)
                        dist = ((x-cx)**2 + (y-cy)**2) ** 0.5
                        if dist < 20:
                            c = tuple(max(0, base_color[j] + random.randint(-10, 10)) for j in range(3))
                        else:
                            c = (200+random.randint(-15,15),)*3
                    else:  # bright spot
                        cx, cy = 64 + random.randint(-20, 20), 64 + random.randint(-20, 20)
                        dist = ((x-cx)**2 + (y-cy)**2) ** 0.5
                        if dist < 25:
                            c = tuple(min(255, base_color[j] + random.randint(-10, 10)) for j in range(3))
                        else:
                            c = (200+random.randint(-15,15),)*3
                    pixels[x, y] = c
            img.save(cls_dir / f"{cls_name}_{i:02d}.png")
    
    print("   ✅ 模拟数据已创建")
    
    # 2. 初始化检测器
    print("\n🔧 初始化检测器...")
    detector = IndustrialFewShotDetector(backbone="resnet50")
    
    # 3. 注册参考样本 (只用前3张!)
    print("\n📌 注册参考样本 (每类仅3张)...")
    for cls_name in categories:
        cls_dir = demo_dir / cls_name
        images = sorted(cls_dir.glob("*.png"))[:3]  # 只用3张!
        detector.register_class(cls_name, [str(p) for p in images])
    
    detector.status()
    
    # 4. 在剩余样本上测试
    print("🔬 测试检测效果...")
    print("-" * 50)
    
    correct = 0
    total = 0
    
    for cls_name in categories:
        cls_dir = demo_dir / cls_name
        test_images = sorted(cls_dir.glob("*.png"))[3:]  # 后5张作为测试
        
        for img_path in test_images:
            result = detector.detect(str(img_path))
            is_correct = result["class"] == cls_name
            correct += int(is_correct)
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} {img_path.name:25s} → "
                  f"预测: {result['class']:20s} "
                  f"({result['confidence']:5.1%}) "
                  f"耗时: {result['inference_time_ms']:.1f}ms")
    
    accuracy = correct / total * 100
    print(f"\n{'='*50}")
    print(f"📊 总体准确率: {correct}/{total} = {accuracy:.1f}%")
    print(f"   每类仅用 3 张参考样本!")
    print(f"{'='*50}")
    
    # 5. 保存特征库
    detector.save_feature_bank("./feature_bank.json")
    
    print(f"\n💡 在实际项目中:")
    print(f"   1. 将 OK/NG 参考图放入对应文件夹")
    print(f"   2. 调用 detector.register_from_directory()")
    print(f"   3. 调用 detector.detect(image) 进行检测")
    print(f"   4. 集成到 PyQt5 界面中即可部署")


if __name__ == "__main__":
    import sys
    if "--help" in sys.argv:
        print(__doc__)
    else:
        demo()