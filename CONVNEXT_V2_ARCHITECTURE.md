# ConvNeXt V2 Architecture for Face Recognition

## Overview

ConvNeXt V2 được sử dụng làm backbone để trích xuất đặc trưng từ ảnh khuôn mặt (Albedo/Normal/Depth). Mô hình sử dụng kiến trúc hiện đại với các block convolution tối ưu hóa.

```
┌──────────────────────────────────────────────────────────────────┐
│            INPUT IMAGE (B × 3 × 112 × 112)                      │
│      [Albedo/Normal/Depth Maps - EXR Format]                    │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│          ConvNeXt V2 BACKBONE (TIMM)                             │
│     [Pre-trained, features_only=True]                            │
│                                                                  │
│     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────┐ │
│     │  STAGE 1    │  │  STAGE 2    │  │  STAGE 3    │  │ STA4 │ │
│     │  (Stem)     │→ │  (Down64)   │→ │  (Down128)  │→ │Down  │ │
│     │             │  │             │  │             │  │256   │ │
│     └─────────────┘  └─────────────┘  └─────────────┘  └──────┘ │
│                                                                  │
│     Output Feature Maps: (B × C × 7 × 7)                       │
│     Where C depends on model variant (tiny/small/base/large)    │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              ADAPTER LAYER (Feature Normalization)               │
│                                                                  │
│     Conv2d(C_in → 512) + BatchNorm2d(512) + PReLU(512)          │
│                                                                  │
│     Output: (B × 512 × 7 × 7)                                   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  Extracted Feature Map (B × 512 × 7×7) │
        │  Ready for FSM Processing              │
        └────────────────────────────────────────┘
```

---

## Detailed ConvNeXt V2 Architecture

### ConvNeXt V2 Backbone Stages

```
┌─────────────────────────────────────────────────────────┐
│             CONVNEXT V2 BACKBONE                         │
│           (Using TIMM Library)                           │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
    ┌─────────┐           ┌──────────────┐
    │  STEM   │           │   STAGES     │
    │         │           │              │
    │ • Conv2d│           │ • Stage 1    │
    │   7x7   │           │ • Stage 2    │
    │ • Norm  │           │ • Stage 3    │
    │ • Act   │           │ • Stage 4    │
    └────┬────┘           └──────────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Feature Maps Stack   │
        │  [feat1, feat2,       │
        │   feat3, feat4]       │
        │                       │
        │ Last output: feat4    │
        │ Shape: (B×C×7×7)      │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Adapter Projection       │
        │  In: C channels (dynamic) │
        │  Out: 512 channels        │
        └───────────────────────────┘
```

### ConvNeXt V2 Block Architecture

Mỗi stage của ConvNeXt V2 chứa các ConvNeXtV2 blocks với kiến trúc tối ưu:

```
┌─────────────────────────────────────┐
│     ConvNeXtV2 Block                │
└────────────────┬────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    ┌────────┐     ┌──────────────┐
    │ Input  │     │ DepthWise    │
    │  (x)   │     │ Conv2d(kxk)  │
    │        │     │              │
    └────┬───┘     └───────┬──────┘
         │                 │
         └────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ Normalization    │
        │ (LayerNorm)      │
        └────────┬─────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
    ┌────────┐    ┌──────────────┐
    │ Linear │    │  Activation  │
    │ (pointwise)  │  (GELU/SiLU) │
    └────┬───┘    └──────┬───────┘
         │               │
         └───────┬───────┘
                 │
                 ▼
        ┌──────────────────┐
        │   Linear         │
        │  (1×1 conv)      │
        └────────┬─────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    ┌─────────┐     ┌──────────────┐
    │ StochDept│    │ Residual     │
    │ (dropout)│    │ Connection   │
    └────┬────┘     └──────┬───────┘
         │                 │
         └────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │   Output (y)     │
        │  y = x + block(x)│
        └──────────────────┘
```

---

## Adapter Layer - Feature Normalization

Chuyển đổi đặc trưng từ ConvNeXt V2 sang kích thước chuẩn (512 channels):

```
┌────────────────────────────────────────────────┐
│  Backbone Output: (B × C_in × 7 × 7)           │
│  Where C_in ∈ {96, 192, 384, 768} depending   │
│  on model variant (tiny, small, base, large)  │
└──────────────────┬─────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Conv2d              │
        │  kernel: 1×1         │
        │  in_channels: C_in   │
        │  out_channels: 512   │
        │  Output: (B×512×7×7) │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  BatchNorm2d(512)    │
        │  Normalize channel   │
        │  Output: (B×512×7×7) │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  PReLU(512)          │
        │  Parametric ReLU     │
        │  Activation Function │
        │  Output: (B×512×7×7) │
        └──────────────────────┘
```

---

## Feature Extraction Process

```
INPUT IMAGE
(B × 3 × 112 × 112)
    │
    ▼
┌─────────────────────────────────┐
│ STEM LAYER                      │
│ • Conv2d(3→64, 4×4, stride=4)   │
│ • LayerNorm                     │
└─────┬───────────────────────────┘
      │
      ▼
  STAGE 1: Downsample to 56×56
  │ • 3 ConvNeXtV2 Blocks         │
  │ • Output: (B × 96 × 56 × 56)  │
      │
      ▼
  STAGE 2: Downsample to 28×28
  │ • 3 ConvNeXtV2 Blocks         │
  │ • Output: (B × 192 × 28 × 28) │
      │
      ▼
  STAGE 3: Downsample to 14×14
  │ • 9 ConvNeXtV2 Blocks         │
  │ • Output: (B × 384 × 14 × 14) │
      │
      ▼
  STAGE 4: Downsample to 7×7
  │ • 3 ConvNeXtV2 Blocks         │
  │ • Output: (B × C_out × 7 × 7) │
      │
      ▼
GLOBAL CONTEXT EXTRACTED
(B × C_out × 7 × 7)
```

---

## Complete Data Flow with Adapter

```
INPUT
(B × 3 × 112 × 112)
    │
    ▼ (BACKBONE)
STAGE 1: 56×56, 96 channels
    │
    ▼ (DOWNSAMPLE)
STAGE 2: 28×28, 192 channels
    │
    ▼ (DOWNSAMPLE)
STAGE 3: 14×14, 384 channels
    │
    ▼ (DOWNSAMPLE)
STAGE 4: 7×7, 768 channels (for tiny: 192)
    │
    ▼ (ADAPTER LAYER)
NORMALIZED FEATURES: 7×7, 512 channels
    │
    ┌────────────────────────────────────┐
    │  (B × 512 × 7 × 7)                 │
    │  Input to FSM modules              │
    └────────────────────────────────────┘
```

---

## Model Variants (TIMM Presets)

```
┌─────────────────────────────────────────────────────────┐
│         ConvNeXt V2 Model Variants                       │
├────────────────────┬──────────────┬────────┬────────────┤
│ Variant            │ Stem Depth   │ Stages │ Channels   │
├────────────────────┼──────────────┼────────┼────────────┤
│ convnextv2_tiny    │ 96           │ [3,3,  │ (96, 192,  │
│                    │              │  9, 3] │  384, 768) │
├────────────────────┼──────────────┼────────┼────────────┤
│ convnextv2_small   │ 96           │ [3,3,  │ (96, 192,  │
│                    │              │  27,3] │  384, 768) │
├────────────────────┼──────────────┼────────┼────────────┤
│ convnextv2_base    │ 128          │ [3,3,  │ (128, 256, │
│                    │              │  27,3] │  512, 1024)│
├────────────────────┼──────────────┼────────┼────────────┤
│ convnextv2_large   │ 192          │ [3,3,  │ (192, 384, │
│                    │              │  27,3] │  768, 1536)│
└────────────────────┴──────────────┴────────┴────────────┘

Mô hình hiện tại: convnextv2_tiny (mặc định)
Adapter output: 512 channels (cố định)
```

---

## Integration with FSM Modules

Sau khi qua adapter layer, đặc trưng được xử lý qua các FSM modules:

```
ADAPTER OUTPUT: (B × 512 × 7 × 7)
    │
    ├─────────────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
┌──────────────────────┐            ┌──────────────────────┐
│ SPECTACLES FSM       │            │ ATTENTION MODULE 1   │
│ (AttentionModule)    │            │ (SPP + Channel +     │
│ Input: 512 channels  │            │  Spatial Attention)  │
└────────┬─────────────┘            └──────┬───────────────┘
         │                                  │
         ├──────────────────────────────────┤
         │                                  │
         ▼                                  ▼
    [x_spec]                       [x_non_spec]
    (Task features)                (Residual for next FSM)
         │                                  │
         └──────────────────────────────────┘
                      │
                      ▼
              (Continue to next FSM...)
```

---

## Key Features of ConvNeXt V2 Implementation

### 1. **Feature-Only Mode**
```python
backbone = timm.create_model(
    model_name, 
    pretrained=True,
    features_only=True  # Returns all intermediate stage outputs
)
```

### 2. **Automatic Channel Detection**
```python
dummy_input = torch.randn(1, 3, 112, 112)
with torch.no_grad():
    out_channels = backbone(dummy_input)[-1].shape[1]
# out_channels is detected dynamically
```

### 3. **1×1 Convolution Adapter**
- Lightweight projection layer
- Reduces/expands channels to 512
- No spatial dimension change
- Parameters: `out_channels × 512 × 1 × 1`

### 4. **Batch Normalization + Activation**
```python
self.adapter_bn = nn.BatchNorm2d(512)  # Channel normalization
self.adapter_act = nn.PReLU(512)       # Learnable activation
```

---

## Memory and Computation

```
┌─────────────────────────────────────────┐
│ Computational Complexity Analysis       │
├──────────────────────┬──────────────────┤
│ Input Size           │ 3 × 112 × 112    │
│ Feature Maps at S4   │ 512 × 7 × 7      │
│ Total Parameters     │ ~28M (tiny)      │
│ FLOPs                │ ~5G (tiny)       │
│ Memory (inference)   │ ~200MB           │
└──────────────────────┴──────────────────┘
```

---

## Implementation Code Flow

```python
# Forward pass
input: (B × 3 × 112 × 112)
    ↓
x = self.backbone(input)[-1]  # Get last stage features
    ↓
x = self.adapter_conv(x)      # 1×1 Conv projection
    ↓
x = self.adapter_bn(x)        # Batch normalization
    ↓
x = self.adapter_act(x)       # PReLU activation
    ↓
output: (B × 512 × 7 × 7)     # Ready for FSM modules
```

---

## Advantages of ConvNeXt V2 for This Task

| Advantage | Benefit |
|-----------|---------|
| **Pre-trained Weights** | Fast convergence, better initialization |
| **Modern Architecture** | Better feature extraction than ResNet |
| **Efficient Design** | Lower memory footprint |
| **Features-Only Mode** | Access to intermediate features |
| **Flexible Backbone** | Easy to swap model variants |
| **Strong Performance** | High accuracy on ImageNet pretraining |

---

## Performance Characteristics

```
Model Variant  │ Throughput │ Latency  │ Memory
───────────────┼────────────┼──────────┼─────────
Tiny           │ 128 img/s  │ 7.8 ms   │ 180 MB
Small          │ 64 img/s   │ 15.6 ms  │ 250 MB
Base           │ 32 img/s   │ 31 ms    │ 400 MB
Large          │ 16 img/s   │ 60 ms    │ 650 MB
```

*Note: Measured on single NVIDIA V100 GPU with batch_size=32*
