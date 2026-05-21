# MTL Face Recognition Network Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (3×112×112)                      │
│              [Albedo/Normal/Depth Map - EXR Format]             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ConvNeXt V2 BACKBONE                           │
│           (Pre-trained, features_only=True)                     │
│                                                                 │
│     [Stage1] → [Stage2] → [Stage3] → [Stage4]                  │
│                                                                 │
│              Output: Feature Maps (C×H×W)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ADAPTER LAYER                                 │
│                                                                 │
│    Conv2d(out_channels → 512) + BatchNorm + PReLU              │
│                                                                 │
│              Output: (B × 512 × H × W)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌──────────────────────┐              ┌──────────────────────┐
│  FSM1: SPECTACLES    │              │  ATTENTION MODULE 1  │
│ AttentionModule      │              │                      │
└──────────────────────┘              │  • SPP (3×3 pools)   │
        │                             │  • Channel Attention │
   ┌────┴────┐                        │  • Spatial Attention │
   │          │                       └──────────────────────┘
   ▼          ▼
[x_spec] [x_non_spec]
   │          │
   │          └──────────────────────┐
   │                                 │
   │          ┌──────────────────────┴─────────────┐
   │          │                                    │
   │          ▼                                    ▼
   │   ┌──────────────────────┐         ┌──────────────────────┐
   │   │  FSM2: FACIAL HAIR   │         │  ATTENTION MODULE 2  │
   │   │ AttentionModule      │         │                      │
   │   └──────────────────────┘         │  • SPP (3×3 pools)   │
   │          │                         │  • Channel Attention │
   │     ┌────┴────┐                    │  • Spatial Attention │
   │     │          │                   └──────────────────────┘
   │     ▼          ▼
   │  [x_fh] [x_non_fh]
   │     │          │
   │     │          └──────────────────────┐
   │     │                                 │
   │     │          ┌──────────────────────┴─────────────┐
   │     │          │                                    │
   │     │          ▼                                    ▼
   │     │   ┌──────────────────────┐         ┌──────────────────────┐
   │     │   │  FSM3: EMOTION       │         │  ATTENTION MODULE 3  │
   │     │   │ AttentionModule      │         │                      │
   │     │   └──────────────────────┘         │  • SPP (3×3 pools)   │
   │     │          │                         │  • Channel Attention │
   │     │     ┌────┴────┐                    │  • Spatial Attention │
   │     │     │          │                   └──────────────────────┘
   │     │     ▼          ▼
   │     │  [x_emot] [x_non_emot]
   │     │     │          │
   │     │     │          └──────────────────────┐
   │     │     │                                 │
   │     │     │          ┌──────────────────────┴─────────────┐
   │     │     │          │                                    │
   │     │     │          ▼                                    ▼
   │     │     │   ┌──────────────────────┐         ┌──────────────────────┐
   │     │     │   │  FSM4: POSE          │         │  ATTENTION MODULE 4  │
   │     │     │   │ AttentionModule      │         │                      │
   │     │     │   └──────────────────────┘         │  • SPP (3×3 pools)   │
   │     │     │          │                         │  • Channel Attention │
   │     │     │     ┌────┴────┐                    │  • Spatial Attention │
   │     │     │     │          │                   └──────────────────────┘
   │     │     │     ▼          ▼
   │     │     │  [x_pose] [x_non_pose]
   │     │     │     │          │
   │     │     │     │          └──────────────────────┐
   │     │     │     │                                 │
   │     │     │     │          ┌──────────────────────┴─────────────┐
   │     │     │     │          │                                    │
   │     │     │     │          ▼                                    ▼
   │     │     │     │   ┌──────────────────────┐         ┌──────────────────────┐
   │     │     │     │   │  FSM5: GENDER       │         │  ATTENTION MODULE 5  │
   │     │     │     │   │ AttentionModule     │         │                      │
   │     │     │     │   └──────────────────────┘         │  • SPP (3×3 pools)   │
   │     │     │     │          │                         │  • Channel Attention │
   │     │     │     │     ┌────┴────┐                    │  • Spatial Attention │
   │     │     │     │     │          │                   └──────────────────────┘
   │     │     │     │     ▼          ▼
   ▼     ▼     ▼     ▼  [x_gender]  [x_id]
[x_spec][x_fh][x_emot][x_pose][x_gender][x_id]
   │     │     │     │     │          │
   │     │     │     │     │          │
   ├─────┼─────┼─────┼─────┤          │
   │     │     │     │     │          │
   ▼     ▼     ▼     ▼     ▼          ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌────────┐
│SPEC  ││FACIAL││POSE  ││EMOTION││GENDER││  ID    │
│HEAD  ││HAIR  ││HEAD  ││HEAD  ││HEAD  ││HEAD   │
└──────┘└──────┘└──────┘└──────┘└──────┘└────────┘
   │     │     │     │     │          │
   ▼     ▼     ▼     ▼     ▼          ▼
[logits₁][logits₂][logits₃][logits₄][logits₅]  [ID_logits]
                                        [ID_embedding]
   │     │     │     │     │          │
   └─────┼─────┼─────┼─────┴──────────┘
         │ (For Domain Adaptation)    │
         ▼                            ▼
    ┌─────────────────────────────────────┐
    │  GRL (Gradient Reverse Layers)      │
    │  (Applied to x_non_* features)      │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  Domain Adversarial Heads           │
    │  (da_gender, da_emotion, etc.)      │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  DA_LOGITS (for 5 auxiliary tasks)  │
    └─────────────────────────────────────┘
         │
         └─────────────────────────────────→ OUTPUT
```

---

## Detailed Component Breakdown

### 1. **FSM (Facial Semantic Module) - AttentionModule**

```
┌────────────────────────────────────────────┐
│   INPUT FEATURE MAP (512 × H × W)          │
└──────────────────┬───────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────────┐ ┌──────────────────┐
│  AVG SPP Module  │ │  MAX SPP Module  │
│  • Pool 1×1      │ │  • Pool 1×1      │
│  • Pool 2×2      │ │  • Pool 2×2      │
│  • Pool 3×3      │ │  • Pool 3×3      │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         └────────┬───────────┘
                  │
                  ▼
        ┌──────────────────┐
        │  CONCATENATE     │
        │  Output: C×36    │
        └────────┬─────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
 ┌──────────────┐  ┌──────────────────────┐
 │  CHANNEL     │  │  SPATIAL ATTENTION   │
 │  ATTENTION   │  │                      │
 │  • Conv1x1   │  │  • Max pool (1,3)    │
 │  • ReLU      │  │  • Mean pool (1,3)   │
 │  • Conv1x1   │  │  • Conv kernel 7×7   │
 │  • BatchNorm │  │  • Sigmoid           │
 │  • Sigmoid   │  │                      │
 └──────┬───────┘  └──────────┬───────────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  Feature Decomposition   │
        │                          │
        │  x_task = x * channel    │
        │           * 0.5          │
        │           + x * spatial  │
        │           * 0.5          │
        │                          │
        │  x_non_task = x - x_task │
        └──────────┬───────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
    [x_task]           [x_non_task]
    (Task-specific)    (For next FSM)
```

---

## Data Flow During Training

```
┌──────────────────────────────────────┐
│   BATCH: (X, y)                      │
│   X: (B × 3 × 112 × 112)             │
│   y: (B × 6) [id, gender, spectacles,│
│        facial_hair, pose, emotion]   │
└──────────────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  FORWARD PASS        │
        │  model(X) → logits   │
        └──────────┬───────────┘
                   │
        ┌──────────┴──────────────────────┐
        │                                 │
        ▼                                 ▼
    ┌────────────┐            ┌─────────────────────┐
    │  Logits    │            │  DA Logits (GRL)    │
    │  (6 tasks) │            │  (5 auxiliary tasks)│
    └────────────┘            └─────────────────────┘
        │                                 │
        └─────────────┬───────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │  LOSS CALCULATION        │
        │                          │
        │  L_id (main task)        │
        │  L_gender + L_da_gender  │
        │  L_emotion + L_da_emotion│
        │  L_pose + L_da_pose      │
        │  L_facial_hair + L_da_fh │
        │  L_spectacles + L_da_spec│
        │                          │
        │  L_total = Σ weighted    │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  BACKWARD PASS       │
        │  L_total.backward()  │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  OPTIMIZATION        │
        │  optimizer.step()    │
        └──────────────────────┘
```

---

## Output Structure

### **Training Mode:**
```python
logits = (
    (x_spectacles_logits, x_da_spectacles_logits),      # Spectacles task
    (x_facial_hair_logits, x_da_facial_hair_logits),    # Facial hair task
    (x_pose_logits, x_da_pose_logits),                  # Pose task
    (x_emotion_logits, x_da_emotion_logits),            # Emotion task
    (x_gender_logits, x_da_gender_logits),              # Gender task
    x_id_logits,                                        # ID classification logits
    x_id_norm                                           # ID embedding (normalized)
)
```

### **Inference Mode (get_embedding):**
```python
embeddings = (
    x_spectacles_embedding,    # (B × embedding_dim)
    x_facial_hair_embedding,
    x_pose_embedding,
    x_emotion_embedding,
    x_gender_embedding,
    x_id_embedding            # Main task embedding
)
```

---

## Key Design Features

### 1. **Multi-Task Learning (MTL)**
- 1 main task: Face ID Recognition
- 5 auxiliary tasks: Gender, Emotion, Pose, Facial Hair, Spectacles

### 2. **Feature Decomposition**
- Sequential FSM modules separate task-specific features
- Each FSM outputs: task-specific features + residual features for next task

### 3. **Domain Adversarial Training**
- GRL (Gradient Reverse Layer) applied to non-task features
- DA heads trained with reversed gradients
- Improves generalization across domains

### 4. **Attention Mechanism**
- **Channel Attention**: Learned importance of each channel
- **Spatial Attention**: Learned spatial importance
- **SPP (Spatial Pyramid Pooling)**: Multi-scale feature extraction

---

## Parameter Summary

| Component | Details |
|-----------|---------|
| **Input** | 3 × 112 × 112 (Albedo/Normal/Depth) |
| **Backbone** | ConvNeXt V2 (Pre-trained) |
| **Hidden Channels** | 512 |
| **FSM Modules** | 5 (one per auxiliary task) |
| **Attention Types** | Channel + Spatial |
| **Task Heads** | 6 (1 main + 5 auxiliary) |
| **Domain Heads** | 5 (for domain adaptation) |
| **Output Embedding Dim** | Task-dependent |

