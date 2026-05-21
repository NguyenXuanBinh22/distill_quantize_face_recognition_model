import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 24)
ax.axis('off')

# Color scheme
color_input = '#FFE5E5'
color_backbone = '#E5F3FF'
color_adapter = '#E5FFE5'
color_fsm = '#FFF5E5'
color_head = '#F0E5FF'
color_output = '#FFE5F5'

def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=False):
    """Draw a box with text"""
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, weight=weight)

def draw_arrow(ax, x1, y1, x2, y2, style='->'):
    """Draw an arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

# Title
ax.text(5, 27, 'MTL Face Recognition Network Architecture',
        ha='center', fontsize=18, weight='bold')

# Layer 1: Input
draw_box(ax, 5, 25, 3, 1.2, 'INPUT\n(3 × 112 × 112)', color_input, fontsize=11, bold=True)
draw_arrow(ax, 5, 24.4, 5, 23.6)

# Layer 2: Backbone
draw_box(ax, 5, 22.5, 3.5, 1.2, 'ConvNeXt V2 Backbone\n(Pre-trained)', color_backbone, fontsize=11, bold=True)
draw_arrow(ax, 5, 21.9, 5, 21.1)

# Layer 3: Adapter
draw_box(ax, 5, 20, 3.5, 1.2, 'Adapter Layer\nConv2d + BN + PReLU\n→ 512 channels', color_adapter, fontsize=10, bold=True)
draw_arrow(ax, 5, 19.4, 5, 18.6)

# Layer 4: FSM Modules (Sequential)
y_start = 18
fsm_boxes = [
    ('FSM1: Spectacles', 2, 16.5),
    ('FSM2: Facial Hair', 5, 16.5),
    ('FSM3: Emotion', 8, 16.5),
]

for text, x, y in fsm_boxes[:3]:
    draw_box(ax, x, y, 2, 1, text, color_fsm, fontsize=9, bold=True)

# FSM flow visualization
draw_arrow(ax, 5, 18.6, 2.5, 17)
draw_arrow(ax, 2, 16, 5, 15.4)
draw_arrow(ax, 5.5, 16, 8, 15.4)

# FSM4 and FSM5
draw_box(ax, 3, 14, 1.8, 1, 'FSM4: Pose', color_fsm, fontsize=9, bold=True)
draw_box(ax, 7, 14, 1.8, 1, 'FSM5: Gender', color_fsm, fontsize=9, bold=True)

draw_arrow(ax, 8, 15.4, 3.5, 14.5)
draw_arrow(ax, 3.5, 14.5, 6.5, 14.5)

# Layer 5: Decomposed Features
y_features = 12.5
features = [
    ('x_spectacles', 1, y_features),
    ('x_facial_hair', 2.5, y_features),
    ('x_emotion', 4, y_features),
    ('x_pose', 5.5, y_features),
    ('x_gender', 7, y_features),
    ('x_id', 8.5, y_features),
]

for text, x, y in features:
    draw_box(ax, x, y, 1.3, 0.8, text, '#FFFFCC', fontsize=8)

# Arrows from FSM to features
draw_arrow(ax, 3, 14, 1.5, 12.9)
draw_arrow(ax, 3, 14, 3, 12.9)
draw_arrow(ax, 8, 14, 4.5, 12.9)
draw_arrow(ax, 8, 14, 5.5, 12.9)
draw_arrow(ax, 7, 14, 7, 12.9)
draw_arrow(ax, 7, 14, 8.5, 12.9)

# Layer 6: Task Heads
y_heads = 10.5
heads = [
    ('Spectacles\nHead', 1, y_heads),
    ('Facial Hair\nHead', 2.5, y_heads),
    ('Emotion\nHead', 4, y_heads),
    ('Pose\nHead', 5.5, y_heads),
    ('Gender\nHead', 7, y_heads),
    ('ID\nHead', 8.5, y_heads),
]

for text, x, y in heads:
    draw_box(ax, x, y, 1.3, 1.2, text, color_head, fontsize=9, bold=True)
    draw_arrow(ax, x, y_features-0.4, x, y+0.6)

# Layer 7: Domain Adversarial Branch
y_da = 9
draw_box(ax, 8, y_da, 1.5, 0.8, 'GRL Layer', '#FFD700', fontsize=9, bold=True)
draw_arrow(ax, 8.5, 10.5, 8.5, 9.4)

draw_box(ax, 8, 7.5, 1.5, 0.8, 'DA Heads', '#FFD700', fontsize=9, bold=True)
draw_arrow(ax, 8, 8.6, 8, 7.9)

# Layer 8: Logits Output
y_logits = 6
logits = [
    ('Spec\nLogits', 1, y_logits),
    ('Hair\nLogits', 2.5, y_logits),
    ('Emotion\nLogits', 4, y_logits),
    ('Pose\nLogits', 5.5, y_logits),
    ('Gender\nLogits', 7, y_logits),
    ('ID\nLogits', 8.5, y_logits),
]

for text, x, y in logits:
    draw_box(ax, x, y, 1.2, 0.8, text, color_output, fontsize=8)
    if x < 8:
        draw_arrow(ax, x, y_heads-0.6, x, y+0.4)
    else:
        draw_arrow(ax, x, y_heads-0.6, x, y+0.4)

# DA Logits
draw_box(ax, 8, 5, 1.5, 0.8, 'DA\nLogits', color_output, fontsize=8)
draw_arrow(ax, 8, 7.1, 8, 5.4)

# Loss Computation
y_loss = 3.5
draw_box(ax, 5, y_loss, 4, 1.2, 'Loss Computation\nL_total = L_id + Σ(L_task + L_da_task)', '#FFCCCC', fontsize=10, bold=True)

# Arrows from all outputs to loss
for _, x, _ in logits:
    draw_arrow(ax, x, y_logits-0.4, 5, y_loss+0.6, style='->')

draw_arrow(ax, 8, 5-0.4, 5, y_loss+0.6, style='->')

# Backward & Optimization
draw_arrow(ax, 5, 2.9, 5, 2.1)
draw_box(ax, 5, 1.5, 4, 1, 'Backward Pass & Optimization\noptimizer.step()', '#CCFFCC', fontsize=10, bold=True)

# Add information box on the right
info_y = 4
ax.text(9.5, info_y+4, 'INPUTS:', fontsize=10, weight='bold', ha='left')
ax.text(9.5, info_y+3.5, '• Albedo/Normal/Depth Maps', fontsize=8, ha='left')
ax.text(9.5, info_y+3.1, '• Size: 112×112×3', fontsize=8, ha='left')
ax.text(9.5, info_y+2.7, '• EXR format (.npy)', fontsize=8, ha='left')

ax.text(9.5, info_y+2, 'LABELS:', fontsize=10, weight='bold', ha='left')
ax.text(9.5, info_y+1.5, '• ID (main task)', fontsize=8, ha='left')
ax.text(9.5, info_y+1.1, '• Gender (auxiliary)', fontsize=8, ha='left')
ax.text(9.5, info_y+0.7, '• Emotion, Pose, etc.', fontsize=8, ha='left')

ax.text(9.5, info_y-0.5, 'OUTPUTS:', fontsize=10, weight='bold', ha='left')
ax.text(9.5, info_y-1, '• 6 task logits', fontsize=8, ha='left')
ax.text(9.5, info_y-1.4, '• 5 DA logits', fontsize=8, ha='left')
ax.text(9.5, info_y-1.8, '• ID embedding', fontsize=8, ha='left')

plt.tight_layout()
plt.savefig('network_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
print("[OK] Saved: network_architecture.png")
plt.close()

# Create a second detailed diagram showing FSM module internals
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

ax.text(5, 13.5, 'FSM (Facial Semantic Module) - AttentionModule Details',
        ha='center', fontsize=16, weight='bold')

# Input
draw_box(ax, 5, 12, 3, 0.8, 'Input: Feature Map (512×H×W)', color_adapter, fontsize=10, bold=True)
draw_arrow(ax, 5, 11.6, 5, 10.9)

# SPP Modules
draw_box(ax, 2.5, 10.2, 2, 0.8, 'AVG SPP Module\n(Pool 1x1, 2x2, 3x3)', '#FFE5E5', fontsize=9)
draw_box(ax, 7.5, 10.2, 2, 0.8, 'MAX SPP Module\n(Pool 1x1, 2x2, 3x3)', '#FFE5E5', fontsize=9)

draw_arrow(ax, 4, 11.6, 2.5, 10.6)
draw_arrow(ax, 6, 11.6, 7.5, 10.6)

# Concatenate
draw_arrow(ax, 2.5, 9.8, 3.5, 9.2)
draw_arrow(ax, 7.5, 9.8, 6.5, 9.2)
draw_box(ax, 5, 8.8, 2, 0.8, 'Concatenate\n(C×36)', '#E5FFE5', fontsize=9)

# Two branches
draw_arrow(ax, 4, 8.4, 2.5, 7.8)
draw_arrow(ax, 6, 8.4, 7.5, 7.8)

# Channel Attention
draw_box(ax, 2.5, 7.2, 2, 1.2, 'Channel Attention\nConv(C×36 > C/16)\nReLU\nConv(C/16 > C)\nSigmoid', '#F0E5FF', fontsize=8)

# Spatial Attention
draw_box(ax, 7.5, 7.2, 2, 1.2, 'Spatial Attention\nConcat(Max, Mean)\nConv(2 > 1, k=7)\nBatchNorm\nSigmoid', '#F0E5FF', fontsize=8)

# Combine
draw_arrow(ax, 2.5, 6.6, 3.5, 6)
draw_arrow(ax, 7.5, 6.6, 6.5, 6)
draw_box(ax, 5, 5.6, 2.5, 0.8, 'Combine & Scale\n(xchannel + xspatial) x 0.5', '#FFFFCC', fontsize=9)

# Output decomposition
draw_arrow(ax, 5, 5.2, 5, 4.5)
draw_box(ax, 5, 4.1, 3, 0.8, 'Feature Decomposition', '#E5FFE5', fontsize=10, bold=True)

draw_arrow(ax, 4, 3.7, 2.5, 3.2)
draw_arrow(ax, 6, 3.7, 7.5, 3.2)

draw_box(ax, 2.5, 2.8, 2, 0.8, 'x_task = x - x_non_task\n(Task-specific)', color_output, fontsize=9)
draw_box(ax, 7.5, 2.8, 2, 0.8, 'x_non_task\n(For next FSM)', color_output, fontsize=9)

# Add attention info
ax.text(0.5, 1, 'Attention Mechanism:', fontsize=10, weight='bold')
ax.text(0.5, 0.6, '* Channel Attention: Learns importance of each feature channel', fontsize=8)
ax.text(0.5, 0.2, '* Spatial Attention: Learns spatial importance map', fontsize=8)

plt.tight_layout()
plt.savefig('fsm_attention_module.png', dpi=150, bbox_inches='tight', facecolor='white')
print("[OK] Saved: fsm_attention_module.png")
plt.close()

print("\n[DONE] Network architecture diagrams created successfully!")
print("       * network_architecture.png - Main network flow")
print("       * fsm_attention_module.png - FSM module details")
