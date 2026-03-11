"""
Model architectures for keypoint detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .config import HEATMAP_HEIGHT, HEATMAP_WIDTH, HRNET_WEIGHTS_PATH


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class HRNet(nn.Module):
    """HRNet-W32 backbone for keypoint detection - simplified multi-resolution network."""
    def __init__(self, width=32, ocr_width=256):
        super().__init__()
        self.width = width
        
        # Stem: initial downsampling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: High resolution backbone (similar to ResNet)
        # Output: 64 * BasicBlock.expansion = 64 channels
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)
        stage1_output_channels = 64  # After BasicBlock expansion
        
        # Stage 2: Two branches (1x and 1/2x resolution)
        # After stage1, we have stage1_output_channels (64) channels
        self.transition1_1 = nn.Sequential(
            nn.Conv2d(stage1_output_channels, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.transition1_2 = nn.Sequential(
            nn.Conv2d(stage1_output_channels, width*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width*2),
            nn.ReLU(inplace=True)
        )
        # Stage 2 branches: input channels = output channels (BasicBlock.expansion = 1)
        self.stage2_branch1 = self._make_layer(BasicBlock, width, width, 4)
        self.stage2_branch2 = self._make_layer(BasicBlock, width*2, width*2, 4)
        # After stage2: branch1 outputs width channels, branch2 outputs width*2 channels
        
        # Stage 3: Three branches (1x, 1/2x, 1/4x resolution)
        # Branch1 and branch2 continue from stage2 (no transition needed)
        self.transition2_1 = None  # Keep same resolution
        self.transition2_2 = None  # Keep same resolution
        # Branch3 is created from branch2
        self.transition2_3 = nn.Sequential(
            nn.Conv2d(width*2, width*4, 3, 2, 1, bias=False),  # width*2 = 64, width*4 = 128
            nn.BatchNorm2d(width*4),
            nn.ReLU(inplace=True)
        )
        # Stage 3 branches: each branch processes its input and maintains same output channels
        self.stage3_branch1 = self._make_layer(BasicBlock, width, width, 4)  # width -> width
        self.stage3_branch2 = self._make_layer(BasicBlock, width*2, width*2, 4)  # width*2 -> width*2
        self.stage3_branch3 = self._make_layer(BasicBlock, width*4, width*4, 4)  # width*4 -> width*4
        
        # Stage 4: Four branches (1x, 1/2x, 1/4x, 1/8x resolution)
        self.transition3_1 = None
        self.transition3_2 = None
        self.transition3_3 = None
        self.transition3_4 = nn.Sequential(
            nn.Conv2d(width*4, width*8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width*8),
            nn.ReLU(inplace=True)
        )
        self.stage4_branch1 = self._make_layer(BasicBlock, width, width, 4)
        self.stage4_branch2 = self._make_layer(BasicBlock, width*2, width*2, 4)
        self.stage4_branch3 = self._make_layer(BasicBlock, width*4, width*4, 4)
        self.stage4_branch4 = self._make_layer(BasicBlock, width*8, width*8, 4)
        
        # Fusion: Upsample and concatenate all branches to highest resolution
        # After stage4: branch1=width, branch2=width*2, branch3=width*4, branch4=width*8
        # Use 1x1 convs to reduce channels, then we'll interpolate to exact size in forward
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(width*2, width, 1, 1, 0, bias=False),  # 64 -> 32
            nn.BatchNorm2d(width)
        )
        self.fusion_conv3 = nn.Sequential(
            nn.Conv2d(width*4, width, 1, 1, 0, bias=False),  # 128 -> 32
            nn.BatchNorm2d(width)
        )
        self.fusion_conv4 = nn.Sequential(
            nn.Conv2d(width*8, width, 1, 1, 0, bias=False),  # 256 -> 32
            nn.BatchNorm2d(width)
        )
        
        # Final layer
        total_channels = width * 4  # Sum of all branches
        self.final_layer = nn.Conv2d(total_channels, ocr_width, kernel_size=1, stride=1, padding=0)
        
        # Output channels from HRNet
        self.output_channels = ocr_width
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Stage 2: Split into two branches
        x_b1 = self.transition1_1(x)
        x_b2 = self.transition1_2(x)
        x_b1 = self.stage2_branch1(x_b1)
        x_b2 = self.stage2_branch2(x_b2)
        
        # Stage 3: Add third branch
        x_b3 = self.transition2_3(x_b2)
        x_b1 = self.stage3_branch1(x_b1)
        x_b2 = self.stage3_branch2(x_b2)
        x_b3 = self.stage3_branch3(x_b3)
        
        # Stage 4: Add fourth branch
        x_b4 = self.transition3_4(x_b3)
        x_b1 = self.stage4_branch1(x_b1)
        x_b2 = self.stage4_branch2(x_b2)
        x_b3 = self.stage4_branch3(x_b3)
        x_b4 = self.stage4_branch4(x_b4)
        
        # Fusion: Upsample all branches to highest resolution and concatenate
        # First reduce channels
        x_b2_reduced = self.fusion_conv2(x_b2)
        x_b3_reduced = self.fusion_conv3(x_b3)
        x_b4_reduced = self.fusion_conv4(x_b4)
        
        # Get target size from branch1 (highest resolution)
        _, _, h_b1, w_b1 = x_b1.shape
        
        # Interpolate all branches to exact size of branch1
        x_b2_up = F.interpolate(x_b2_reduced, size=(h_b1, w_b1), mode='bilinear', align_corners=False)
        x_b3_up = F.interpolate(x_b3_reduced, size=(h_b1, w_b1), mode='bilinear', align_corners=False)
        x_b4_up = F.interpolate(x_b4_reduced, size=(h_b1, w_b1), mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        y = torch.cat([x_b1, x_b2_up, x_b3_up, x_b4_up], dim=1)
        y = self.final_layer(y)
        
        return y


class KeypointDetector(nn.Module):
    """Keypoint detector using HRNet backbone and heatmap regression."""
    def __init__(self, num_keypoints=5, heatmap_size=(HEATMAP_HEIGHT, HEATMAP_WIDTH), weights_path=HRNET_WEIGHTS_PATH):
        super().__init__()
        self.num_keypoints = num_keypoints
        if isinstance(heatmap_size, (list, tuple)) and len(heatmap_size) == 2:
            self.heatmap_height, self.heatmap_width = heatmap_size
        else:
            self.heatmap_height = self.heatmap_width = int(heatmap_size)

        # HRNet backbone
        self.backbone = HRNet(width=32)
        backbone_output_channels = self.backbone.output_channels
        print(f"HRNet output channels: {backbone_output_channels}")
        
        # Simplified decoder for HRNet output (already high-resolution)
        self.decoder = nn.Sequential(
            nn.Conv2d(backbone_output_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),  # Increased dropout from 0.1 to 0.15 for stronger regularization (reduces overfitting)
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),  # Increased dropout from 0.1 to 0.15 for stronger regularization (reduces overfitting)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_keypoints, kernel_size=1)
        )
        
        # Load weights if provided
        weights_path = Path(weights_path) if weights_path else None
        if weights_path and weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                # Handle different weight formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                # Remove 'module.' prefix if present
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                # Load only matching keys
                model_dict = self.backbone.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                self.backbone.load_state_dict(model_dict)
                print(f"Loaded HRNet weights from {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load HRNet weights from {weights_path}: {e}")
                print("Training from scratch...")

    def forward(self, x):
        # HRNet outputs high-resolution features
        features = self.backbone(x)
        # Debug: verify feature channels
        if features.shape[1] != self.backbone.output_channels:
            raise RuntimeError(
                f"HRNet output channels mismatch: expected {self.backbone.output_channels}, "
                f"got {features.shape[1]}"
            )
        # Decode to heatmaps
        heatmaps = self.decoder(features)
        # Interpolate to target heatmap size
        return torch.nn.functional.interpolate(
            heatmaps, size=(self.heatmap_height, self.heatmap_width), mode='bilinear', align_corners=False
        )
