# GaitBase, Gait3D
torchrun --nproc_per_node=8 opengait/main.py --cfgs configs/gaitbase-gait3d-open_set.yaml --phase train

# SwinGait, Gait3D
torchrun --nproc_per_node=8 opengait/main.py --cfgs configs/swingait-gait3d-open_set.yaml --phase train
