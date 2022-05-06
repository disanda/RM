#!/bin/bash
python orojar_vis_directions.py \
--load_A checkpoints/directions/orojar/goldens_mid.pt \
--search_space mid \
--path_size 3.5 \
--ndirs 40 \
--fix_class 207 \
--experiment_name golden_retrievers_mid \
--parallel --batch_size 16  \
--G_B2 0.999 \
--G_attn 64 \
--G_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 \
--ema --use_ema --ema_start 20000 \
--test_every 10000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 750 \
--use_multiepoch_sampler \
--resume \
--num_epochs 50

#  the path_size for visualizing direction (idx 37, color in paper) is 15