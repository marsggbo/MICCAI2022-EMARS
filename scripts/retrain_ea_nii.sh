srun -n 1 -w hkbugpusrv01 --cpus-per-task 2 python retrain.py \
-cn eaconfig \
--config_file outputs/ea_nii_potential_acc/version_0/search_nii_png.yml \
--arc_path outputs/ea_nii_mnasnet3d_baseline/version_3/epoch_26_8.json  \
input.size [256,256] \
cam.enable 0 \
dataset.batch_size 8 \
dataset.slice_num 40 \
trainer.device_ids [0] \
transforms.ct.randomblur.enable 0 \
transforms.ct.randomswap.enable 0 \
transforms.ct.randomnoise.enable 0 \
seed 6666 \
loss.name CrossEntropyLabelSmooth \
optim.scheduler.t_max 160 \

# --arc_path outputs/ea_nii_mnasnet3d_baseline/version_3/epoch_22_3.json  \
# --arc_path outputs/ea_nii_mnasnet3d_baseline/version_3/version_0/epoch_20_12.json  \
# --arc_path outputs/ea_ct_mnas_baseline_findsmall/version_0/archs/epoch_62_1.json  \
# --arc_path outputs/ea_ct_mnas_baseline/version_0/archs/epoch_49_9.json  \
# --arc_path outputs/ea_nii_potential_acc/version_0/archs/epoch_72_9.json  \
# --arc_path outputs/ea_ct_mnas_acceleration_accuracy/version_0/archs/epoch_65_7.json  \
