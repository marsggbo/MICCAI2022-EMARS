srun -n 1 -w hkbugpusrv01 --cpus-per-task 2 python retrain.py \
-cn eaconfig \
--config_file outputs/ea_covidct_potential_acc/version_0/search_covidct.yml \
--arc_path outputs/ea_covidct_mnasnet3d_baseline/version_0/archs/epoch_43_14.json  \
input.size [512,512] \
cam.enable 0 \
dataset.batch_size 8 \
trainer.device_ids [0,1] \
dataset.slice_num 32 \
transforms.ct.randomnoise.enable 0 \
seed 2323 \
loss.name CrossEntropyLabelSmooth \
optim.scheduler.t_max 160 \
optim.name sgd \
optim.base_lr 0.01 \

# --arc_path outputs/ea_covidct_mnasnet3d_baseline/version_0/archs/epoch_42_3.json  \
# --arc_path outputs/ea_covidct_mnasnet3d_baseline/version_0/archs/epoch_67_39.json  \
# --arc_path outputs/ea_covidct_mnasnet3d_baseline/version_0/epoch_65_15.json  \
# --arc_path outputs/ea_covidct_mnasnet3d_baseline/version_0/epoch_66_18.json  \
# --arc_path outputs/ea_covidct_mnasnet3d_baseline/version_0/epoch_42_23.json  \
# --arc_path outputs/ea_covidct_mnasnet3d_baseline/version_0/epoch_65_35.json  \
# --arc_path outputs/ea_covidct_mnasnet3d_baseline/version_0/epoch_67_19.json  \
# --arc_path outputs/ea_covidct_potential_acc/version_0/archs/epoch_56_13.json  \
# --arc_path outputs/ea_covidct_potential_acc/version_0/archs/epoch_71_14.json  \
# --arc_path outputs/ea_ct_mnas_baseline_findlarge/version_0/archs/epoch_90_17.json  \
# --arc_path outputs/ea_ct_mnas_baseline_findsmall/version_0/archs/epoch_62_1.json  \