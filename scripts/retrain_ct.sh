srun -n 1 --cpus-per-task 2 python retrain.py \
--config_file outputs/{Your_path}/version_{your_version}/search_ct.yaml \
--arc_path outputs/{Your_path}/version_{your_version}/epoch_0.json  \
input.size [128,128]