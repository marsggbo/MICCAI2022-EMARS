srun -n 1 -w hkbugpusrv01 --cpus-per-task 4 python search.py \
-cn eaconfig \
--config_file ./configs/search_covidct.yml \
trainer.device_ids [0] \
logger.name ea_covidct_mnasnet3d_baseline \
model.width_stages [24,40,80,96,192,320] \
input.size [64,64] \
dataset.batch_size 32 \
trainer.name EATrainer \
mutator.EAMutator.warmup_epochs 3 \
mutator.EAMutator.num_population 10 \
trainer.validate_always True \
mutator.EAMutator.num_population 20