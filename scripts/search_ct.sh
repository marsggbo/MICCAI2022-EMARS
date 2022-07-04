srun -n 1 --cpus-per-task 4 python search.py \
--config_file ./configs/search_ct.yaml \
--config_name CTConfig \
model.name Mobile3DNet \
dataset.name FakeData \
trainer.name OnehotTrainer \
dataset.batch_size 2 \
dataset.workers 1 \
trainer.device_ids [0]