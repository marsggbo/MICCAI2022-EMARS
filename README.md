

<div align="center">

<h1><a href='https://arxiv.org/abs/2101.10667'>[MICCAI2022] Evolutionary-Multi-objective-Architecture-Search-Framework: Application to COVID-19 3D CT Classification<a></h1>

<div>
    <a href='https://marsggbo.github.io/' target='_blank'>He Xin<sup>1</sup></a>, Guohao Ying<sup>2</sup>, Jiyong Zhang<sup>3</sup>, and Xiaowen Chu<sup>14</sup>
    <br>
    <span><sup> 1</sup> Hong Kong Baptist University, Hong Kong, China </span><br>
    <span><sup> 2</sup> University of Southern California, CA, USA </span><br>
    <span><sup> 3</sup> School of Automation, Hangzhou Dianzi University, Hang Zhou, China </span><br>
    <span><sup> 4</sup> The Hong Kong University of Science and Technology (Guangzhou), China </span><br>
</div>
</div>


## Install


```bash
pip install -r requirements.txt
```


## search

You can refer to `scripts/search_ct.sh` for more run scripts.

```bash
CUDA_VISIBLE_DEVICES=0 python search.py --config_file ./configs/search.yaml logger.name MyExp
```


## retrain


You can refer to `scripts/retrain_ct.sh`

there are two mode for retraininig：
- you can manually choose a promising architecture by specifying `--arch_path` to the path of json file, e.g., `output/MyExp/version_0/epoch_66.json`, and then run the following command

```
CUDA_VISIBLE_DEVICES=0 python retrain.py --config_file ./configs/retrain.yaml --arc_path outputs/MyExp/version_0/epoch_66.json input.size [128,128]
```
- the second is to finetune each selected candidate architecture for a few epochs, and then choose the best-performing one for further training. In this case, you can specify `--arc_path` to the log path, e.g., `output/MyExp/version_0`. The json files in this path will be loaded automatically:


```
CUDA_VISIBLE_DEVICES=0 python retrain.py --config_file ./configs/retrain.yaml --arc_path outputs/MyExp/version_0  input.size [128,128]
```
