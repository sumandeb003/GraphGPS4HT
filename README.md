

# GraphGPS4HT

[GraphGPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/5d4834a159f1547b267a05a4e2b7cf5e-Paper-Conference.pdf) is a tool for Graph ML. 

## Goals of Project:

1. **Train GraphGPS on HT dataset**

    a) Phases through which the training goes (**GOT A HIGH-LEVEL IDEA FROM IMPLEMENTATION PERSPECTIVE; EXPLAINED BELOW ALL THE INFORMATION I HAVE DISCOVERED SO FAR ABOUT THE OVERALL WORKFLOW OF GRAPHGPS; I WANT MORE CLARITY**) (**Done by $\color{red}{28.08.2023}$**)

    b) Where are the hyperparameters set?(**Done by $\color{red}{28.08.023}$**)

    c) Understand the meaning of the hyperparameters (**NEED TO DICUSS WITH VIJAY ABOUT THE HYPERPARAMETERS THAT I DIDN'T UNDERSTAND**)

    d) How a dataset is called? (**Done by $\color{red}{08.09.2023}$**)

    e) How to add a new dataset to GraphGPS? How to call the new dataset? (**NEXT**)

    f) Convert the TrustHub benchmarks using the ckt-to-graph conversion code of HW2VEC (**NEAR FUTURE**)

2. **Compare with other GNN-based tools (trained on the same dataset and tested on the same dataset).**
3. **Proliferate the TrustHub dataset using the [GAINESIS tool](https://www.mdpi.com/2079-9292/11/2/245), [S. Bhunia's tool](https://arxiv.org/pdf/2204.08580.pdf) and another tool (can't recall the title; need to check my collection of papers)**
4. **Extend the work to node classification**

## August 28, 2023

## Workflow of GraphGPS (in Short):

**Parse arguments of the execution command from CLI** 

⬇️

**Extract the location of configuration file from the list of arguments** 

⬇️

**Set default values of parameters of the experiment** 

⬇️

**Load configurations from the above configuration file  and also any configuration specifically mentioned  through command line** 

 ⬇️

**Set output directory where the results are stored** 

⬇️

**Combine the configurations specified in the user-given configuration file and the default values of unspecified configurations from `graphgym/config.py` and from the configuration files in `graphgps/config` into the `config.yaml` file in `custom_out_dir`**

⬇️

**create run-directory (in `custom_out_dir`) during each run of experiment** 

⬇️

**create dataset loader, a logger that logs info in `logging.log` inside the run-directory, create model, optimizer and an LR scheduler**

⬇️

**Perform training**

⬇️

**Display best epoch and the corresponding performances on training, validation and test sets.**



Steps in GrahGPS Workflow (in Detail):
--------------------------------------
1. Load cmd line args
```
args = parse_args()
print (args)
```
args: `Namespace(cfg_file='configs/GPS/zinc-GPS+RWSE.yaml', repeat=1, mark_done=False, opts=['wandb.use', 'False'])`

Parses the command line for arguments like `cfg` - configuration file path, `repeat` - the number of repeated jobs, `mark_done` - marking yaml as done after a job has finished, `opt` - configuration options.

The configurations for the executed experiment (`python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml  wandb.use False`) are given in the file: `configs/GPS/zinc-GPS+RWSE.yaml`

The default values of these and other unspecified parameters in this file are stated in the `set_cfg(cfg)` method of the file:  https://github.com/snap-stanford/GraphGym/blob/master/graphgym/config.py

Note that some of the parameters in `configs/GPS/zinc-GPS+RWSE.yaml` are custom defined for this project and not present in the `set_cfg` method. The default values of these custom parameters are stated in https://github.com/rampasek/GraphGPS/tree/main/graphgps/config

The `set_cfg()` method combines the default values of the parameters of GraphGym and those of the custom parameters of the project. This is done by the following code snippet in `set_cfg()`:

```
for func in register.config_dict.values():
        func(cfg)
```
:thinking: :thinking: :thinking:<span style="color:red">**I AM STILL NOT CLEAR ABOUT HOW THE** </span> `config_dict` <span style="color:red"> **DICTIONARY IS POPULATED WITH KEYS AND VALUES. WHICH CODE CARRIES IT OUT?** </span>:thinking: :thinking: :thinking:

2. Load config file
```
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
```
**set_cfg(cfg)**: Sets default values of parameters of the experiment. The default values of these and other unspecified parameters in this file are stated in the `set_cfg(cfg)` method of the file:  https://github.com/snap-stanford/GraphGym/blob/master/graphgym/config.py

Note that the parameters in 'configs/GPS/zinc-GPS+RWSE.yaml' custom defined for this project are not present in the `set_cfg` method of GraphGym. The default values of these custom parameters are stated in the .py files of https://github.com/rampasek/GraphGPS/tree/main/graphgps/config

**load_cfg(cfg, args)**: Loads configurations from the configuration file mentioned in command line and also any configuration specifically mentioned  through command line.

**custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)**: result is the `custom_out_dir` (= 'results/' + 'zinc-GPS+RWSE')

**dump_cfg(cfg)**: Combines the configurations specified in the configuration file (e.g. 'configs/GPS/zinc-GPS+RWSE.yaml'; argument in CLI)  and the default values of unspecified configurations from `graphgym/config.py` to `custom_out_dir`.

3. `custom_set_run_dir(cfg, run_id)`: it sets custom output directory for each experiment run. Inside the custom output directory (here, 'results/zinc-GPS+RWSE'), a separate directory is created during each run. The title of this directory is the run-id.

4. `set_printing`: set printing options

5. Set split index (to choose which split to use in case of multiple available splits), seed, and run id as per the current run.

6. If configured for pretrained model, update `cfg` from the pretrained-model configurations in `pretrained_cfg_fname` (= `osp.join(cfg.pretrained.dir, 'config.yaml'`); e.g. `/home/sumandeb/GraphGPS/pretrained/pcqm4m-GPS+RWSE.deep/config.yaml`). This is done by `load_pretrained_model_cfg(cfg)` in `GraphGPS/graphgps/finetuning.py`

7. `seed_everything`: sets the seed for generating random numbers in pytorch

8. `create_loader()`: creates loaders for each dataset. It is in `GraphGym/graphgym/loader.py`

9. `create_logger()`: create a list of logger objects. It is in `GraphGPS/graphgps/logger.py`

10. `create_model()`: Creates and returns a Python dictionary to register a model. It is in `GraphGym/graphgym/model_builder.py`. Result of `print(f'model:{model}')` is given at the end of this note.

11. If pretrained model is provided, `init_model_from_pretrained()`: uploads the `pretrained_dict` to the `state_dict` of the model (created using `create_model()`).

12. `create_optimizer()`: It is located in `GraphGym/graphgym/optimizer.py`. It loads an ADAM or SGD optimizer as per the configurations (`config.optim`) in `results/'benchmarkname'/config.yaml`

13. `create_scheduler()`: Creates a config-driven LR scheduler. It is located in `GraphGym/graphgym/optimizer.py`

14. ```
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)```

It logs the model, configurations and the given string+variable (`cfg.params`) into `results/'benchmark-name'/logging.log`. 
Note:`logging.info(f"some_string {some_variable}")`: Python provides a module called `logging` for logging messages. 



15. Perform training: Executed by `train_dict[cfg.train.mode](loggers, loaders, model, optimizer,scheduler)`

16. `agg_runs()`: aggregates the training, validation and test results of each epoch and prints the best epoch and the corresponding statistics.


Upon executing the command `python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml  wandb.use False`,  the details of the training, validation, testing of an epoch , say epoch 1291, are output as:
```
train: {'epoch': 1291, 'time_epoch': 18.93798, 'eta': 13315.24609, 'eta_hours': 3.69868, 'loss': 0.02500516, 'lr': 0.00029223, 'params': 423717, 'time_iter': 0.0605, 'mae': 0.02501, 'r2': 0.9997, 'spearmanr': 0.99983, 'mse': 0.00122, 'rmse': 0.03492}

val: {'epoch': 1291, 'time_epoch': 0.4733, 'loss': 0.08178774, 'lr': 0, 'params': 423717, 'time_iter': 0.01479, 'mae': 0.08179, 'r2': 0.96319, 'spearmanr': 0.99765, 'mse': 0.145, 'rmse': 0.38079}

test: {'epoch': 1291, 'time_epoch': 0.46763, 'loss': 0.07286437, 'lr': 0, 'params': 423717, 'time_iter': 0.01461, 'mae': 0.07286, 'r2': 0.99062, 'spearmanr': 0.99668, 'mse': 0.03816, 'rmse': 0.19534}
```

Aslo, the best epoch so far (at any point of training) and its essential details are summarised as follows, for each epoch until the next best epoch is found:

```
> Epoch 1999: took 20.3s (avg 19.7s) | Best so far: epoch 1291	train_loss: 0.0250 train_mae: 0.0250	val_loss: 0.0818 val_mae: 0.0818	test_loss: 0.0729 test_mae: 0.0729
```


 


## Example Configuration File: configs/GPS/zinc-GPS+RWSE.yaml

```
out_dir: results
metric_best: mae 
metric_agg: argmin 
wandb: 
  use: True 	#additional
  project: ZINC #additional
dataset:
  format: PyG-ZINC
  name: subset
  task: graph
  task_type: regression
  transductive: False
  node_encoder: True
  node_encoder_name: TypeDictNode+RWSE
  node_encoder_num_types: 28
  node_encoder_bn: False
  edge_encoder: True
  edge_encoder_name: TypeDictEdge
  edge_encoder_num_types: 4
  edge_encoder_bn: False
posenc_RWSE:
  enable: True 				#additional
  kernel:
    times_func: range(1,21) #additional
  model: Linear 			#additional
  dim_pe: 28 				#additional
  raw_norm_type: BatchNorm 	#additional
train:
  mode: custom
  batch_size: 32
  eval_period: 1
  ckpt_period: 100
model:
  type: GPSModel
  loss_fun: l1
  edge_decoding: dot
  graph_pooling: add
gt:
  layer_type: GINE+Transformer  # CustomGatedGCN+Performer 		#additional
  layers: 10 													#additional
  n_heads: 4 													#additional
  dim_hidden: 64  # `gt.dim_hidden` must match `gnn.dim_inner` 	#additional
  dropout: 0.0 													#additional
  attn_dropout: 0.5 											#additional
  layer_norm: False 											#additional
  batch_norm: True 												#additional
gnn:
  head: san_graph
  layers_pre_mp: 0
  layers_post_mp: 3  # Not used when `gnn.head: san_graph`
  dim_inner: 64  # `gt.dim_hidden` must match `gnn.dim_inner`
  batchnorm: True
  act: relu
  dropout: 0.0
  agg: mean
  normalize_adj: False
optim:
```

### GNN Model: 

**Output of print(f'model:{model}')**

**🤔 🤔 NEED TO DISCUSS WITH VIJAY TO UNDERSTAND THE MEANING OF EACH OF THE FOLLOWING DETAILS 🤔 🤔**

```
model:GraphGymModule(
  (model): GPSModel(
    (encoder): FeatureEncoder(
      (node_encoder): Concat2NodeEncoder(
        (encoder1): AtomEncoder(
          (atom_embedding_list): ModuleList(
            (0): Embedding(119, 236)
            (1): Embedding(5, 236)
            (2): Embedding(12, 236)
            (3): Embedding(12, 236)
            (4): Embedding(10, 236)
            (5): Embedding(6, 236)
            (6): Embedding(6, 236)
            (7): Embedding(2, 236)
            (8): Embedding(2, 236)
          )
        )
        (encoder2): RWSENodeEncoder(
          (raw_norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pe_encoder): Linear(in_features=16, out_features=20, bias=True)
        )
      )
      (edge_encoder): BondEncoder(
        (bond_embedding_list): ModuleList(
          (0): Embedding(5, 256)
          (1): Embedding(6, 256)
          (2): Embedding(2, 256)
        )
      )
    )
    (layers): Sequential(
      (0): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (1): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (2): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (3): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (4): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (5): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (6): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (7): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (8): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (9): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (10): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (11): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inplace=False)
        (dropout_attn): Dropout(p=0.1, inplace=False)
        (ff_linear1): Linear(in_features=256, out_features=512, bias=True)
        (ff_linear2): Linear(in_features=512, out_features=256, bias=True)
        (act_fn_ff): GELU(approximate='none')
        (norm2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ff_dropout1): Dropout(p=0.1, inplace=False)
        (ff_dropout2): Dropout(p=0.1, inplace=False)
      )
      (12): GPSLayer(
        summary: dim_h=256, local_gnn_type=CustomGatedGCN, global_model_type=Transformer, heads=8
        (local_model): GatedGCNLayer()
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (norm1_local): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm1_attn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (dropout_local): Dropout(p=0.1, inp

```
## Sep 2, 2023
### Loading Custom Datasets in PyG

PyG is based on PyTorch. **PyTorch provides two data primitives that allow you to use pre-loaded datasets as well as your own data**:

  - **`torch.utils.data.Dataset`**: an abstract class representing a dataset. Its `__init__` constructor stores the data samples and their corresponding labels.  
    - **PyTorch provides a number of pre-loaded datasets that subclass `torch.utils.data.Dataset` and implement functions specific to the particular data**.
    - The `torch.utils.data.Dataset` has the `__getitem__` and `__len__` methods implemented in it.
    - **The behavior of the Dataset object is like any Python iterable, such as a list or a tuple.**
  -  **`torch.utils.data.DataLoader`**: wraps an iterable around the `Dataset` to enable easy access to the samples. The datasets can all be passed to a `torch.utils.data.DataLoader` which can load multiple samples in parallel using `torch.multiprocessing` workers.

A Dataset class has three functions: `__init__`, `__len__`, and `__getitem__`. 

1. **The `__init__` function is run once when instantiating the Dataset object. We initialize the directory containing the images, the annotations file, and both transforms.**

2. **The `__len__` function (called as `len(CustomImageDataset)`) returns the number of samples in our dataset.**

3. **The `__getitem__` function provides access to the data samples in the dataset by supporting indexing operation. For example, dataset[i] retrieves the i-th data sample.** Based on the index, it:
    - identifies the image’s location on disk,
    - converts that to a tensor using `read_image`,
    - retrieves the corresponding label from the csv data in `self.img_labels`,
    - calls the transform functions on them (if applicable), and
    - returns the tensor image and corresponding label in a tuple.

### An example of custom dataset:
```
class SimpleDataset(Dataset):
    # defining values in the constructor
    def __init__(self, data_length = 20, transform = None):
        self.x = 3 * torch.eye(data_length, 2)
        self.y = torch.eye(data_length, 4)
        self.transform = transform
        self.len = data_length
     
    # Getting the data samples
    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Getting data size/length
    def __len__(self):
        return self.len
```
**In the object constructor `__init__`, we have created the values of features and targets, namely x and y, assigning their values to the tensors `self.x` and `self.y`.** Each tensor carries 20 data samples while the attribute data_length stores the number of data samples. 

```
dataset = SimpleDataset()
print("length of the SimpleDataset object: ", len(dataset))
print("accessing value at index 1 of the simple_dataset object: ", dataset[1])
```
This prints:
```
length of the SimpleDataset object:  20
accessing value at index 1 of the simple_dataset object:  (tensor([0., 3.]), tensor([0., 1., 0., 0.]))
```
**The behavior of the SimpleDataset object is like any Python iterable, such as a list or a tuple.**

```

for i in range(4):
    x, y = dataset[i]
    print(x, y)
```
This prints:

```
tensor([3., 0.]) tensor([1., 0., 0., 0.])
tensor([0., 3.]) tensor([0., 1., 0., 0.])
tensor([0., 0.]) tensor([0., 0., 1., 0.])
tensor([0., 0.]) tensor([0., 0., 0., 1.])
```

### Another Example of Custom Dataset Creation:

```
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```
### Calling a dataset

All the datasets have almost similar API. They all have two common arguments: `transform` and `target_transform` to transform the input and the target, respectively. **You can also create your own datasets using the provided [base classes](https://pytorch.org/vision/stable/datasets.html#base-classes-datasets).**

Here is an example of how to load the Fashion-MNIST dataset from TorchVision. Fashion-MNIST consists of 60,000 training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes. We load the FashionMNIST Dataset with the following parameters:

1. **`root` is the path where the train/test data is stored**
2. **`train` specifies training or test dataset**
3. **`download=True` downloads the data from the internet if it’s not available at root**
4. **`transform` and `target_transform` specify the feature and label transformations**


```
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

```
We can index `Datasets` manually like a list:  

```
for index in range(len(training_data)):
    img, label = training_data[index] 
```

Here, the index-based access to the individual samples in the dataset is provided by the `__getitem__` function.

**As seen, the `Dataset` retrieves our dataset’s features and labels, one sample at a time. But, while training a model, we typically want to pass samples in mini-batches, reshuffle the data to form new mini-batches after every epoch (to reduce model overfitting), and use Python’s multiprocessing to speed up data retrieval. `DataLoader` is an iterable that abstracts this complexity for us through a simple API.**

```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

```
Having loaded that dataset into the DataLoader, one can iterate through the dataset as needed. **Each time the `DataLoader` returns a new mini-batch of `train_features` and `train_labels` (containing `batch_size=64` features and labels respectively).** Because we specified `shuffle=True`, after we iterate over all batches the data is shuffled.

```
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```
### Another Example of Custom Dataset

Download the dataset from [here](https://download.pytorch.org/tutorial/faces.zip) so that the images are in a directory named ‘data/faces/’. Dataset comes with a csv file with annotations which looks like this:

```
image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
0805personali01.jpg,27,83,27,98, ... 84,134
1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
```

```
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

# Read the csv in __init__ but leave the reading of images to __getitem__. This is memory efficient because
# all the images are not stored in the memory at once but read as required.

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

        # Our dataset will take an optional argument transform so that any required processing can be applied on the sample.
        self.transform = transform 

    def __len__(self):
        return len(self.landmarks_frame)

# Sample of our dataset will be a dict {'image': image, 'landmarks': landmarks}.

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Instantiate an object of this dataset class and iterate through the data samples.

face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

for i, sample in enumerate(face_dataset):

```
