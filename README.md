# GraphGPS4HT

## Goals of Project:

1. **Train GraphGPS on my HT dataset**
2. **Compare with other GNN-based tools (trained on the same dataset and tested on the same dataset).**
3. **Proliferate the TrustHub dataset using the GAINESIS, S Bhunia's tool and another tool**
4. **Extend the work to node classification**

## Progress or Questions Answered So Far

a) Phases through which the training goes (**GOT A HIGH-LEVEL IDEA FROM IMPLEMENTATION PERSPECTIVE; EXPLAINED BELOW ALL THE INFORMATION I HAVE DISCOVERED SO FAR ABOUT THE OVERALL WORKFLOW; NEED MORE CLARITY**)

b) Where are the hyperparameters set? (**DONE**)

c) Understand the meaning of the hyperparameters (**NEED TO DICUSS WITH VIJAY ABOUT THE HYPERPARAMETERS THAT I DIDN'T UNDERSTAND**)

d) How a dataset is called? (**NEXT**) 

e) How to add a new dataset TO GraphGPS? How to call the new dataset? (**NEAR FUTURE**)

f) Convert the TrustHub benchmarks using the ckt-to-graph conversion code of HW2VEC (**FUTURE**)

-------------------------------
GrahGPS Workflow:
-------------------------------
1. Load cmd line args
```
args = parse_args()
print (args)
```
args: Namespace(cfg_file='configs/GPS/zinc-GPS+RWSE.yaml', repeat=1, mark_done=False, opts=['wandb.use', 'False'])

Parses the command line for arguments like cfg - configuration file path, repeat - the number of repeated jobs, mark_done - marking yaml as done after a job has finished, opt - configuration options.

The configurations for the executed experiment (python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml  wandb.use False) are given in the file: 'configs/GPS/zinc-GPS+RWSE.yaml'
The default values of these and other unspecified parameters in this file are stated in the set_cfg(cfg) method of the file:  https://github.com/snap-stanford/GraphGym/blob/master/graphgym/config.py
Note that some of the parameters in 'configs/GPS/zinc-GPS+RWSE.yaml' are custom defined for this project and not present in the set_cfg method. The default values of these custom parameters are stated in https://github.com/rampasek/GraphGPS/tree/main/graphgps/config

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
**set_cfg(cfg)**: Sets default values of parameters of the experiment. The default values of these and other unspecified parameters in this file are stated in the set_cfg(cfg) method of the file:  https://github.com/snap-stanford/GraphGym/blob/master/graphgym/config.py
Note that the parameters in 'configs/GPS/zinc-GPS+RWSE.yaml' custom defined for this project are not present in the set_cfg method of GraphGym. The default values of these custom parameters are stated in the .py files of https://github.com/rampasek/GraphGPS/tree/main/graphgps/config

**load_cfg(cfg, args)**: Loads configurations from the configuration file mentioned in command line and also any configuration specifically mentioned  through command line.

**custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)**: result is the custom_out_dir (= 'results/' + 'zinc-GPS+RWSE')

**dump_cfg(cfg)**: Combines the configurations specified in the configuration file (e.g. 'configs/GPS/zinc-GPS+RWSE.yaml'; argument in CLI)  and the default values of unspecified configurations from graphgym/config.py to custom_out_dir.

3. custom_set_run_dir(cfg, run_id): it sets custom output directory for each experiment run. Inside the custom output directory (here, 'results/zinc-GPS+RWSE'), a separate directory is created during each run. The title of this directory is the run-id.
4. set_printing: set printing options
5. Set split index (to choose which split to use in case of multiple available splits), seed, and run id as per the current run.
6. If configured for pretrained model, update cfg from the pretrained-model configurations in pretrained_cfg_fname (= osp.join(cfg.pretrained.dir, 'config.yaml'); e.g. /home/sumandeb/GraphGPS/pretrained/pcqm4m-GPS+RWSE.deep/config.yaml). This is done by load_pretrained_model_cfg(cfg) in GraphGPS/graphgps/finetuning.py
7. seed_everything: sets the seed for generating random numbers in pytorch
8. create_loader(): creates loaders for each dataset. It is in GraphGym/graphgym/loader.py
9. create_logger(): create a list of logger objects. It is in GraphGPS/graphgps/logger.py
10. create_model(): Creates and returns a Python dictionary to register a model. It is in GraphGym/graphgym/model_builder.py. Result of print(f'model:{model}') is given at the end of this note. 

	b) Set machine learning pipeline
	c) Start training



configs/GPS/zinc-GPS+RWSE.yaml
-------------------------------
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
===========================================

===============================================
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

11. if pretrained model is provided, init_model_from_pretrained(): uploads the pretrained_dict to the state_dict of the model (created using create_model()).
12.  create_optimizer(): It is located in GraphGym/graphgym/optimizer.py. It loads an ADAM or SGD optimizer as per the configurations (config.optim) in results/'benchmarkname'/config.yaml
13. create_scheduler(): Creates a config-driven LR scheduler. It is located in GraphGym/graphgym/optimizer.py 
14. logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)

It logs the model, configurations and the given string+variable (cfg.params) into results/'benchmark-name'/logging.log. Note:-logging.info(f"some_string {some_variable}"): Python provides a module called 'logging' for logging messages. 
15. 
16. agg_runs(): aggregates the results and prints the best epoch and the corresponding statistics.
 
Workflow:
----------------
Parse arguments of the execution command from CLI --> Extract the location of configuration file from the list of arguments --> Set default values of parameters of the experiment --> Load configurations from the above configuration file  and also any configuration specifically mentioned  through command line --> set output directory where the results are stored --> Combine the configurations specified in the above configuration file and the default values of unspecified configurations from graphgym/config.py to custom_out_dir --> 

Upon executing the command 'python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml  wandb.use False',  the details of the training, validation, testing of an epoch , say epoch 1291, are output as:

train: {'epoch': 1291, 'time_epoch': 18.93798, 'eta': 13315.24609, 'eta_hours': 3.69868, 'loss': 0.02500516, 'lr': 0.00029223, 'params': 423717, 'time_iter': 0.0605, 'mae': 0.02501, 'r2': 0.9997, 'spearmanr': 0.99983, 'mse': 0.00122, 'rmse': 0.03492}

val: {'epoch': 1291, 'time_epoch': 0.4733, 'loss': 0.08178774, 'lr': 0, 'params': 423717, 'time_iter': 0.01479, 'mae': 0.08179, 'r2': 0.96319, 'spearmanr': 0.99765, 'mse': 0.145, 'rmse': 0.38079}

test: {'epoch': 1291, 'time_epoch': 0.46763, 'loss': 0.07286437, 'lr': 0, 'params': 423717, 'time_iter': 0.01461, 'mae': 0.07286, 'r2': 0.99062, 'spearmanr': 0.99668, 'mse': 0.03816, 'rmse': 0.19534}

Aslo, the best epoch so far (at any point of training) and its essential details are summarised as follows, for each epoch until the next best epoch is found:

> Epoch 1999: took 20.3s (avg 19.7s) | Best so far: epoch 1291	train_loss: 0.0250 train_mae: 0.0250	val_loss: 0.0818 val_mae: 0.0818	test_loss: 0.0729 test_mae: 0.0729
