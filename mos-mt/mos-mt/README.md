# Mixture-of-Supernets: Improving Weight-Sharing Supernet Training with Architecture-Routed Mixture-of-Experts

This repository contains code and data used in Mixture-of-Supernet (MoS) work. This repository builds on [Hardware Aware Transformer (HAT)'s repository](https://github.com/mit-han-lab/hardware-aware-transformers).


## Quick Setup

### (1) Install
Run the following commands to install AutoMoE:
```bash
git clone https://github.com/mos-mt.git
cd hatv2
pip install --editable .
```

### (2) Prepare Data
Run the following commands to download preprocessed MT data:
```bash
bash configs/[task_name]/get_preprocessed.sh
```
where `[task_name]` can be `wmt14.en-de` or `wmt14.en-fr` or `wmt19.en-de`.

### (3) Run full MoS pipeline
Run the following commands to start MoS pipeline:
```bash
python generate_script.py --task wmt14.en-de --output_dir /tmp --num_gpus 4 --trial_run 0 --hardware_spec gpu_titanxp --max_experts 6 --frac_experts 1 > automoe.sh
bash automoe.sh
```
where,
* `task` - MT dataset to use: `wmt14.en-de` or `wmt14.en-fr` or `wmt19.en-de` (default: `wmt14.en-de`)
* `output_dir` - Output directory to write files generated during experiment (default: `/tmp`)
* `num_gpus` - Number of GPUs to use (default: `4`)
* `trial_run` - Run trial run (useful to quickly check if everything runs fine without errors.): 0 (final run), 1 (dry/dummy/trial run) (default: `0`)
* `hardware_spec` - Hardware specification: `gpu_titanxp` (For GPU) (default: `gpu_titanxp`)
* `max_experts` - Maximum experts (for Supernet) to use (default: `6`)
* `mos_type` - MoS variant (default: `neuronrouting_jack_drop_2L`)
* `hypernet_hidden_size` - hidden size for the router network (default: `128`)
* `sandwich_rule` - Use sandwich sampling (default: `1`)
* `latency_compute` - Use (partially) gold or predictor latency (default: `gold`)
* `latiter` - Number of latency measurements for using (partially) gold latency (default: `100`)
* `latency_constraint` - Latency constraint in terms of milliseconds (default: `200`)
* `evo_iter` - Number of iterations for evolutionary search (default: `10`)
