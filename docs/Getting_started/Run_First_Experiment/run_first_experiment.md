
# Run First Experiment: Tiny Dataset Example

## Introduction

In this section, we will demonstrate how to launch the first deterministic experiment using the NeuralSTPP model on a tiny toy dataset, specifically the Pinwheel dataset.
This experiment is designed as a minimal example to validate that the full training pipeline works correctly: data loading, model initialization, logging, and checkpoint creation.

The configuration can be defined:

- either using separate YAML configuration files,

- or directly from the command line (CLI) by overriding parameters interactively.

We’ll start with example YAML configurations, then show how to run the experiment, and finally describe all the artifacts that are generated.

## Configuration

The STPPGC framework uses Hydra configuration management, which organizes configuration files in directories (e.g., [conf/model/, conf/data/, conf/trainer/, etc](https://github.com/YahyaAalaila/STPPGC/tree/main/conf)).
Each YAML file defines a logical part of the experiment (model, dataset, or trainer).

### 1. With YAML files

Below are minimal examples of configuration files for this first experiment:

!!! example "YAML Configuration Files"
    === "conf/dat/pinwheel.yaml"
        ``` yaml linenums="1"
            dataset_id: PinwheelHawkes
            batch_size: 64 # 128
            max_events: 1024

            train_bsz:   16
            val_bsz:     64
            test_bsz:    64
            ddim: 2
            num_workers: 13
            shuffle:     true
            drop_last:   true
            pin_memory:  true
            seed:        42
            log_normalisation: 1
        ```
        This defines a small synthetic dataset of 300 total samples, sufficient for a lightweight deterministic run.
  
    === "conf/model/neuralstpp.yaml"
        ``` yaml linenums="1"
            defaults:
            - /data: pinwheel
            - _self_

            model_sub_id: jump-cnf

            search_space:
            hdims:             [64, 64, 64]
            tpp_hidden_dims:   [8, 20]
            actfn:             "softplus"
            tpp_cond:          false
            tpp_style:         "split"
            tpp_actfn:         "softplus"
            share_hidden:      false
            solve_reverse:     false
            l2_attn:           false
            tol:               0.001
            otreg_strength:    0.0
            tpp_otreg_strength: 0.0
            layer_type:        "concat"
            naive_hutch:       false
            lowvar_trace:      false

            lr:                [1e-3, 1e-5]
            warmup_itrs:       2
            num_iterations:    10
            momentum:          0.9
            weight_decay:      0.0
            gradclip:          1.0
            max_events:        100
        ```

    === "conf/trainer/default.yaml"
        ``` yaml linenums="1"
            gpus:         1
            accelerator: "cpu" # If you utilize MacOS"mps"
            max_epochs:   1
            precision:    32
            seed:         123
            ckpt_dir:     "./checkpoints/PinwheelHawkesSweep"
            save_top_k:   1
            monitor:      "val_loss"
            resume_from:  null

            extra_callbacks:
            - class_path: callbacks.common.test_scheduler.TestSchedulerCallback
                init_args:
                test_every_n_epochs: 5
        ```

Together, these files fully define the experiment.
Once saved in your conf/ directory, Hydra automatically merges them when you run the experiment.

### 2. Via Command Line Interface (CLI) Overrides

Alternatively, you can specify all configuration parameters directly from the command line using Hydra's override syntax.
This approach is useful for quick tests or when you want to avoid creating multiple YAML files. **BUT, How can we write the correct command?**

The STPPGC framework uses Hydra, which allows hierarchical configuration defined in multiple .yaml files (under configs/ directory).
Each file corresponds to a logical module of the experiment:

```bash
    conf/
    │   config.yaml          # main entry (imports all defaults)
    ├───data
    │       pinwheel.yaml    # dataset description
    │
    ├───hpo
    │       default.yaml
    │
    ├───logging
    │       default.yaml
    │
    ├───model
    │       neuralstpp.yaml  # model architecture, search space, regularization
    │       ...              # other model configs
    │
    └───trainer
            default.yaml     # training strategy (epochs, GPU, seed, etc.)
```
So when you launch:

```bash
$ python3 scripts/cli.py --config-name config.yaml ...
```
Hydra loads config.yaml as the base configuration, which contains something like:

```yaml linenums="1"
defaults:
    - data: pinwheel
    - model: neuralstpp
    - trainer: default
    - logging: default
    - hpo: default
...
```
Then, every argument you pass in the CLI in the form of key=value acts as an override of a field in the hierarchical YAML tree.

!!! tip "Breakdown of an Override Command"
    Here is an example command that runs the same experiment as defined in the YAML files above:

    ```bash
    $ python3 scripts/cli.py --config-name config.yaml \
        trainer.accelerator=cpu trainer.gpus=0 trainer.max_epochs=1 \
        hpo.resources.cpu=64 \
        model=neuralstpp \
        model.search_space.tpp_hidden_dims=[32,32] \
        model.search_space.actfn="swish" \
        model.search_space.tpp_cond=true \
        model.search_space.tpp_style="gru" \
        model.search_space.tpp_actfn="swish" \
        model.search_space.tol=1e-4 \
        model.search_space.lr=1e-3 \
        model.search_space.weight_decay=1e-6 \
        model.search_space.otreg_strength=1e-4 \
        model.search_space.tpp_otreg_strength=1e-4 \
        trainer.seed=42
    ```
    💡 How to read this command

    The logic follows the directory structure:

    - `model.search_space.*` → keys under configs/model/neuralstpp.yaml

    - `data.*` → keys under configs/data/pinwheel.yaml

    - `trainer.*` → keys under configs/trainer/trainer.yaml

    Thus, each `.` in the CLI corresponds to a hierarchy inside the YAML file.
    This structure ensures your overrides remain consistent with the developer’s configuration schema.

#### Understanding Dot Notation Mapping
- The key to modifying any experiment parameter is understanding how the command line overrides map directly to the configuration files via **dot notation**. Our configuration is organized hierarchically, typically spanning modules like `model.*`, `data.*`, and `trainer.*`. For example, a configuration file like **`configs/trainer/trainer.yaml`** holds settings for the training loop. When you pass an argument like `trainer.accelerator=cpu` in the CLI, you are explicitly navigating to the `accelerator` field inside that `trainer.yaml` file and setting its value. For more complex settings, like architecture parameters, the dots represent **nested dictionary levels**. For instance, `model.search_space.tpp_hidden_dims=[32,32]` tells Hydra to look for the `tpp_hidden_dims` field, which is nested under `search_space`, within the currently selected model configuration (e.g., `neuralstpp.yaml`). This consistent mapping ensures that any setting, no matter how deep, can be overridden directly from the CLI. Since **CLI overrides always take the highest priority** over the base YAML values, this approach ensures the experiment is fully defined and reproducible.

---

## Running Experiment

After preparing the configuration files, the experiment can be launched using a single command in the CLI.

=== "Run with YAML Configuration Files"
    ```bash
    $ python3 scripts/cli.py --config-name config.yaml
    ```
    Hydra automatically reads and merges the YAML files model/neuralstpp.yaml, data/pinwheel.yaml, and trainer/trainer.yaml.
    This command will run a single, deterministic experiment using the Neural STPP model on the Pinwheel dataset.

=== "Overriding Configuration Directly from the CLI"
    ```bash
    python3 scripts/cli.py --config-name config.yaml \
    data.dataset_id=Pinwheel \
    trainer.accelerator=cpu trainer.gpus=0 trainer.max_epochs=1 \
    model=neuralstpp \
    model.search_space.tpp_hidden_dims=[32,32] \
    model.search_space.actfn="swish" \
    model.search_space.tpp_cond=true \
    model.search_space.tpp_style="gru" \
    model.search_space.tpp_actfn="swish" \
    model.search_space.tol=1e-4 \
    model.search_space.lr=1e-3 \
    model.search_space.weight_decay=1e-6 \
    trainer.seed=42
    ```
    As we already explained in [last part](#understanding-dot-notation-mapping), this command overrides all necessary parameters directly from the CLI, achieving the same effect as the YAML files.

???+ info "Running on Pegasus HPC"

    These experiments were executed on the Pegasus HPC cluster using a custom containerized environment (CUDA, PyTorch, and all dependencies).

    The container image was built and deployed as described in the document:
    📄 Custom Software Environment for BenchSTPP — [Download the Full PDF Guide](../../sources/CSE_BenchSTPP.pdf){:download}

    For Pegasus users, you can directly load the image and run the same experiment with:
    ```bash
    srun --partition=A100-80GB \
     --gres=gpu:1 \
     --cpus-per-task=128 \
     --mem=48G \
     --time=12:00:00 \
     --container-image=/netscratch/drief/images/lightning-stpp.sqsh \
     --container-workdir="$(pwd)" \
     --container-mounts=/netscratch/$USER/projects/STPPGC:/netscratch/$USER/projects/STPPGC,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
     python3 scripts/cli.py --config-name config.yaml trainer.accelerator=gpu trainer.gpus=1 model=neuralstpp data=pinwheel trainer.seed=42
    ```
    This ensures a consistent environment across CPU and GPU runs.
---

## Artifacts Overview

After the experiment finishes, several artifacts are automatically generated within the Ray Tune trial directory `(e.g., ray_results/Pinwheel_neuralstpp/.../)`. These artifacts guarantee reproducibility and traceability.

| Artifact Type | Description | Example Path |
| :--- | :--- | :--- |
| **Logs** | Training and validation statistics, time per iteration, etc., for tracking progress. | `ray_results/.../events.out.tfevents.*` |
| **Checkpoints** | Saved model weights, allowing for future fine-tuning or evaluation without rerunning the entire pipeline. | `ray_results/.../checkpoint_000000` |
| **Config Copy** | The merged configuration (including all CLI overrides) used for this specific run. | `ray_results/.../config.yaml` |
| **Metrics** | Final results, including the validation loss, stored as a JSON file. | `ray_results/.../metrics.json` |
| **System Report** | Hostname, PID, total runtime, and resource usage. | Printed in terminal output and logged in the trial directory. |

???+ example "Terminal Output Snippet"
    ```plaintext
        Trial status: 1 TERMINATED
        Current time: 2025-09-17 14:17:26. Total running time: 4min 28s
        Logical resource usage: 64.0/384 CPUs, 0/1 GPUs (0.0/1.0 accelerator_type:H100)
        ╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
        │ Trial name                       status                     iter     total time (s)        val_loss │
        ├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │ _run_single_trial_a3025_00000    TERMINATED                  1        264.314               404.105 │
        ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

        🕵️‍♂️ best_trial.last_result keys:
        dict_keys(['val_loss', 'timestamp', 'checkpoint_dir_name', 'should_checkpoint', 'done', 'training_iteration', 'trial_id', 'date', 'time_this_iter_s', 'time_total_s', 'pid', 'hostname', 'node_ip', 'config', 'time_since_restore', 'iterations_since_restore', 'experiment_tag'])
        🕵️‍♂️ full last_result dict:
        {'checkpoint_dir_name': 'checkpoint_000000',
        'config': {},
        'date': '2025-09-17_14-17-26',
        'done': True,
        'experiment_tag': '0',
        'hostname': 'serv-7101',
        'iterations_since_restore': 1,
        'node_ip': '192.168.71.181',
        'pid': 3463648,
        'should_checkpoint': True,
        'time_since_restore': 264.3144028186798,
        'time_this_iter_s': 264.3144028186798,
        'time_total_s': 264.3144028186798,
        'timestamp': 1758111446,
        'training_iteration': 1,
        'trial_id': 'a3025_00000',
        'val_loss': 404.104736328125}
    ``` 

The system report (logged as best_trial.last_result) captures all environmental and performance metrics for the trial:

- **Final Loss**: `val_loss`: 404.1047

- **Total Time**: `time_total_s`: 264.314 seconds (approx. 4min 24s)

- **Host Environment**: `hostname`: serv-7101, pid: 3463648

- **Trial Identity**: `trial_id`: a3025_00000, training_iteration: 1

This level of detail ensures full experiment transparency.