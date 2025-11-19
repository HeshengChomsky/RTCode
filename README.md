# RT

RT is a novel Decision Transformer approach that enables trajectory stitching by adopting different history length.

## Installation
We prepared a Dockerfile and bash scripts to set up the environment.

1. Build the Docker image and start a Docker container 
```bash
# Download the code from this repo
git clone https://github.com/kristery/Elastic-DT.git
cd Elastic-DT
bash build_image.sh
bash start_container.sh
```

## Training
1. Download D4RL datasets
```bash
cd /workspace
python data/download_d4rl_datasets.py
```

2. Train the EDT agent
```bash
python scripts/train_edt.py --env hopper --dataset medium-replay
```

## Evaluation
```bash
python scripts/eval_edt.py --chk_pt_name saved_model_name_from_training.pt
```

## Acknowledgement
The implementation of EDT is based on [min-decision-transformer](https://github.com/nikhilbarhate99/min-decision-transformer)
