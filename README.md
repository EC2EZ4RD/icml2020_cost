# Google Research Football

## Quick Start

### Installation on Linux
```
conda create -n football python=3.6
conda activate football
pip install -r requirements.txt
bash setup_envs.sh
```

### Test
```
python -m gfootball.play_game --action_set=full
```

## Running in Docker

### Build Image

```
docker build -t username/football -f ./gfootball_marl_dockerfile .
```

### Interactive Mode

```
nvidia-docker run -it --rm --cpus=30 --memory 80gb --name football -v /home/username/football:/home/football username/football bash
```

### Running Background

```
bash run_docker_container.sh 2 bash run_python_in_docker.sh > /tmp/rllib_football_banditeval.log 2>&1 &
```


## Training

Configurations are in directory [configs](configs).


### Run PPO parameter sharing
```
python run_shared_ppo.py
```

### Run PPO parameter sharing with curriculum
```
python run_shared_ppo_curriculum.py
```

## Code Structure
```
.
├── communication           # Attention communication
├── configs                 # Configuration files
├── curriculum              # Curriculum learning
│   └── teachers            # Teacher algorithms
├── envs                    # Wrapped environments of gfootball and MPE
├── football                # Source code of gfootball environment
├── models                  # Neural network models
├── mpe                     # Source code of MPE environment
├── trainers                # RLLib trainers
├── utils                   # Utility files (plot, replay, etc.)
├── ...
├── requirements.txt
├── setuo_envs.sh           # Install environments
└── README.md
```
