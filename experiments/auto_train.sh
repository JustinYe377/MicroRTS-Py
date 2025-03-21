#!/bin/bash

# ====== CONFIG ======
WANDB_ENTITY="yejustin213-ohio-university"   # <== replace this with your wandb entity
WANDB_PROJECT="gym-microrts"   # <== replace this with your wandb project name

timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="ppo_microrts_${timestamp}"

# ====== Hyperparameters (edit here in one place) ======
TOTAL_TIMESTEPS=100000000
LEARNING_RATE=1e-4
NUM_BOT_ENVS=0
NUM_SELFPLAY_ENVS=24
PARTIAL_OBS=False
NUM_STEPS=512
UPDATE_EPOCHS=5
N_MINIBATCH=8
CLIP_COEF=0.2
ENT_COEF=0.015
GAMMA=0.99
GAE_LAMBDA=0.95
VF_COEF=0.5
ANNEAL_LR=True
CLIP_VLOSS=True
MAX_GRAD_NORM=0.5
NUM_MODELS=20
TRAIN_MAPS="maps/16x16/basesWorkers16x16A.xml"
EVAL_MAPS="maps/16x16/basesWorkers16x16A.xml"
REWARD_WEIGHTS="10.0 1.0 1.0 0.2 1.0 4.0"

# ====== Save config to logs ======
mkdir -p logs
CONFIG_FILE="logs/config_${exp_name}.txt"

echo "Saving hyperparameters to $CONFIG_FILE"
cat <<EOT > $CONFIG_FILE
EXP_NAME=$exp_name
WANDB_ENTITY=$WANDB_ENTITY
WANDB_PROJECT=$WANDB_PROJECT
TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS
LEARNING_RATE=$LEARNING_RATE
NUM_BOT_ENVS=$NUM_BOT_ENVS
NUM_SELFPLAY_ENVS=$NUM_SELFPLAY_ENVS
PARTIAL_OBS=$PARTIAL_OBS
NUM_STEPS=$NUM_STEPS
UPDATE_EPOCHS=$UPDATE_EPOCHS
N_MINIBATCH=$N_MINIBATCH
CLIP_COEF=$CLIP_COEF
ENT_COEF=$ENT_COEF
GAMMA=$GAMMA
GAE_LAMBDA=$GAE_LAMBDA
VF_COEF=$VF_COEF
ANNEAL_LR=$ANNEAL_LR
CLIP_VLOSS=$CLIP_VLOSS
MAX_GRAD_NORM=$MAX_GRAD_NORM
NUM_MODELS=$NUM_MODELS
TRAIN_MAPS=$TRAIN_MAPS
EVAL_MAPS=$EVAL_MAPS
REWARD_WEIGHTS=$REWARD_WEIGHTS
EOT

# ====== Start training ======
xvfb-run -a python train.py \
  --exp-name $exp_name \
  --gym-id MicroRTSGridModeVecEnv \
  --total-timesteps $TOTAL_TIMESTEPS \
  --learning-rate $LEARNING_RATE \
  --num-bot-envs $NUM_BOT_ENVS \
  --num-selfplay-envs $NUM_SELFPLAY_ENVS \
  --partial-obs $PARTIAL_OBS \
  --num-steps $NUM_STEPS \
  --update-epochs $UPDATE_EPOCHS \
  --n-minibatch $N_MINIBATCH \
  --clip-coef $CLIP_COEF \
  --ent-coef $ENT_COEF \
  --gamma $GAMMA \
  --gae-lambda $GAE_LAMBDA \
  --vf-coef $VF_COEF \
  --anneal-lr $ANNEAL_LR \
  --clip-vloss $CLIP_VLOSS \
  --max-grad-norm $MAX_GRAD_NORM \
  --num-models $NUM_MODELS \
  --prod-mode \
  --wandb-entity $WANDB_ENTITY \
  --wandb-project-name $WANDB_PROJECT \
  --train-maps $TRAIN_MAPS \
  --eval-maps $EVAL_MAPS \
  --reward-weights $REWARD_WEIGHTS \
  2>&1 | tee "logs/${exp_name}.log"

echo "Training complete. Log saved to logs/${exp_name}.log"
echo "Hyperparameters for this run saved to $CONFIG_FILE"


