from mlagents.trainers.trainer_util import load_config

from animalai_train.run_options_aai import RunOptionsAAI
from animalai_train.run_training_aai import run_training_aai

trainer_config_path = (
    "configurations/training_configurations/train_ml_agents_config_ppo_curiosity_intensive.yaml"
)
environment_path = "env/AnimalAI"
curriculum_path = "configurations/detour_curriculum_40/0. food_no_rotation"
run_id = "train_curriculum_custom_curiosity_medium"
base_port = 5008
number_of_environments = 8
number_of_arenas_per_environment = 1

args = RunOptionsAAI(
    trainer_config=load_config(trainer_config_path),
    env_path=environment_path,
    load_model=False,
    run_id=run_id,
    base_port=base_port,
    num_envs=number_of_environments,
    curriculum_config=curriculum_path,
    n_arenas_per_env=number_of_arenas_per_environment,
)

run_training_aai(0, args)
