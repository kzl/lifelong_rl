from lifelong_rl.samplers.data_collector.base import DataCollector, PathCollector, StepCollector
from lifelong_rl.samplers.utils.rollout_functions import rollout, multitask_rollout


DataCollector, PathCollector, StepCollector = DataCollector, PathCollector, StepCollector
rollout, multitask_rollout = rollout, multitask_rollout
