Logs from experiments are saved in the data folder.

Folders represents experiment names, and subfolders contain individual seeds/runs.

Inside of the folder of a seed contains:
  - variant.json: json file containing hyperparameters of experiment
  - stdout.log: standard terminal output from experiment
  - progress.csv: metrics logged by agent at every timestep
  - itr_X.pt: Pytorch snapshot of agent after epoch X

Offline experiments contain the following variants:
  - offline_progress.csv: analogous to progress.csv
  - offline_itr_X.pt: analogous to itr_X.pt

Experiments with both an online and an offline phase will contain both types of files.
