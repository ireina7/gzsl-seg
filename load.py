import sys

sys.path.append('./src')
print('[info] Everything loaded.')

run_experiment = True
# For convenience ...
import train_voc

if run_experiment:
  train_voc.main()
