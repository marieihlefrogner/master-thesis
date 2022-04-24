import os
import sys
import datetime

# Disable tensorflow gpu logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

timestamp = datetime.datetime.now().strftime("%m-%d-%YT%H:%M:%S")

if '--segment_length' in sys.argv:
    segment_length = int(sys.argv[sys.argv.index('--segment_length')+1])
else:
    segment_length = 8*60

if '--step' in sys.argv:
    step = int(sys.argv[sys.argv.index('--step')+1])
else:
    step = 60

if '--epochs' in sys.argv:
    epochs = int(sys.argv[sys.argv.index('--epochs')+1])
else:
    epochs = 40

if '--batch_size' in sys.argv:
    batch_size = int(sys.argv[sys.argv.index('--batch_size')+1])
else:
    batch_size = 32

if '--test_hours' in sys.argv:
    test_hours = int(sys.argv[sys.argv.index('--test_hours')+1])
else:
    test_hours = None

if '--k_folds' in sys.argv:
    k_folds = int(sys.argv[sys.argv.index('--k_folds')+1])
else:
    k_folds = 1

if '--model_path' in sys.argv:
    model_path = sys.argv[sys.argv.index('--model_path')+1]
else:
    model_path = None

if '--optimizer' in sys.argv:
    optimizer = sys.argv[sys.argv.index('--optimizer')+1]
else:
    optimizer = 'adam'

if '--learning_rate' in sys.argv:
    learning_rate = float(sys.argv[sys.argv.index('--learning_rate')+1])
else:
    learning_rate = None

if '--history_graph' in sys.argv:
    history_graph = sys.argv[sys.argv.index('--history_graph')+1]
else:
    history_graph = None

if '--dropout' in sys.argv:
    dropout = float(sys.argv[sys.argv.index('--dropout')+1])
else:
    dropout = None

madrs = '-m' in sys.argv or '--madrs' in sys.argv
verbose = '-v' in sys.argv or '--verbose' in sys.argv
do_load = '--load' in sys.argv

if '--early_stop' in sys.argv:
    early_stop = int(sys.argv[sys.argv.index('--early_stop')+1])
else:
    early_stop = None

_type = madrs and '-MADRS' or ''

identifier = f'Conv1D{_type}_{timestamp}_{segment_length}_{step}_{epochs}_{batch_size}'

if optimizer and learning_rate:
    identifier = f'{identifier}_opt-{optimizer}_lr-{learning_rate}'

if '--logfile' in sys.argv:
    logfile = sys.argv[sys.argv.index('--logfile')+1]
elif '--log' in sys.argv:
    logfile = f'../logs/python/{identifier}.log' 
else:
    logfile = None

if verbose:
    print('Segment length:', segment_length) 
    print('Step:', step)
