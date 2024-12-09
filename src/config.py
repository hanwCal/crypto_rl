import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

HYPERPARAMS = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'min_epsilon': 0.01,
    'buffer_size': 10000,
    'batch_size': 64,
    'train_start': 1000,
}

EPISODES = 500
MAX_STEPS = 200
