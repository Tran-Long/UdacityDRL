import torch

class Config:
    SEED = 3737

    BATCH_SIZE = 64
    BUFFER_SIZE = 100000
    TAU = 1e-3
    NETWORK_ARCHITECTURE = [64, 128, 64]
    LEARN_EVERY = 4
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9995
    GAMMA = 0.99

    LR = 0.001
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    NUM_EPISODE = 20000
    MAX_TIMESTEP = 300

    CHECKPOINT = "checkpoint.pth"