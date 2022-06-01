class Config:


    INPUT_FILE_TRAIN = "./data/train_parall3.csv"
    INPUT_FILE_EVAL = "./data/dev_parall3.csv"

    OUTPUT_DIR = "output"
    LOG_DIR = "log"
    FILE_DIR = "files"
    MODEL_DIR = "models"
    RUN_ID = "6"

    MAX_LEN = 125
    EMBED_DIM = 240
    OUTPUT_DIM = 64
    INTER_EVAL_BATCH_SIZE = 1000
    DEVICE="cuda"

    TRIPLET_MINE_EVERY_N_STEPS = 5
    EPOCHS = 100
    BATCH_SIZE = 128
    DEBUG = False
    ACCURACY_CHECK_FREQ = 2