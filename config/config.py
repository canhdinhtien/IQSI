import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.run_name = ""
    config.seed = 42
    config.logdir = "logs"

    config.num_epochs = 100
    config.save_freq = 20

    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"

    config.allow_tf32 = True
    config.resume_from = ""

    config.use_lora = True
    config.n_shot = 16

    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "sd2-community/stable-diffusion-2-1"

    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 32
    train.use_8bit_adam = False
    train.lr = 3e-4
    
    config.path = path = ml_collections.ConfigDict()
    path.real_train_dir = "real_data"
    path.fewshot_dir = "real_data/train/dtd/fewshot_real_images"
    path.synthesis_dir = "synthetic_data"


    return config

    
