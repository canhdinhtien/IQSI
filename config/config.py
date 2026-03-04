import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.run_name = ""                
    config.seed = 42
    config.logdir = "logs"
    config.mixed_precision = "fp16"     # "no", "fp16", "bf16"
    config.allow_tf32 = True
    config.resume_from = ""             

    config.dataset_name = "dtd"         
    config.n_shot = 16                  
    config.is_synth_train = False       
    config.is_pooled_fewshot = False    
    config.model_type = "clip"
    config.use_lora = True

    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "sd2-community/stable-diffusion-2-1"
    pretrained.revision = "main"
    pretrained.use_tiny_decoder = False

    config.classifier = classifier = ml_collections.ConfigDict()
    classifier.is_lora_image = True
    classifier.is_lora_text = False
    classifier.clip_download_dir = "model_clip"
    classifier.clip_version = "ViT-B/16"
    classifier.precomputed_text_embs_path = "weights\dtd_text_emb.pt"

    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 32
    train.num_epochs = 100
    train.num_epochs_warm_up = 20
    train.save_freq = 20
    train.num_checkpoint_limit = 5     
    train.use_8bit_adam = False
    train.lr = 1e-4
    train.is_rand_aug = True
    train.target_label=None
    train.n_img_per_cls=None
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999      
    train.adam_weight_decay = 1e-2
    train.adam_epsilon = 1e-10 # Default is 1e-8
    train.lamda1 = 10
    train.lamda2 = 10
    train.lamda3 = 1
    train.lamda4 = 10
    train.lamda5 = 1

    config.path = path = ml_collections.ConfigDict()
    path.real_train_dir = "real_data"   
    path.real_test_dir = "real_data"
    path.metadata_dir = "metadata"    
    path.fewshot_dir = "real_data/train/dtd/fewshot_real_images"
    path.synthesis_dir = "synthetic_data"

    return config