from src import dataset, experiment_logs

train_dataset = dataset.PatchifySegmentationDataset(
    txt_file="train_sample.txt",
    images_dir="/data/integracar/amostras_car_orotofoto/",
    masks_dir="/data/integracar/amostras_car_mask/",
    image_size=1024,
    patch_size=512,
    transform=None,
    use_patches=True
)

logger = experiment_logs.ExperimentLogger(
    experiment_name="teste",
    config=None,
    use_wandb=False
)

