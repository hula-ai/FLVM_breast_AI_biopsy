from torch.utils.data import Dataset, DataLoader


def get_dataloaders(
    train_dataset,
    test_dataset,
    val_dataset=None,
    train_sampler=None,
    val_sampler=None,
    batch_size: int = 64,
    njobs: int = 4,
):
    "Builds a set of dataloaders with a batch_size"

    # Instantiate dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        batch_size=batch_size,
        num_workers=njobs,
        sampler=train_sampler,
        persistent_workers=True,
    )

    if val_dataset is None:
        val_dataset = train_dataset
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=njobs,
        sampler=val_sampler,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=njobs,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader, test_dataloader
