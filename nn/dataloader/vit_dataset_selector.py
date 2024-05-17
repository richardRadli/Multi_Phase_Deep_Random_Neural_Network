from torchvision import datasets, transforms


def select_dataset(dataset_name):
    dataset_classes = {
        'cifar10': datasets.CIFAR10,
        'mnist': datasets.MNIST
    }

    if dataset_name not in dataset_classes:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported")

    return dataset_classes[dataset_name]


def get_transforms(num_channels):
    if num_channels == 1:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    elif num_channels == 3:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        raise NotImplementedError(f"'{num_channels}' is not supported")

    return transform


def create_dataset(train: bool, dataset_info):
    dataset_class = select_dataset(dataset_info.get("dataset_name"))

    transform = get_transforms(dataset_info.get("num_channels"))

    dataset = dataset_class(
        root=dataset_info.get("dataset_original_files"), train=train, download=True, transform=transform
    )

    return dataset
