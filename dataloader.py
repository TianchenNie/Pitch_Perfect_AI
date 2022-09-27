
import torch
from torchvision import datasets, transforms
import numpy as np

def parse_data():
    BATCH_SIZE = 256
    PATH_TO_TEST = "./aps360_split_data/test"
    PATH_TO_TRAIN = "./aps360_split_data/train"
    PATH_TO_VALID = "./aps360_split_data/valid"
    transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5)),
                                    ])

    def loader(path_to_file):
        # print(torch.from_numpy(torch.load(path_to_file)).shape)
        return torch.from_numpy(torch.load(path_to_file))
    def is_valid_file(path_to_file):
        return True

    np.random.seed(50)
    test_data_folder = datasets.DatasetFolder(root=PATH_TO_TEST, loader = loader, is_valid_file=is_valid_file, transform=transform)


    test_loader = torch.utils.data.DataLoader(test_data_folder, batch_size=BATCH_SIZE, shuffle = True)


    train_data_folder = datasets.DatasetFolder(root=PATH_TO_TRAIN, loader = loader, is_valid_file=is_valid_file, transform=transform)
    # np.random.shuffle(train_data_folder)
    train_loader = torch.utils.data.DataLoader(train_data_folder, batch_size=BATCH_SIZE, shuffle=True)



    valid_data_folder = datasets.DatasetFolder(root=PATH_TO_VALID, loader = loader, is_valid_file=is_valid_file, transform=transform)
    # np.random.shuffle(valid_data_folder)
    valid_loader = torch.utils.data.DataLoader(valid_data_folder, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_loader, test_loader, valid_loader