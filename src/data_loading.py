import io

from dataclasses import dataclass
from folder import DatasetFolder
from torch.utils.data import DataLoader

from configuration import Configuration
from preprocessing import Preprocessor

def image_loader(image):
    image = io.imread(image)
    return image

class ImageFolder(DatasetFolder):
    def __init__(
        self, 
        root, 
        transform=None, 
        target_transform=None,
        loader=image_loader
        ):
        
        super(
            ImageFolder, 
            self
            ).__init__(
                root, 
                loader, 
                ['.png'],
                transform=transform,
                target_transform=target_transform
                )
        
        self.imgs = self.samples       


@dataclass
class DataSetLoader():
    config: Configuration
    preprocessor: Preprocessor
    extra_data_loader_args: dict

    def get_training_data_loader(self, data_path: str) -> ImageFolder:
        return ImageFolder(
            root = data_path, 
            transform = self.preprocessor.get_training_data_transformer()
        )

    def get_testing_data_loader(self, data_path: str) -> DataLoader:
        testing_data = ImageFolder(
            root=data_path, 
            transform=self.preprocessor.get_testing_data_transformer()
        )

        return DataLoader(
            testing_data, 
            batch_size=self.config.test_batch_size, 
            shuffle=True, 
            **self.extra_data_loader_args
        )

    def get_hard_negative_testing_data_loader(self, data_path: str) -> DataLoader:
        hard_negative_testing_data = ImageFolder(
            root=data_path, 
            transform=self.preprocessor.get_hard_negative_testing_data_transformer()
        )

        return DataLoader(
            hard_negative_testing_data, 
            batch_size=self.config.validate_batch_size, 
            shuffle=False, 
            **self.extra_data_loader_args
        )

