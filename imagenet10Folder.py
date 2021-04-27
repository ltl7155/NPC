from  torchvision.datasets import DatasetFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

# from torchvision.datasets.folder import default_loader

from PIL import Image

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
    
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def default_loader(path: str) -> Any:

    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

    
class imagenet10Folder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

#     def __init__(
#             self,
#             root: str,
#             loader: Callable[[str], Any] = default_loader,
#             extensions: Optional[Tuple[str, ...]] = None,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             is_valid_file: Optional[Callable[[str], bool]] = None,
#     ) -> None:
        
#         super(imagenet10Folder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
#                                           transform=transform,
#                                           target_transform=target_transform,
#                                           is_valid_file=is_valid_file)
        
        
#         self.loader = loader
#         self.classes = self.classes[:10]
#         tmp = {}
#         for k, v in self.class_to_idx.items():  
#             if v >=0 and v < 10:
#                 tmp[k] = v
#         self.class_to_idx = tmp     
#         tmp = []
#         for p, t in self.samples:
#             if t >=0 and t < 10:
#                 tmp.append((p, t))
#         self.samples = tmp
#         self.targets = [s[1] for s in self.samples]

    def __init__(
                self,
                root: str,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None,
        ):
        

        super(imagenet10Folder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        
        
        self.loader = loader
        self.classes = self.classes[:10]
        tmp = {}
        for k, v in self.class_to_idx.items():  
            if v >=0 and v < 10:
                tmp[k] = v
        self.class_to_idx = tmp     
        tmp = []
        for p, t in self.samples:
            if t >=0 and t < 10:
                tmp.append((p, t))
        self.samples = tmp
        self.targets = [s[1] for s in self.samples]

        
                