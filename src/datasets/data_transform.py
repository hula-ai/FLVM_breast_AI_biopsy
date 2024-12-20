import random
import numpy as np
import gc
import torch
import os

from PIL import Image
from torchvision import transforms


class IntensityShift(object):
    def __init__(self, intensity_range):
        assert isinstance(intensity_range, tuple)
        self.intensity_range = intensity_range

    def __call__(self, image):
        shift_value = random.randint(
            self.intensity_range[0], self.intensity_range[1])

        image = np.array(image)
        image = image + shift_value
        image = np.clip(image, a_min=0, a_max=255)

        image = Image.fromarray(image.astype(np.uint8))

        return image
    

class LLaVA_transform(object):

    def __init__(self, image_processor, model_config):
        self.image_processor = image_processor
        self.model_config = model_config

    def __call__(self, image):
        from llava.mm_utils import process_images
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]      

        return image_tensor


def build_transform_dict(input_size=224, 
                         norm_mean=[0.485, 0.456, 0.406], 
                         norm_std=[0.229, 0.224, 0.225]):
    transform_dict = {
        'train': transforms.Compose([
            transforms.RandomRotation(180),
            transforms.Resize((input_size, input_size)),
            # transforms.RandomResizedCrop(
            #     input_size, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(25, scale=(0.8, 1.2)),
            IntensityShift((-20, 20)),            
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.Resize(int(input_size/7*8)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.Resize(int(input_size/7*8)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
    }

    return transform_dict


def build_transform_dict_blip2(model_name, model_type, input_size=224):
    from omegaconf import OmegaConf
    from lavis.models import load_preprocess    
    from lavis.common.registry import registry

    model_cls = registry.get_model_class(model_name)
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    preprocess_cfg = cfg.preprocess

    vis_processors, txt_processors = load_preprocess(preprocess_cfg)

    transform_dict = {
        'train': vis_processors['train'].build(image_size=input_size).transform,
        'val': vis_processors['eval'].build(image_size=input_size).transform,
        'test': vis_processors['eval'].build(image_size=input_size).transform
    }

    return transform_dict, txt_processors


def build_transform_dict_openclip(model_name, pretrained_data=None):
    import open_clip

    if pretrained_data is not None:
        _, preprocess_train, preprocess_val = \
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained_data)
    else:
        _, preprocess_train, preprocess_val = \
            open_clip.create_model_and_transforms(model_name)
    
    tokenizer = open_clip.get_tokenizer(model_name)

    return {
        'train': preprocess_train,
        'val': preprocess_val,
        'test': preprocess_val
    }, tokenizer


def build_transform_dict_pubmedclip():
    from transformers import CLIPProcessor, CLIPModel    

    class PubmedCLIPTransform(object):
        def __init__(self):
            self.processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
            self.tokenizer = self.processor.tokenizer
        
        def __call__(self, sample):
            return self.processor.image_processor(sample)['pixel_values'][0]

    preprocess_transform = PubmedCLIPTransform()

    return {
        'train': preprocess_transform,
        'val': preprocess_transform,
        'test': preprocess_transform
    }, preprocess_transform.tokenizer


def build_transform_dict_llava(model_path, model_base):
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    model_config = model.config

    # Free Model
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
        

    return {
        'train': None,
        'val': LLaVA_transform(image_processor, model_config),
        'test': LLaVA_transform(image_processor, model_config),
    }, tokenizer


def build_transform_dict_mamm(input_size=(1152, 896)):
    transform_dict = {
        'train': transforms.Compose([            
            transforms.Resize(input_size),
            # transforms.RandomResizedCrop(
            #     input_size, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(25, scale=(0.8, 1.2), shear=12),            
            IntensityShift((-20, 20)),            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([            
            transforms.Resize(input_size),            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([            
            transforms.Resize(input_size),            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return transform_dict
