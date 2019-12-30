import random
import re
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
import torchvision.transforms.functional as F


class CreateDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt

        self.img_source_paths, self.img_source_size = make_dataset(opt.img_source_file)
        self.img_target_paths, self.img_target_size = make_dataset(opt.img_target_file)

        if self.opt.isTrain:
            self.lab_source_paths, self.lab_source_size = make_dataset(opt.lab_source_file)
            # for visual results, not for training
            self.lab_target_paths, self.lab_target_size = make_dataset(opt.lab_target_file)

        self.transform_augment = get_transform(opt, True)
        self.transform_no_augment = get_transform(opt, False)
        self.transform_labels = get_transform_label()

    def _read_source_depth(self, f_path, max_depth=1.28):
        arr = np.load(f_path).astype('float32')
        arr = np.where(arr >= max_depth, max_depth, arr)
        arr = arr[:, :, 0] / max_depth
        arr = arr[::-1, :]
        return Image.fromarray(arr)

    def _read_target_depth(self, filename, byteorder='>'):
        """Return image data from a raw PGM file as numpy array.
        Format specification: http://netpbm.sourceforge.net/doc/pgm.html
        """
        with open(filename, 'rb') as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()

            arr = np.frombuffer(buffer,
                                dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                                count=int(width)*int(height),
                                offset=len(header)
                                ).reshape((int(height), int(width)))
        except:# AttributeError:
            #raise ValueError("Not a raw PGM file: '%s'" % filename)
            arr = np.zeros((256, 256))
    
        return Image.fromarray(arr.astype('float32'))

    def __getitem__(self, item):
        index = random.randint(0, self.img_target_size - 1)
        img_source_path = self.img_source_paths[item % self.img_source_size]
        if self.opt.dataset_mode == 'paired':
            img_target_path = self.img_target_paths[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            img_target_path = self.img_target_paths[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_source = Image.open(img_source_path).convert('RGB')
        img_target = Image.open(img_target_path).convert('RGB')
        img_source = img_source.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)
        img_target = img_target.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)

        if self.opt.isTrain:
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            if self.opt.dataset_mode == 'paired':
                lab_target_path = self.lab_target_paths[item % self.img_target_size]
            elif self.opt.dataset_mode == 'unpaired':
                lab_target_path = self.lab_target_paths[index]
            else:
                raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)
            #lab_source = Image.open(lab_source_path)#.convert('RGB')
            lab_source = self._read_source_depth(lab_source_path)
            #lab_target = Image.open(lab_target_path)#.convert('RGB')
            lab_target = self._read_target_depth(lab_target_path)
            lab_source = lab_source.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)
            lab_target = lab_target.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)

            img_source, lab_source, scale = paired_transform(self.opt, img_source, lab_source)
            img_source = self.transform_augment(img_source)
            #lab_source = self.transform_no_augment(lab_source)
            lab_source = self.transform_labels(lab_source)
            #print(lab_source.size(), '-' * 20)

            img_target, lab_target, scale = paired_transform(self.opt, img_target, lab_target)
            img_target = self.transform_no_augment(img_target)
            #lab_target = self.transform_no_augment(lab_target)
            lab_target = self.transform_labels(lab_target)
            #print(lab_target.size(), '='*20)

            return {'img_source': img_source, 'img_target': img_target,
                    'lab_source': lab_source, 'lab_target': lab_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path, 'lab_target_paths': lab_target_path
                    }

        else:
            img_source = self.transform_augment(img_source)
            img_target = self.transform_no_augment(img_target)
            return {'img_source': img_source, 'img_target': img_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    }

    def __len__(self):
        return max(self.img_source_size, self.img_target_size)

    def name(self):
        return 'T^2Dataset'


def dataloader(opt):
    datasets = CreateDataset()
    datasets.initialize(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=opt.shuffle, num_workers=int(opt.nThreads))
    return dataset

def paired_transform(opt, image, depth):
    scale_rate = 1.0

    if opt.flip:
        n_flip = random.random()
        if n_flip > 0.5:
            image = F.hflip(image)
            depth = F.hflip(depth)

    if opt.rotation:
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BICUBIC)
            depth = F.rotate(depth, degree, Image.BILINEAR)

    return image, depth, scale_rate


def get_transform(opt, augment):
    transforms_list = []

    if augment:
        if opt.isTrain:
            transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    transforms_list += [
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    return transforms.Compose(transforms_list)

def get_transform_label():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
