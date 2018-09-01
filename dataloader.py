import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from PIL import ImageFile
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True



def loadImgs(des_dir = "./data/",img_size=128,batchSize = 20):

    dataset = dset.ImageFolder(root=des_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size= batchSize,
                                             shuffle=True)

    return dataloader
