# 把coordinate[64, 64, 64, 3]切割成token序列

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# x = torch.ones([64,64,64,3])
#[64,64,64,3]切片 --> 
def split(img, block_size, stride):
    blocks = []
    for i in range(0, img.shape[0]-block_size+1, stride):
        for j in range(0, img.shape[1]-block_size+1, stride):
            for k in range(0, img.shape[2]-block_size+1, stride):
                block = img[i:i+block_size, j:j+block_size, k:k+block_size]
                
                block = block.contiguous().view(-1, block.shape[3])
                
                blocks.append(block)
    return blocks
                
    
#3d卷积
class PatchEmbed(nn.Module):
    def __init__(self, img, pool_size, pool_stride, block_size, block_stride) -> None:
        super().__init__()
        self.pool_stride = pool_stride
        self.block_stride = block_stride
        self.block_size = block_size
        self.block_stride = block_stride
        self.img_pool_size = ((img.shape[0]-pool_size) // pool_stride +1, (img.shape[1]-pool_size) // pool_stride +1, (img.shape[2]-pool_size) // pool_stride +1)
        
        self.pool = nn.MaxPool3d(kernel_size=self.block_size, stride=self.pool_stride)
        
    def forward(self,x):
        #[d,h,w,c] --> [c,d,h,w]
        x = self.pool(x.permute(3,0,1,2))
        #[c,d,h,w] --> [d,h,w,c]
        x = split(x.permute(1,2,3,0), self.block_size, self.block_stride)
        
        return x


#自定义datasets

class Mydataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data
    
    def __len__(self):
        return len(self.data_list)




    