from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os 
import matplotlib.image as mpimg
import scipy.misc as misc


def load_data(dataset_loc, partition_type):
    file_names = sorted(os.listdir(dataset_loc))
    meta_file_name = dataset_loc + "list_eval_partition.txt"
    f = open(meta_file_name, "r")
    file_names = []

    for line in f:
        file_partition = line.split(" ")[1]
        if int(file_partition) != int(partition_type):
            continue
        png_file = line.split(" ")[0]
        file_name = dataset_loc + png_file
        file_names.append(file_name)
       
    #print(file_names[0], len(file_names))
    return file_names     

class Celeba_Dataset(Dataset):
    def __init__(self, dataset_loc, partition=0, img_dim=128):
        self.file_names = load_data(dataset_loc, partition)
        self.partition = partition   
        self.img_dim = img_dim
        self.height = 218
        self.width = 178
        self.center_height = int((self.height - self.width)/ 2)  
        print("length of dataset {}".format(len(self.file_names)))           

    def imread(self, file_name):
        img = misc.imread(file_name)
        img = img[self.center_height:self.center_height+self.width, :]
        img = misc.imresize(img, (self.img_dim, self.img_dim) )
        img = img.astype(np.float32) # / 255 * 2 - 1 
        return img

    def __getitem__(self, item):
        file_name = self.file_names[item]
        img = self.imread(file_name)
        img = np.transpose(img, (2,0,1))
        label =  np.zeros((1, 1))
        
        return img, label

    def __len__(self):
        return len(self.file_names)


if __name__ == '__main__':

    train = Celeba_Dataset("/media/adityasan92/New Volume/dataset/img_align_celeba/")
    #test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        data = np.transpose(data, (1,2,0))
        misc.imshow(data)
        print(label.shape)
        break
  