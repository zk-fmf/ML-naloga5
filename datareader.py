import numpy as np
import torch.utils.data as tdata

class DataReader(tdata.Dataset):
    def __init__(self, main_path_to_data, data_info):
        super(DataReader, self).__init__()
        self.data = data_info 
        self.num_sample = len(self.data)
        self.main_path_to_data = main_path_to_data

    def __len__(self):
        return self.num_sample
        
    def __getitem__(self, n):  
        filename, label = self.data[n]
        path_to_file = self.main_path_to_data + filename
        img = np.load(path_to_file)

        return img, np.float32([label])
    