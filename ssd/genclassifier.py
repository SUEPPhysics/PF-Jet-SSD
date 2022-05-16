import h5py
import torch
import torch.cuda as tcuda
import torch.nn.functional as F


class SUEPDataset(torch.utils.data.Dataset):

    def __init__(self,
                 rank,
                 hdf5_source_path,
                 input_dimensions,
                 flip_prob=None):
        """Generator for calorimeter and jet data"""

        self.rank = rank
        self.source = hdf5_source_path
        self.channels = input_dimensions[0]  # Number of channels
        self.width = input_dimensions[1]  # Width of input
        self.height = input_dimensions[2]  # Height of input
        self.flip_prob = flip_prob

    def __getitem__(self, index):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        idx_PFCand_Eta = tcuda.LongTensor([self.PFCand_Eta[index]],
                                          device=self.rank)
        idx_PFCand_Phi = tcuda.LongTensor([self.PFCand_Phi[index]],
                                          device=self.rank)
        val_PFCand_PT = tcuda.FloatTensor(self.PFCand_PT[index],
                                          device=self.rank)

        calorimeter = self.process_images(idx_PFCand_Eta,
                                          idx_PFCand_Phi,
                                          val_PFCand_PT)
        # Set labels
        labels = tcuda.FloatTensor(self.labels[index], device=self.rank)
        labels = self.process_labels(labels)

        tracks = torch.tensor(len(self.PFCand_PT[index]),
                              device=self.rank) + .0

        if self.flip_prob:
            if torch.rand(1) < self.flip_prob:
                calorimeter = self.flip_image(calorimeter, vertical=True)
            if torch.rand(1) < self.flip_prob:
                calorimeter = self.flip_image(calorimeter, vertical=False)
        return calorimeter, tracks, labels

    def __len__(self):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        return self.dataset_size

    def flip_image(self, image, vertical=True):
        if vertical:
            axis = 1
        else:
            axis = 2
        image = torch.flip(image, [axis])
        return image

    def normalize(self, tensor):
        m = torch.mean(tensor)
        s = torch.std(tensor)
        return tensor.sub(m).div(s)

    def open_hdf5(self):
        self.hdf5_dataset = h5py.File(self.source, 'r')
        self.PFCand_Eta = self.hdf5_dataset['PFCand_Eta']
        self.PFCand_Phi = self.hdf5_dataset['PFCand_Phi']
        self.PFCand_PT = self.hdf5_dataset['PFCand_PT']
        self.labels = self.hdf5_dataset['labels']
        self.dataset_size = len(self.labels)

    def process_images(self, idx_Eta, idx_Phi, idx_Pt):
        v0 = torch.zeros(idx_Eta.size(1), dtype=torch.long).cuda(self.rank)
        idx_channels = v0.unsqueeze(0)
        i = torch.cat((idx_channels, idx_Eta, idx_Phi), 0)
        v = idx_Pt
        pixels = torch.sparse.FloatTensor(i, v, torch.Size([self.channels,
                                                            self.width,
                                                            self.height]))
        pixels = pixels.to_dense()
        pixels = self.normalize(pixels)
        return pixels

    def process_labels(self, labels_raw):
        labels_reshaped = labels_raw.reshape(-1, 5)
        labels = torch.tensor(0 + (1. in labels_reshaped[:, 0]))
        return labels
