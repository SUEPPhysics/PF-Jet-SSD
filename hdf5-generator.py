import argparse
import awkward as ak
import h5py
import numpy as np
import itertools
import os
import uproot
import yaml
import vector

vector.register_awkward()

from utils import IsReadableDir
from tqdm import tqdm
from typing import Dict, Iterable, List, Optional, Tuple


class PhysicsConstants():

    def __init__(self, config=None):

        self.ht_threshold = 500.

        self.eta_span = (-2.5, 2.5)
        self.eta_steps = 500
        self.phi_span = (-np.pi, np.pi)
        self.phi_steps = 500
        self.set_edges()
        
        if config is not None: self.config = yaml.safe_load(open(config))

    def set_edges(self):

        self.edges_eta = np.linspace(self.eta_span[0],
                                     self.eta_span[1],
                                     self.eta_steps)
        self.edges_phi = np.linspace(self.phi_span[0],
                                     self.phi_span[1],
                                     self.phi_steps)

    def get_edges(self) -> Tuple[List[int], List[int]]:
        return self.edges_eta, self.edges_phi


class HDF5Generator:

    def __init__(self,
                 hdf5_dataset_path: str,
                 hdf5_dataset_size: int,
                 config: str,
                 verbose: bool = True):

        self.constants = PhysicsConstants(config)
        self.jet_size = self.constants.config['ssd_settings']['object_size']
        self.edges_eta, self.edges_phi = self.constants.get_edges()
        self.hdf5_dataset_path = hdf5_dataset_path
        self.hdf5_dataset_size = hdf5_dataset_size
        self.verbose = verbose

    def create_hdf5_dataset(self, suep: Iterable, qcd: Iterable):

        if self.verbose:
            progress = tqdm(total=self.hdf5_dataset_size,
                            desc=('Processing {}'.format(
                                self.hdf5_dataset_path)))

        # Create the HDF5 file.
        with h5py.File(self.hdf5_dataset_path, 'w') as hdf5_dataset:

            hdf5_labels = hdf5_dataset.create_dataset(
                name='labels',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

            hdf5_PFCand_Eta = hdf5_dataset.create_dataset(
                name='PFCand_Eta',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

            hdf5_PFCand_Phi = hdf5_dataset.create_dataset(
                name='PFCand_Phi',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

            hdf5_PFCand_PT = hdf5_dataset.create_dataset(
                name='PFCand_PT',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

            get_suep = itertools.cycle([True, False])
            
            # debug
            masses, pts = [], []
            
            for i in range(self.hdf5_dataset_size):

                if next(get_suep):
                    event_details = next(suep)
                else:
                    event_details = next(qcd)

                pt = event_details.get('pt')
                eta = event_details.get('eta')
                phi = event_details.get('phi')
                mass = event_details.get('mass')
                flag = event_details.get('flag')
                suep_eta = event_details.get('suep_eta')
                suep_phi = event_details.get('suep_phi')

                jpt = event_details.get('jpt')
                jeta = event_details.get('jeta')
                jphi = event_details.get('jphi')
                jmass = event_details.get('jmass')

                px_eta, px_phi, values = self.get_energy_map(eta, phi, pt)

                # Get SUEP labels
                suep_label = self.get_suep_label(eta[flag], phi[flag], 
                                                 pt[flag], mass[flag], 
                                                 suep_eta, suep_phi)

                # Get jet labels
                labels = self.get_jet_labels(jeta,
                                             jphi,
                                             jpt,
                                             jmass,
                                             suep_label,
                                             suep_eta, suep_phi)

                # Concatenate labels
                if suep_label: labels.append(suep_label)
                    
                # make sure that labels are randomly ordered
                labels = np.random.permutation(labels)

                # Flatten the labels array and write it to the dataset
                hdf5_labels[i] = np.hstack(labels)
                

                hdf5_PFCand_Eta[i] = px_eta
                hdf5_PFCand_Phi[i] = px_phi
                hdf5_PFCand_PT[i] = values

                if self.verbose:
                    progress.update(1)
        if self.verbose:
            progress.close()

    def get_energy_map(self,
                       etas: np.ndarray,
                       phis: np.ndarray,
                       values: np.ndarray) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
        """Translate eta/phi to pixel coordinates"""
        img, _, _ = np.histogram2d(etas,
                                   phis,
                                   bins=[self.edges_eta, self.edges_phi],
                                   weights=values)
        bins = np.argwhere(img)
        indices_eta = bins[:, 0]
        indices_phi = bins[:, 1]
        values = img[indices_eta, indices_phi]
        return indices_eta, indices_phi, values

    def get_jet_labels(self,
                       etas: np.ndarray,
                       phis: np.ndarray,
                       pts: np.ndarray,
                       mass: np.ndarray,
                       label: Optional[list],
                       suep_eta: int,
                       suep_phi: int) -> list:
        """Returns labels for jets"""
        coordinates = []
        for e, p, pt, m in zip(etas, phis, pts, mass):
            x = np.argmax(self.edges_eta >= e)
            y = np.argmax(self.edges_phi >= p)
            
            # exclude jets within dR < 08 of a SUEP
            if label and np.sqrt((suep_eta - e)**2 + (suep_phi - p)**2) < 0.8: continue
            
            coordinates.append([2, x, y, pt, m])
        return coordinates
    
    def eta_to_pixel(self, eta):
        """ Linear transformation from eta to pixels """
        slope = (self.constants.eta_steps/(self.constants.eta_span[1] - self.constants.eta_span[0]))
        return slope*(eta - self.constants.eta_span[0])
    
    def phi_to_pixel(self, phi):
        """ Linear transformation from phi to pixels """
        slope = (self.constants.phi_steps/(self.constants.phi_span[1] - self.constants.phi_span[0]))
        return slope*(phi - self.constants.phi_span[0])

    def get_suep_label(self,
                       etas: np.ndarray,
                       phis: np.ndarray,
                       pts: np.ndarray,
                       mass: np.ndarray,
                       suep_eta: int,
                       suep_phi: int) -> Optional[list]:
        """Returns labels for suep"""
        
        if len(pts) == 0: return None

        # calculate pt and mass
        tracks = vector.array({
            "eta": etas, "phi": phis, "pt": pts, "M": mass
        })
        SUEP = vector.obj(px=0,py=0,pz=0,E=0)
        for track in tracks: SUEP = SUEP + track
        m = SUEP.mass
        pt = SUEP.pt
        
        return [1, self.eta_to_pixel(suep_eta), self.phi_to_pixel(suep_phi), pt, m]


class EventGenerator():

    def __init__(self, path: str):
        self.constants = PhysicsConstants()
        self.path = path
        self.root_files = self.get_files_from_dir(path)

    def __iter__(self):

        for root_file in self.root_files:

            rf = uproot.open('{}{}'.format(self.path, root_file))
            if not len(rf.keys()):
                continue
            tree = rf['mmtree/tree']

            hts = tree['ht'].array()
            pts = tree['PFcand_pt'].array()
            mass = tree['PFcand_m'].array()
            phis = tree['PFcand_phi'].array()
            etas = tree['PFcand_eta'].array()
            is_suep = tree['PFcand_fromsuep'].array()
            suep_etas = tree['SUEP_Truth_eta'].array()
            suep_phis = tree['SUEP_Truth_phi'].array()
            
            # track selections
            ### FIXME: missing dz, dzErr, fromPV for full selection on tracks
            cut = (pts > 0.7) & (abs(etas) <= 2.5)
            pts = pts[cut]
            mass = mass[cut]
            phis = phis[cut]
            etas = etas[cut]
            
            n_jet = tree['n_fatjet'].array()
            jet_pts = tree['FatJet_pt'].array()
            jet_mass = tree['FatJet_mass'].array()
            jet_etas = tree['FatJet_eta'].array()
            jet_phis = tree['FatJet_phi'].array()
            

            for i, ht in enumerate(hts):

                # event selections
                if ht < self.constants.ht_threshold or n_jet[i] < 2:
                    continue

                yield {'pt': np.array(pts[i]),
                       'eta': np.array(etas[i]),
                       'phi': np.array(phis[i]),
                       'mass': np.array(mass[i]),
                       'flag': np.array(is_suep[i]),
                       'suep_eta': np.array(suep_etas[i]),
                       'suep_phi': np.array(suep_phis[i]),
                       'jpt': np.array(jet_pts[i]),
                       'jeta': np.array(jet_etas[i]),
                       'jphi': np.array(jet_phis[i]),
                       'jmass': np.array(jet_mass[i])}

    def get_files_from_dir(self, path: str) -> List[str]:
        return [i for i in os.listdir(path) if i.endswith(".root")]


def parse_config_for_dataset_sizes(config: str) -> List[int]:
    c = yaml.safe_load(open(config))
    return c['dataset']['size']


def main(path_suep: str,
         path_qcd: str,
         path_target: str,
         config: str,
         verbose: bool = True):

    dataset_sizes = parse_config_for_dataset_sizes(config)

    eg_suep = iter(EventGenerator(path_suep))
    eg_qcd = iter(EventGenerator(path_qcd))

    for i, dataset_size in enumerate(dataset_sizes):

        generator = HDF5Generator(
            hdf5_dataset_path='{}/SUEPPhysicsSSD_{}.h5'.format(path_target, i),
            hdf5_dataset_size=dataset_size,
            config=config,
            verbose=verbose)

        generator.create_hdf5_dataset(eg_suep, eg_qcd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Process SUEP and QCD ROOT files and store events and labels in to H5')

    parser.add_argument('source_dir_suep',
                        action=IsReadableDir,
                        help='SUEP files source folder',
                        type=str)

    parser.add_argument('source_dir_qcd',
                        action=IsReadableDir,
                        help='QCD files source folder',
                        type=str)

    parser.add_argument('target_dir',
                        action=IsReadableDir,
                        help='H5 files target folder',
                        type=str)

    parser.add_argument('-c', '--config',
                        action=IsReadableDir,
                        default='./ssd-config.yml',
                        dest='config',
                        help='Configuration file path',
                        type=str)

    parser.add_argument('-v', '--verbose',
                        action="store_true",
                        help='Speak')

    args = parser.parse_args()

    main(args.source_dir_suep,
         args.source_dir_qcd,
         args.target_dir,
         args.config,
         args.verbose)
