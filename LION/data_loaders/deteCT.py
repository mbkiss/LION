# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Emilien Valat
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import warnings
from LION.utils.paths import DETECT_PROCESSED_DATASET_PATH
from LION.utils.parameter import Parameter
import LION.CTtools.ct_geometry as ctgeo
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk, nag_ls


class deteCT(Dataset):
    def __init__(self, mode, params: Parameter = None):
        if params is None:
            params = self.default_parameters()

        if params.sinogram_mode != params.reconstruction_mode:
            warnings.warn(
                "Sinogram mode and reconstruction mode don't match, so reconstruction is not from the sinogram you are getting... \n This should be an error, but I trust that you won what you are doing"
            )
        if params.query == "" and (
            not params.flat_field_correction or not params.dark_field_correction
        ):
            warnings.warn(
                "You are not using any detector query, but you are not using flat field or dark field correction. This is not recommended, as different detectors perform differently"
            )
        ### Defining the path to data
        self.path_to_dataset = params.path_to_dataset
        """
        The path_to_dataset attribute (pathlib.Path) points towards the folder
        where the data is stored
        """
        ### Defining the path to scan data
        self.path_to_scan_data = self.path_to_dataset.joinpath("scan_settings.json")
        ### Defining the path to the data record
        self.path_to_data_record = self.path_to_dataset.joinpath(
            f"default_data_records.csv"
        )

        ### Defining the data record
        self.data_record = pd.read_csv(self.path_to_data_record)
        """
        The data_record (pd.Dataframe) maps a slice index/identifier to
            - the sample index of the sample it belongs to
            - the number of slices expected
            - the number of slices actually sampled
            - the first slice of the sample to which a given slice belongs
            - the last slice of the sample to which a given slice belongs
            - the mix 
            - the detector it was sampled with
        """
        # Defining the sinogram mode
        self.sinogram_mode = params.sinogram_mode
        """
        The sinogram_mode (str) argument is a keyword defining what sinogram mode of the dataset to use:
                         |  mode1   |   mode2  |  mode3
            Tube Voltage |   90kV   |   90kV   |  60kV
            Tube power   |    3W    |    90W   |  60W
            Filter       | Thoraeus | Thoraeus | No Filter
        """
        # Defining the sinogram mode
        self.reconstruction_mode = params.reconstruction_mode
        """
        The reconstruction_mode (str) argument is a keyword defining what image mode of the dataset to use:
                         |  mode1   |   mode2  |  mode3
            Tube Voltage |   90kV   |   90kV   |  60kV
            Tube power   |    3W    |    90W   |  60W
            Filter       | Thoraeus | Thoraeus | No Filter
        """
        # Defining the task
        self.task = params.task
        """
        The task (str) argument is a keyword defining what is the dataset used for:
            - task == 'reconstruction' -> the dataset returns the sinogram and the reconstruction
            - task == 'segmentation' -> the dataset returns the reconstruction and the segmentation
            - task == 'joint' -> the dataset returns the sinogram, the reconstruction and the segmentation
        """

        assert self.task in [
            "reconstruction",
            "segmentation",
            "joint",
        ], f'Wrong task argument, must be in ["reconstruction", "segmentation", "joint"]'
        assert mode in [
            "train",
            "validation",
            "test",
        ], f'Wrong mode argument, must be in ["train", "validation", "test"]'
        # Defining the training mode
        self.mode = mode
        """
        The train (bool) argument defines if the dataset is used for training or testing
        """
        self.training_proportion = params.training_proportion
        self.validation_proportion = params.validation_proportion

        # Defining the train proportion
        """
        The training_proportion (float) argument defines the proportion of the training dataset used for training
        """
        self.transforms = None  # params.transforms

        ### We query the dataset subset
        self.slice_dataframe: pd.DataFrame
        """
        The slice_dataframe (pd.Dataframe) is the subset of interest of the dataset.
        The self.data_record argument becomes the slice_dataframe once we have queried it
        Example of query: 'detector==1'
        If the no query argument is passed, data_record_subset == data_record
        """
        if params.query:
            self.slice_dataframe = self.data_record.query(params.query)
        else:
            self.slice_dataframe = self.data_record

        ### We split the dataset between training and testing
        self.compute_sample_dataframe()
        self.sample_dataframe: pd.DataFrame
        """
        The sample_dataframe (pd.Dataframe) is a dataframe linking sample index to slices.
        It is used to partition the dataset on a sample basis, rather than on a slice basis,
        avoiding 'data leakage' between training, validation and testing
        """
        if self.mode == "train":
            self.sample_dataframe = self.sample_dataframe.head(
                int(len(self.sample_dataframe) * self.training_proportion)
            )
        elif self.mode == "validation":
            self.sample_dataframe = self.sample_dataframe.iloc[
                int(len(self.sample_dataframe) * self.training_proportion) : int(
                    len(self.sample_dataframe)
                    * (self.training_proportion + self.validation_proportion)
                )
            ]
        else:
            self.sample_dataframe = self.sample_dataframe.tail(
                int(
                    len(self.sample_dataframe)
                    * (1 - self.training_proportion - self.validation_proportion)
                )
            )

        self.slice_dataframe = self.slice_dataframe[
            self.slice_dataframe["sample_index"].isin(
                self.sample_dataframe["sample_index"].unique()
            )
        ]

        self.flat_field_correction = params.flat_field_correction
        self.dark_field_correction = params.dark_field_correction
        self.log_transform = params.log_transform
        self.do_recon = params.do_recon
        self.recon_algo = params.recon_algo

    @staticmethod
    def default_parameters():
        param = Parameter()
        param.path_to_dataset = DETECT_PROCESSED_DATASET_PATH
        param.sinogram_mode = "mode2"
        param.reconstruction_mode = "mode2"
        param.task = "reconstruction"
        param.training_proportion = 0.8
        param.validation_proportion = 0.1
        param.test_proportion = 0.1

        param.do_recon = False
        param.recon_algo = "nag_ls"
        param.flat_field_correction = True
        param.dark_field_correction = True
        param.log_transform = True
        param.query = ""
        return param

    @staticmethod
    def get_geometry():
        geo = ctgeo.Geometry.default_parameters()
        # From Max Kiss code
        SOD = 431.019989
        SDD = 529.000488
        detPix = 0.0748
        detSubSamp = 2
        detPixSz = detSubSamp * detPix
        nPix = 956
        det_width = detPixSz * nPix
        FOV_width = det_width * SOD / SDD
        nVox = 1024
        voxSz = FOV_width / nVox
        scaleFactor = 1.0 / voxSz
        SDD = SDD * scaleFactor
        SOD = SOD * scaleFactor
        detPixSz = detPixSz * scaleFactor

        geo.dsd = SDD
        geo.dso = SOD
        geo.detector_shape = [1, 956]
        geo.detector_size = [detPixSz, detPixSz * 956]
        geo.image_shape = [1, 1024, 1024]
        geo.image_size = [1, 1024, 1024]
        geo.image_pos = [0, -1, -1]
        geo.angles = -np.linspace(0, 2 * np.pi, 3600, endpoint=False) + np.pi
        return geo

    @staticmethod
    def get_operator():
        geo = deteCT.get_geometry()
        return make_operator(geo)

    def compute_sample_dataframe(self):
        unique_identifiers = self.slice_dataframe["sample_index"].unique()
        record = {"sample_index": [], "first_slice": [], "last_slice": []}
        for identifier in unique_identifiers:
            record["sample_index"].append(identifier)
            subset = self.slice_dataframe[
                self.slice_dataframe["sample_index"] == identifier
            ]
            record["first_slice"].append(subset["first_slice"].iloc[0])
            record["last_slice"].append(subset["last_slice"].iloc[0])
        self.sample_dataframe = pd.DataFrame.from_dict(record)

    def __len__(self):
        return (
            self.sample_dataframe["last_slice"].iloc[-1]
            - self.sample_dataframe["first_slice"].iloc[0]
            + 1
        )

    def __getitem__(self, index):
        slice_row = self.slice_dataframe.iloc[index]
        path_to_sinogram = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/{self.sinogram_mode}"
        )
        path_to_recontruction = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/{self.reconstruction_mode}"
        )
        path_to_segmentation = self.path_to_dataset.joinpath(
            f"{slice_row['slice_identifier']}/mode2"
        )

        if self.task == ["segmentation", "joint"]:
            segmentation = torch.from_numpy(
                np.load(path_to_segmentation.joinpath("segmentation.npy"))
            ).unsqueeze(0)

        if self.task in ["reconstruction", "joint"]:
            sinogram = torch.from_numpy(
                np.load(path_to_sinogram.joinpath("sinogram.npy"))
            ).unsqueeze(0)
            if self.flat_field_correction:
                flat = torch.from_numpy(
                    np.load(path_to_sinogram.joinpath("flat.npy"))
                ).unsqueeze(0)
            else:
                flat = 1
            if self.dark_field_correction:
                dark = torch.from_numpy(
                    np.load(path_to_sinogram.joinpath("dark.npy"))
                ).unsqueeze(0)
            else:
                dark = 0
            sinogram = (sinogram - dark) / (flat - dark)
            if self.log_transform:
                sinogram = -torch.log(sinogram)

            sinogram = torch.flip(sinogram, [2])

        if self.do_recon:
            op = deteCT.get_operator()
            if self.recon_algo == "nag_ls":
                reconstruction = nag_ls(op, sinogram, 100, min_constraint=0)
            elif self.recon_algo == "fdk":
                reconstruction = fdk(op, sinogram)
        else:
            reconstruction = torch.from_numpy(
                np.load(path_to_recontruction.joinpath("reconstruction.npy"))
            ).unsqueeze(0)

        if self.task == "reconstruction":
            return sinogram, reconstruction
        if self.task == "segmentation":
            return reconstruction, segmentation
        if self.task == "joint":
            return sinogram, reconstruction, segmentation