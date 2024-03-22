# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri, Max Kiss
# =============================================================================


import numpy as np
import torch
import pathlib
import warnings
from abc import ABC, abstractmethod, ABCMeta

from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.data_loaders.deteCT import deteCT

from LION.experiments.ct_experiments import Experiment

class ExperimentalNoiseDenoising(Experiment):

    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "Experimental Noise Denoising experiment with noisy sinograms from mode1 as input data and sinograms from mode2 as target data"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()
        
        param.log_transform = False # if we do a sino2sino this should not be done

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters(dataset)

        # Change the data loader to be sino2sino
        param.data_loader_params.task = "sino2sino"
        param.data_loader_params.input_mode = "mode1"
        param.data_loader_params.target_mode = "mode2"
        
        return param

class ArtificialNoiseDenoising(Experiment):

    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "Artificial Noise Denoising experiment with noisy sinograms generated from mode2 as input data and sinograms from mode2 as target data"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()
        
        param.log_transform = False # if we do a sino2sino this should not be done

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters(dataset)

        # Change the data loader to be sino2sino
        param.data_loader_params.task = "sino2sino"
        param.data_loader_params.input_mode = "mode2"
        # TODO MK: Look into how to set up the noise simulation in the best way
        param.data_loader_params.target_mode = "mode2"
        
        return param
