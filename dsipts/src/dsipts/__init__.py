# src/dispts/__init__.py

from .data_management.monash import Monash, get_freq
from .data_management.public_datasets import read_public_dataset
from .data_structure.data_structure import TimeSeries, Categorical
from .data_structure.utils import extend_time_df, beauty_string

from .models.RNN import RNN
from .models.LinearTS import LinearTS
from .models.Persistent import Persistent
from .models.D3VAE import D3VAE
from .models.DilatedConv import DilatedConv
from .models.TFT import TFT
from .models.Informer import Informer
from .models.VVA import VVA
from .models.VQVAEA import VQVAEA
from .models.CrossFormer import CrossFormer
from .models.Autoformer import Autoformer
from .models.PatchTST import PatchTST
from .models.Diffusion import Diffusion
from .models.DilatedConvED import DilatedConvED
from .models.TIDE import TIDE
from .models.ITransformer import ITransformer
from .models.TimeXER import TimeXER
from .models.TTM import TTM
from .models.Samformer import Samformer
from .models.Duet import Duet
from .models.Simple import Simple
from .models.TimesNet import TimesNet
from .models.TimeKAN import TimeKAN
from .version import __version__
try:
    import lightning.pytorch as pl
    from .models.base_v2 import Base
    OLD_PL = False
except ImportError:
    import pytorch_lightning as pl
    from .models.base import Base
    OLD_PL = True

__all__ = [
    # Data Management
    "Monash", "get_freq", "read_public_dataset",
    # Data Structure
    "TimeSeries", "Categorical", "extend_time_df", "beauty_string",
    # Models
    "RNN", "LinearTS", "Persistent", "D3VAE", "DilatedConv", "TFT",
    "Informer", "VVA", "VQVAEA", "CrossFormer", "Autoformer", "PatchTST",
    "Diffusion", "DilatedConvED", "TIDE", "ITransformer", "TimeXER",
    "TTM", "Samformer", "Duet", "Base", "Simple","TimesNet","TimeKAN"
]