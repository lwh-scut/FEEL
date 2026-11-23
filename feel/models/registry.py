from feel.models.eeg.eegnet import EEGNet
from feel.models.eeg.nsal_dgat import Domain_adaption_model
from feel.models.eeg.rgnn import RGNN
from feel.models.eeg.tsception import TSception
from feel.models.fnirs.dbjnet import DBJNet
from feel.models.fnirs.fnirs_net import fNIRSNet
from feel.models.fnirs.fnirs_transformer import fNIRS_T
from feel.models.fnirs.hsctn import HSCTN
from feel.models.eeg.acrnn import ACRNN
from feel.models.eeg.cdcn import CDCN
from feel.models.eeg.coral_dgcnn import CoralDgcnn
from feel.models.eeg.dann_dgcnn import DannDgcnn
from feel.models.eeg.dbn import DBN
from feel.models.eeg.dgcnn import DGCNN
from feel.models.eeg.fbstcnet import PowerAndConneMixedNet
from feel.models.eeg.gcbnet import GCBNet
from feel.models.eeg.gcbnet_bls import GCBNet_BLS
from feel.models.eeg.msmda import MSMDA

from feel.models.eeg_fnirs.e_fnet import EFNet
from feel.models.eeg_fnirs.gcn_ca_capsnet import GCN_CA_CapsNet
from feel.models.eeg_fnirs.tcresnet import DualModalEmotionNet_CustomSR

Model = {
    'EEGNet': EEGNet,
    'ACRNN': ACRNN,
    'CDCN': CDCN,
    'CoralDgcnn': CoralDgcnn,
    'DannDgcnn': DannDgcnn,
    'DBN': DBN,
    'DGCNN': DGCNN,
    'FBSTCNet': PowerAndConneMixedNet,
    'GCBNet': GCBNet,
    'GCBNet_BLS': GCBNet_BLS,
    'MsMda': MSMDA,
    'NSAL_DGAT': Domain_adaption_model,
    'RGNN': RGNN,
    'TSception': TSception,

    'DBJNet': DBJNet,
    'fNIRSNet': fNIRSNet,
    'fNIRS_Transformer': fNIRS_T,
    'HSCTN': HSCTN,

    'TCResNet': DualModalEmotionNet_CustomSR,
    'GCN_CA_CapsNet': GCN_CA_CapsNet,
    'E_FNet': EFNet,
}
