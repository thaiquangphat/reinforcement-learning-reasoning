from .model_0 import QueryPathRLV0
from .model_0_1 import QueryPathActorCritic
from .model_0_2 import QueryPathRLPPO
from .model_1 import QueryPathRLV1
from .model_2 import HierarchicalQueryPathRLV1
from .model_3 import HierarchicalQueryPathRLV2

QUERY_PATH_RL = {
    "QueryPathRLV0": QueryPathRLV0,
    "QueryPathRLV01": QueryPathActorCritic,
    "QueryPathRLV02": QueryPathRLPPO,
    "QueryPathRLV1": QueryPathRLV1,
    "QueryPathRLV2": HierarchicalQueryPathRLV1,
    "QueryPathRLV3": HierarchicalQueryPathRLV2
}