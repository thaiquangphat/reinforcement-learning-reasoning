from .model_1 import QueryPathRLV1
from .model_2 import HierarchicalQueryPathRLV1
from .model_3 import HierarchicalQueryPathRLV2

QUERY_PATH_RL = {
    "QueryPathRLV1": QueryPathRLV1,
    "QueryPathRLV2": HierarchicalQueryPathRLV1,
    "QueryPathRLV3": HierarchicalQueryPathRLV2
}