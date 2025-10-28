from nixl._api import nixl_agent
from typing import Tuple
import os
import torch
import uuid
import json
import time
import numpy as np

from lmcache.logging import init_logger
logger = init_logger(__name__)

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
}

TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

def get_kv_cache_torch_dtype(
        cache_dtype,
        model_dtype = None) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype,
                          str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in STR_DTYPE_TO_TORCH_DTYPE:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype

def mla_enabled(model_config: "ModelConfig") -> bool:
    return (
        hasattr(model_config, "use_mla")
        and isinstance(model_config.use_mla, bool)
        and model_config.use_mla
    )

def dtype_to_str(dtype) -> str:
    """
    Convert a torch.dtype or string into a canonical string form.
    Examples:
        torch.float16 -> "torch.float16"
        "float32" -> "torch.float32"
        None -> "unknown"
    """
    if isinstance(dtype, torch.dtype):
        return f"torch.{dtype.__str__().split('.')[-1]}"
    elif isinstance(dtype, str):
        # Normalize to torch.<name> if it's just 'float16' etc.
        return dtype if dtype.startswith("torch.") else f"torch.{dtype}"
    else:
        return "unknown"


def str_to_dtype(s: str):
    """
    Convert a string like 'torch.float16' or 'float32' into torch.dtype.
    Examples:
        "torch.float16" -> torch.float16
        "float32" -> torch.float32
        "unknown" or invalid -> None
    """
    if not isinstance(s, str):
        return s  # already dtype or invalid type
    if s == "torch.auto":
        return "auto"
    s = s.strip()
    if s.startswith("torch."):
        s = s.split(".")[-1]

    try:
        return getattr(torch, s)
    except AttributeError:
        return None

cpu_buffer_map = {}

nixl_agent_map = {}
reg_descs_map = {}
xfer_descs_map = {}
xfer_handler_map = {}

def serialize_vllm_config(vllm_config) -> str:
    """
    Extracts key metadata from vLLM configuration and serializes to JSON.
    Handles torch.dtypes properly.
    """
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    num_layer = model_config.get_num_layers(parallel_config)
    num_mtp_layers = 0
    num_layer += num_mtp_layers
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()

    cache_dtype = vllm_config.cache_config.cache_dtype
    model_dtype = model_config.dtype

    # # Helper to safely convert dtype to string
    # def dtype_to_str(dtype):
    #     if isinstance(dtype, torch.dtype):
    #         return str(dtype)
    #     elif isinstance(dtype, str):
    #         return dtype
    #     else:
    #         return str(dtype)

    data_json = {
        "num_layer": num_layer,
        "num_kv_head": num_kv_head,
        "head_size": head_size,
        "cache_dtype": dtype_to_str(cache_dtype),
        "model_dtype": dtype_to_str(model_dtype),
        "use_mla": mla_enabled(model_config),
    }
    logger.info(f"data_json for vllm_config: {data_json}")

    return data_json

def init_nixl_maps(num_gpus: int, model_path: str):
    value = os.environ.get("NIXL_RECV_BUFFER_SIZE_GB")
    size_gb = float(value) 
    nixl_buffer_size = int(size_gb*1024*1024*1024)
    vllm_config_path = os.path.join(model_path, "vllm_config.json")
    while True:
        try:
            with open(vllm_config_path, "r") as f:
                data_json = json.load(f)
            print(f"Successfully loaded config from {vllm_config_path}, data: {data_json}")
            break
        except Exception as e:
            print(f"Failed to read {vllm_config_path}: {e}. Retrying in 5s...")
            time.sleep(5)
    

    use_mla = data_json['use_mla']
    cache_dtype = str_to_dtype(data_json['cache_dtype'])
    model_dtype = str_to_dtype(data_json['model_dtype'])
    head_size = data_json['head_size']
    num_kv_head = data_json['num_kv_head']
    num_layer = data_json['num_layer']

    for i in range(num_gpus):
        # buffer = torch.empty(
        #     nixl_buffer_size,
        #     dtype=torch.uint8,
        #     device="cpu",
        #     pin_memory=False,
        # )
        # print(f"before send to share memory")
        # buffer = buffer.share_memory_()
        # print(f"after send to share memory")
        # mem_type = "DRAM"
        # cpu_buffer = buffer.view(torch.uint8).flatten()
        
        # buffer_size = cpu_buffer.numel() * cpu_buffer.element_size()
        # buffer_ptr = cpu_buffer.data_ptr()


        # kv_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
        # chunk_size = 256
        # kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)
        # shape = torch.Size(kv_shape)

        # num_elements = shape.numel()
        # bytes_per_element = torch.tensor([], dtype=kv_dtype).element_size()
        # align_bytes = num_elements * bytes_per_element

        # buffer_size = (buffer_size // align_bytes) * align_bytes
        # cpu_buffer = cpu_buffer[:buffer_size]

        agent = nixl_agent(str(uuid.uuid4()), None)

        # # # Register the memory
        # memory_desc = [(buffer_ptr, buffer_size, i, "")]
        # # TODO(Jiayi): remove hardcode `mem_type`
        # reg_descs = agent.get_reg_descs(memory_desc, mem_type=mem_type)
        # agent.register_memory(reg_descs)

        # # # Create xfer handlers
        # page_size = align_bytes
        # xfer_desc = []
        # for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
        #     xfer_desc.append((base_addr, page_size, i))

        # xfer_descs = agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        # xfer_handler = agent.prep_xfer_dlist("", xfer_descs, mem_type=mem_type)

        # cpu_buffer_map[i]= cpu_buffer
        nixl_agent_map[i] = agent
        # reg_descs_map[i] = reg_descs
        # xfer_descs_map[i] = xfer_descs
        # xfer_handler_map[i] = xfer_handler


def get_abs_rank(tp_rank: int) -> int:
    """
    Get the absolute GPU device ID given a tensor-parallel (local) rank.

    Example:
        CUDA_VISIBLE_DEVICES="0,3"
        tp_rank=1  --> returns 3
        tp_rank=0  --> returns 0
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        # Fallback — if env not set, assume direct mapping
        return tp_rank

    # Split by commas and strip whitespace
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    if tp_rank < 0 or tp_rank >= len(devices):
        raise ValueError(
            f"tp_rank={tp_rank} out of range for CUDA_VISIBLE_DEVICES={visible}"
        )

    return int(devices[tp_rank])

def get_cpu_buffer(abs_rank: int):
    assert abs_rank in cpu_buffer_map, f"cpu_buffer_map doesn't have rank {abs_rank}"
    return cpu_buffer_map[abs_rank]

def get_nixl_agent(abs_rank: int):
    assert abs_rank in nixl_agent_map, f"nixl_agent_map doesn't have rank {abs_rank}'s nixl agent"
    return nixl_agent_map[abs_rank]