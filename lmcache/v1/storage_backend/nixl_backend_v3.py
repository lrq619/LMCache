# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from concurrent.futures import Future
from typing import List, Optional
from queue import Queue, Empty
import threading
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector.nixl_connector_v2 import (
    NixlChannel,
    NixlObserverInterface,
)
from lmcache.v1.storage_backend.connector.nixl_utils import NixlConfig, NixlRole

logger = init_logger(__name__)


class NixlBackend(StorageBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the
    implementation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(
        self, 
        nixl_config: NixlConfig,
        config: LMCacheEngineConfig,
    ):
        """
        Initialize the Nixl storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.
        """
        super().__init__(dst_device=nixl_config.buffer_device)
        
        # NOTE(Jiayi): sender/prefiller will not use this pool;
        # only receiver/decoder will.
        self._data: dict[CacheEngineKey, MemoryObj] = {}
        
        # FIXME(Jiayi): do we need this lock?
        # self._data_lock = threading.Lock()

        self._nixl_channel = NixlChannel(
            nixl_config, config, self)
        
        assert nixl_config.role in [
            NixlRole.SENDER,
            NixlRole.RECEIVER,
        ], "Nixl role must be either SENDER or RECEIVER."

    # TODO(Jiayi): handle `pin` smantics
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.

        :param key: The key to check
        :param pin: Whether to pin the object in the backend.

        :return: True if the key exists, False otherwise
        """
        return self._obj_pool.contains(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.

        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        return False

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ):
        self._data[key] = mem_obj
        
    def register_put_tasks(
        self,
        keys: list[CacheEngineKey],
        mem_objs: list[MemoryObj],
    ) -> None:
        """
        Register the put tasks to the backend.
        """
        self._nixl_channel.prepare_send(keys=keys, mem_objs=mem_objs)

    def allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = False,
    ) -> MemoryObj:
        """
        Allocate a zero-copy write object for the given shape and dtype.

        This will be seen as "adding a new payload" to the backend.
        """

        mem_obj = self._nixl_channel.local_allocate(shape=shape, dtype=dtype, fmt=fmt)
        
        # NOTE: The following will never happen since `local_allocate`
        # will always wait for a valid MemoryObj.
        assert mem_obj is not None, "Failed to allocate zero-copy buffer from nixl_channel"
        
        return mem_obj

    def batched_submit_put_task(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> Optional[List[Future]]:
        self.register_put_tasks(keys, memory_objs)
        return None

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        """
        An async function to get the MemoryObj from the storage backend.

        :param key: The key of the MemoryObj.

        :return: a future object. None if the key does not exist.
        """
        raise NotImplementedError

    # FIXME
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param key: The key of the MemoryObj.

        :return: MemoryObj. None if the key does not exist.
        """
        
        # NOTE(Jiayi): we assume that the key must be in local data
        mem_obj = self._data.get(key, None)
        assert mem_obj is not None, f"Key {key} not found in local data."
        
        return mem_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        raise NotImplementedError

    def remove(self, key: CacheEngineKey) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """
        return self._obj_pool.remove(key)

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self._nixl_channel.close()


    def pin(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError

    def unpin(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError

    # FIXME: better dropping this
    @staticmethod
    def CreateNixlBackend(
        config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> "NixlBackend":
        """
        Create a Nixl backend with the given configuration.

        :param nixl_config: The Nixl configuration.
        :param dst_device: The device where the data is stored.

        :return: A NixlBackend instance.
        """
        # Create the Nixl config
        nixl_config = NixlConfig.from_cache_engine_config(config, metadata)
        # Create the Nixl backend
        backend = NixlBackend(nixl_config)
        return backend
