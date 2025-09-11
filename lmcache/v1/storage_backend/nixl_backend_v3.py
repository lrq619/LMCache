# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import List, Optional, Sequence
import threading

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    NixlCPUMemoryAllocator,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.connector.nixl_connector_v3 import (
    NixlChannel,
)
from lmcache.v1.storage_backend.connector.nixl_utils import NixlConfigXpYd, NixlRole

import time

logger = init_logger(__name__)


class NixlBackend(AllocatorBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the
    implementation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        memory_allocator: NixlCPUMemoryAllocator,
    ):
        """
        Initialize the Nixl storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.
        """
        super().__init__(dst_device=nixl_config.buffer_device)
        self.last_count_time = time.time()

        self.last_put_count = 0
        self.put_count = 0

        self.last_get_count = 0
        self.get_count = 0

        # NOTE(Jiayi): sender/prefiller will not use this pool;
        # only receiver/decoder will.
        self._data: dict[CacheEngineKey, MemoryObj] = {}

        self._data_lock = threading.Lock()

        assert nixl_config.role in [
            NixlRole.SENDER,
            NixlRole.RECEIVER,
        ], "Nixl role must be either SENDER or RECEIVER."

        self.memory_allocator = memory_allocator

        self._nixl_channel = NixlChannel(nixl_config, config, self)

    def memcheck(self):
        self.memory_allocator.memcheck()

    def get_allocated_size(self):
        return self.memory_allocator.nixl_allocator.total_allocated_size, self.memory_allocator.nixl_allocator.cpu_total_allocated_size

    def get_max_lifespan(self):
        max_lifespan = 0
        with self._data_lock:
            for key, mem_obj in self._data.items():
                if not mem_obj:
                    continue
                lifespan = time.time() - mem_obj.allocated_ts
                max_lifespan = max(max_lifespan, lifespan)
        return max_lifespan

    def get_olddest_req_id(self):
        olddest_req_id = ""
        max_lifespan = 0
        with self._data_lock:
            for key, mem_obj in self._data.items():
                if not mem_obj:
                    continue
                lifespan = time.time() - mem_obj.allocated_ts
                if max_lifespan < lifespan:
                    olddest_req_id = mem_obj.req_ids[0]
                max_lifespan = max(max_lifespan, lifespan)
                
        return olddest_req_id

    def get_num_mem_objs(self, req_id: str) -> int:
        num_mem_objs = 0
        with self._data_lock:
            for key, mem_obj in self._data.items():
                if not mem_obj:
                    continue
                if req_id in mem_obj.req_ids:
                    num_mem_objs += 1
        return num_mem_objs

    def get_data_key_length(self) -> int:
        with self._data_lock:
            return len(self._data)

    def get_num_valid_mem_obj(self) -> int:
        num_mem_objs = 0
        with self._data_lock:
            for key, mem_obj in self._data.items():
                if not mem_obj:
                    continue
                num_mem_objs += 1
        return num_mem_objs

    def stat(self):
        now = time.time()
        duration = now - self.last_count_time
        new_put = self.put_count - self.last_put_count
        new_get = self.get_count - self.last_get_count
        put_speed = new_put / duration
        get_speed = new_get / duration

        self.last_count_time = now
        self.last_put_count = self.put_count
        self.last_get_count = self.get_count
        return put_speed, get_speed


    # TODO(Jiayi): handle `pin` smantics
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.

        :param key: The key to check
        :param pin: Whether to pin the object in the backend.

        :return: True if the key exists, False otherwise
        """
        assert isinstance(key, CacheEngineKey)
        logger.info(f"search for key {key}")
        with self._data_lock:
            return key in self._data
        # with self._data_lock:
        #     if mem_obj := self._data.get(key, None):
        #         if pin:
        #             mem_obj.ref_count_up()
        #         return True
        #     return False

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
        assert isinstance(key, CacheEngineKey)
        with self._data_lock:
            self.put_count += 1
            if key in self._data:
                assert len(mem_obj.req_ids) == 1
                org_mem_obj = self._data[key]
                logger.warning(f"cache hit for (org) req: {org_mem_obj.req_ids} and {mem_obj.req_ids}, going to merge the req_ids into 1")
                org_mem_obj.req_ids += mem_obj.req_ids
                # count up the mem obj, the object gets released only after both request frees it
                org_mem_obj.ref_count_up()
                # free the original mem_obj
                mem_obj.ref_count_down()
            else:
                logger.info(f"put key: {key}")
                self._data[key] = mem_obj

    def allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
        req_id: str = "",
    ) -> Optional[MemoryObj]:
        """
        Allocate a zero-copy write object for the given shape and dtype.

        This will be seen as "adding a new payload" to the backend.
        """

        # NOTE: no eviction and busy_loop in PD
        mem_obj = self.memory_allocator.allocate(
            shape=shape, dtype=dtype, fmt=fmt, allocator_type="nixl"
        )
        if mem_obj:
            mem_obj.allocated_ts = time.time()
            mem_obj.req_ids = [req_id]

        return mem_obj
    
    def allocate_cpu(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = False,
        busy_loop: bool = True,
        req_id: str = "",
    ) -> MemoryObj:
        mem_obj = self.memory_allocator.allocate_cpu(shape=shape, dtype=dtype, fmt=fmt)
        if mem_obj:
            mem_obj.allocated_ts = time.time()
            mem_obj.req_ids = [req_id]

        return mem_obj

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ):
        return self.memory_allocator.batched_allocate(
            shape, dtype, batch_size, fmt, allocator_type="nixl"
        )

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        for mem_obj in memory_objs:
            mem_obj.ref_count_up()
        for key in keys:
            assert isinstance(key, CacheEngineKey)

        logger.info(f"send keys: {keys}")
        self._nixl_channel.prepare_send(
            keys=keys,  # type: ignore
            mem_objs=memory_objs,
            transfer_spec=transfer_spec,
        )

    def submit_prefetch_task(self, key: CacheEngineKey) -> bool:
        """
        An async function to get the MemoryObj from the storage backend.

        :param key: The key of the MemoryObj.

        :return: a future object. None if the key does not exist.
        """
        raise NotImplementedError

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param key: The key of the MemoryObj.

        :return: MemoryObj. None if the key does not exist.
        """

        assert isinstance(key, CacheEngineKey)
        with self._data_lock:
            # NOTE(Jiayi): we assume that the key must be in local data
            # because we are using a push-based transfer
            mem_obj = self._data.get(key, None)
            assert mem_obj is not None, f"Key {key} not found in local data."

            # NOTE(Jiayi): Currently, we remove the cache from local storage
            # buffer (on decode node) after it is retrieved.
            # Another option is to keep it in the local storage buffer and
            # enable eviction when a new alloc request comes in.
            # To so the second option, we need to ref_count_up or pin here
            # and not use pop above.
            # The second option can potentially make PD and KV reuse compatible.

            # NOTE(Jiayi): Another thing to be noted is that there could be memory
            # leak in decoder buffer when prefix caching is enabled.

            return mem_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        raise NotImplementedError

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """
        # with self._data_lock:
        #     if mem_obj := self._data.get(key, None):
        #         if mem_obj.get_ref_count() == 1:
        #             del self._data[key]
        #         return True
        #     return False
        with self._data_lock:
            return self._data.pop(key, None) is not None

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self._nixl_channel.close()

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True

    # TODO (Jiayi): put this in _init__.py later
    @staticmethod
    def CreateNixlBackend(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        memory_allocator: NixlCPUMemoryAllocator,
    ) -> "NixlBackend":
        """
        Create a Nixl backend with the given configuration.

        :param nixl_config: The Nixl configuration.
        :param dst_device: The device where the data is stored.

        :return: A NixlBackend instance.
        """
        # Create the Nixl config
        nixl_config = NixlConfigXpYd.from_cache_engine_config(config, metadata)
        # Create the Nixl backend
        backend = NixlBackend(nixl_config, config, memory_allocator)
        return backend
