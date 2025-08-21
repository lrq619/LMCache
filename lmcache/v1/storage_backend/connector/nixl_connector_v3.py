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
import os
from dataclasses import dataclass
from queue import Queue
from typing import Any, Optional, Union
import threading
import time
import uuid

# Third Party
from nixl._api import nixl_agent as NixlAgent
import msgspec
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    CacheEngineKey,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector.nixl_utils import NixlConfigXpYd, NixlRole
import socket

logger = init_logger(__name__)


class NixlMsgBase(msgspec.Struct, tag=True):
    """Base class for all nixl-related messages"""

    pass


class NixlAllocRequest(NixlMsgBase):
    """ """

    keys: list[str]  # len(keys) indicates num_chunks
    fmt: int
    shape: list[int]  # The shape of the memory objects
    dtype: str
    last_chunk_toks: int
    req_id: str
    is_cuda: bool = True  # Whether the memory objects are on CUDA
    delete: bool = False


class NixlAllocResponse(NixlMsgBase):
    """ """

    remote_indexes: list[int]


class NixlInitRequest(NixlMsgBase):
    sender_meta_bytes: bytes  # Metadata from the sender nixl agent
    sender_cpu_meta_bytes: bytes


class NixlMemRegRequest(NixlMsgBase):
    is_cuda: bool = True  # Whether the memory objects are on CUDA


class NixlInitResponse(NixlMsgBase):
    receiver_meta_bytes: bytes  # Metadata from the receiver nixl agent
    receiver_cpu_meta_bytes: bytes


class NixlMemRegResponse(NixlMsgBase):
    receiver_xfer_dlist_bytes: bytes  # Serialized transfer descriptors for the receiver


class NixlProxyNotif(NixlMsgBase):
    req_id: str  # The request UUID to notify the proxy


NixlMsg = Union[
    NixlAllocRequest,
    NixlAllocResponse,
    NixlProxyNotif,
    NixlInitRequest,
    NixlInitResponse,
    NixlMemRegRequest,
    NixlMemRegResponse,
]


@dataclass
class NixlReceiverInfo:
    receiver_id: str
    receiver_host: Optional[str] = None
    receiver_init_port: Optional[int] = None
    receiver_alloc_port: Optional[int] = None


# no need to be msgspec
@dataclass
class NixlSenderTask:
    req_id: str
    receiver_info: NixlReceiverInfo
    keys: list[CacheEngineKey]  # The keys to send
    mem_objs: list[MemoryObj]  # The memory objects to send

    def get_alloc_request(self) -> NixlAllocRequest:
        """
        Get the allocation request for this sender task.

        Let's say there are N memory objects in total.
        We have the following assumptions:
        - The first N-1 memory objects are full chunks, each with
        `full_chunk_size` tokens.
        - The last memory object can be a partial chunk, which has
        `last_chunk_toks` tokens.
        """

        fmt = self.mem_objs[0].meta.fmt
        shape = self.mem_objs[0].meta.shape
        dtype = TORCH_DTYPE_TO_STR_DTYPE[self.mem_objs[0].meta.dtype]
        token_dim = fmt.token_dim()
        last_chunk_toks = self.mem_objs[-1].meta.shape[token_dim]

        # TODO(Jiayi): Reomove this for loop
        keys = [key.to_string() for key in self.keys]

        return NixlAllocRequest(
            keys=keys,
            fmt=fmt.value,
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks,
            req_id=self.req_id
        )

    # TODO (Jiayi): reduce for loop
    def get_local_indexes(self) -> list[int]:
        """
        Get the page indexes of the memory objects.
        This is needed for nixl transfer.
        """
        return [mem_obj.meta.address for mem_obj in self.mem_objs]

    def free_mem_objs(self):
        for mem_obj in self.mem_objs:
            mem_obj.ref_count_down()

class NixlSender:
    """Handles sending data through a NixlPipe."""

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: StorageBackendInterface,
    ):
        assert nixl_config.role == NixlRole.SENDER, (
            "NixlSender should only be initialized with NixlRole.SENDER"
        )

        self.device = nixl_config.buffer_device

        self.nixl_config = nixl_config

        self.memory_allocator = backend.memory_allocator
        
        logger.info("[NIXL][Sender] memory_allocator instance: %s", type(self.memory_allocator))
        logger.info("[NIXL][Sender] memory_allocator content: %s", repr(self.memory_allocator))
        logger.info(
            "[NIXL][Sender] Initializing NixlAgentWrapper with buffer_ptr=0x%x, buffer_size=%.2f MB, page_size=%d bytes",
            self.memory_allocator.buffer_ptr,
            self.memory_allocator.buffer_size / (1024 * 1024),
            self.memory_allocator.align_bytes,
        )

        self._sender_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.buffer_size,
            page_size=self.memory_allocator.align_bytes,
            mem_type="cuda"
        )
        self._sender_cpu_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.cpu_buffer_ptr,
            buffer_size=self.memory_allocator.cpu_buffer_size,
            page_size=self.memory_allocator.align_bytes,
            mem_type="DRAM"
        )
        self._nixl_agent = self._sender_nixl_wrapper.agent
        self._nixl_cpu_agent = self._sender_cpu_nixl_wrapper.agent

        # Initialize the ZeroMQ context
        self._context = zmq.Context()

        self._mem_alloc_sockets: dict[str, zmq.Socket] = {}

        self.req_queue = Queue()

        self._remote_xfer_handlers_dict = {}
        self._remote_xfer_handlers_is_cuda_dict = {}

        # Start the seder thread
        self._running = True

        # self._sender_thread = threading.Thread(
        #     target=self._sender_loop, daemon=True
        # )
        # self._sender_thread.start()

        proxy_host = nixl_config.proxy_host
        proxy_port = nixl_config.proxy_port
        proxy_url = f"{proxy_host}:{proxy_port}"

        # each request id can only send notify/error msg once
        self.notified_req_set = set()
        self.notified_req_lock = threading.Lock()

        self._proxy_side_channel = self._context.socket(zmq.PUSH)
        self._proxy_side_channel.connect(get_zmq_path(proxy_url, protocol="tcp"))

    def _send_error_msg(self, req_id: str):
        with self.notified_req_lock:
            if req_id in self.notified_req_set:
            # already notified, no need to send anything
                return
            else:
                self.notified_req_set.add(req_id)
        error_req_id = req_id + "::error"
        notif_msg = NixlProxyNotif(req_id=error_req_id)
        notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
        self._proxy_side_channel.send(notif_msg_bytes)

    def _send_notify_msg(self, req_id: str):
        with self.notified_req_lock:
            if req_id in self.notified_req_set:
            # already notified, no need to send anything
                return
            else:
                self.notified_req_set.add(req_id)
        logger.info(f"Notified kv ready for req: {req_id}")
        notif_msg = NixlProxyNotif(req_id=req_id)
        notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
        self._proxy_side_channel.send(notif_msg_bytes)

    def prepare_send(
        self,
        keys: list[CacheEngineKey],
        mem_objs: list[MemoryObj],
        transfer_spec=None,
    ):
        """
        Put the sender task into the request queue.
        """

        sender_task = NixlSenderTask(
            req_id=transfer_spec.req_id,
            receiver_info=transfer_spec.receiver_info,
            keys=keys,
            mem_objs=mem_objs,
        )

        logger.debug(
            "Preparing to send %s objs with request ID: %s to receiver: %s",
            len(sender_task.keys),
            sender_task.req_id,
            sender_task.receiver_info,
        )

        # self.req_queue.put(sender_task)

        req_id = sender_task.req_id
        receiver_id = sender_task.receiver_info.receiver_id

        # NOTE (Jiayi): Currently, a sender needs to connect to
        # 3 side channels:
        # (1) _init_side_channel (ad-hoc-established and destroyed
        # after nixl connection is established),
        # (2) _alloc_side_channel (ad-hoc-established),
        # (3) _proxy_side_channel (pre-established).
        # NOTE (Jiayi): In addition, a sender also needs to
        # initialize nixl connection.

        # NOTE (Jiayi): `_init_all_comm` checks and initializes
        # _alloc_side_channel and nixl connection.
        receiver_info = sender_task.receiver_info
        if not self._check_init(receiver_info):
            self._init_all_comm(receiver_info)

        # use remote alloc
        alloc_request = sender_task.get_alloc_request()
        alloc_request.is_cuda = self._remote_xfer_handlers_is_cuda_dict.get(receiver_id, True)

        alloc_response = self._remote_allocate(receiver_id, alloc_request)

        # send kv
        local_indexes = sender_task.get_local_indexes()
        remote_indexes = alloc_response.remote_indexes
        if not remote_indexes:
            self._send_error_msg(req_id)

            sender_task.free_mem_objs()
            raise RuntimeError(
                f"Failed to allocate memory objects for request ID: {req_id}")
        self._blocking_send(req_id, receiver_id, local_indexes, remote_indexes)
        logger.info(f"transfer spec: {transfer_spec} for req: {req_id}")
        # Below logic ensures that: each req_id must send notify/error msg once and only once
        if transfer_spec.is_last_prefill:
            # If it's last prefill, send notify msg with req_id
            self._send_notify_msg(req_id)
        else:
            # a partial prefill request is finished, we need to set a 2s timer, if is_last_prefill prefill is not done within this 2s, we sent out error msg
            threading.Timer(2.0, self._send_error_msg, args=[req_id]).start()


        # free local memory
        sender_task.free_mem_objs()

    def _remote_allocate(
        self, receiver_id: str, alloc_request: NixlAllocRequest
    ) -> NixlAllocResponse:
        """Send the allocation request to the remote peer and get the response."""

        logger.debug(
            "Sent allocation request to receiver %s with %s objs needed",
            receiver_id,
            len(alloc_request.keys),  # Use the first key as the request ID
        )

        side_channel = self._mem_alloc_sockets[receiver_id]
        
        logger.debug("ZMQ REQ socket connected to: %s", side_channel.getsockopt_string(zmq.LAST_ENDPOINT))
        
        side_channel.send(msgspec.msgpack.encode(alloc_request))
        msg = side_channel.recv()
        alloc_response = msgspec.msgpack.decode(msg, type=NixlMsg)

        assert isinstance(alloc_response, NixlAllocResponse), (
            "The response from the remote peer is not a NixlAllocResponse"
        )

        logger.debug("Received allocation response.")

        return alloc_response

    @_lmcache_nvtx_annotate
    def _blocking_send(
        self,
        req_id: str,
        receiver_id: str,
        local_indexes: list[int],
        remote_indexes: list[int],
    ):
        """
        Send the KV cache in a blocking manner.
        """
        logger.debug(
            "Blocking send %s objs to receiver %s with request ID: %s",
            len(local_indexes),
            receiver_id,
            req_id,
        )

        is_cuda = self._remote_xfer_handlers_is_cuda_dict.get(receiver_id, True)
        agent = self._nixl_agent if is_cuda else self._nixl_cpu_agent
        agent_wrapper = self._sender_nixl_wrapper if is_cuda else self._sender_cpu_nixl_wrapper
        handle = agent.make_prepped_xfer(
            "WRITE",
            agent_wrapper.xfer_handler,
            local_indexes,
            self._remote_xfer_handlers_dict[receiver_id],
            remote_indexes,
            # notif_msg_bytes,
        )
        agent.transfer(handle)

        # TODO (Jiayi): offload the following to another thread
        # TODO (Jiayi) tune hyperparameters
        wait_time = 0.0007
        decay = 1.1
        while True:
            status = agent.check_xfer_state(handle)
            logger.debug(f"Transfer status: {status}")

            if status == "ERR":
                logger.error("Error in send operation")
                raise RuntimeError("Failed to send data to remote peer")
            elif status == "PROC":
                time.sleep(wait_time)  # Avoid busy waiting
                wait_time /= decay
                continue
            assert status == "DONE", f"Transfer status is {status}, expected DONE"
            # self._proxy_side_channel.send(notif_msg_bytes)
            break

    def _is_same_node(self, receiver_host: str) -> bool:
        host_name = socket.gethostname()
        logger.debug(f"host_name is {host_name}")
        return receiver_host == host_name

    def _initialize_nixl_sender_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_init_url: str,
    ) -> None:
        """
        Initialize the NIXL sender connection with the receiver.
        """

        # Exchange nixl metadata
        init_tmp_socket = self._context.socket(zmq.REQ)
        init_tmp_socket.connect(get_zmq_path(receiver_init_url, protocol="tcp"))

        nixl_init_req = NixlInitRequest(
            sender_meta_bytes=self._nixl_agent.get_agent_metadata(),
            sender_cpu_meta_bytes=self._nixl_cpu_agent.get_agent_metadata(),
        )

        init_tmp_socket.send(msgspec.msgpack.encode(nixl_init_req))

        nixl_init_resp_bytes = init_tmp_socket.recv()

        nixl_init_resp = msgspec.msgpack.decode(nixl_init_resp_bytes, type=NixlMsg)

        remote_meta_bytes = nixl_init_resp.receiver_meta_bytes
        remote_cpu_meta_bytes = nixl_init_resp.receiver_cpu_meta_bytes
        
        remote_agent_name = self._nixl_agent.add_remote_agent(remote_meta_bytes)
        remote_cpu_agent_name = self._nixl_cpu_agent.add_remote_agent(remote_cpu_meta_bytes)

        # Register memory
        is_same_node = self._is_same_node(receiver_host)
        logger.debug(f"is same node {is_same_node}")
        nixl_mem_reg_req = NixlMemRegRequest(is_cuda=is_same_node)
        init_tmp_socket.send(msgspec.msgpack.encode(nixl_mem_reg_req))
        nixl_mem_reg_resp_bytes = init_tmp_socket.recv()
        nixl_mem_reg_resp = msgspec.msgpack.decode(
            nixl_mem_reg_resp_bytes, type=NixlMsg
        )
        
        remote_xfer_dlist_bytes = nixl_mem_reg_resp.receiver_xfer_dlist_bytes
        if is_same_node:
            remote_xfer_dlist = self._nixl_agent.deserialize_descs(remote_xfer_dlist_bytes)
            remote_xfer_handlers = self._nixl_agent.prep_xfer_dlist(
                remote_agent_name, remote_xfer_dlist
            )
        else:
            remote_xfer_dlist = self._nixl_cpu_agent.deserialize_descs(remote_xfer_dlist_bytes)
            remote_xfer_handlers = self._nixl_cpu_agent.prep_xfer_dlist(
                remote_cpu_agent_name, remote_xfer_dlist
            )
        self._remote_xfer_handlers_dict[receiver_id] = remote_xfer_handlers
        self._remote_xfer_handlers_is_cuda_dict[receiver_id] = is_same_node

        init_tmp_socket.close()

    def _initialize_mem_alloc_side_channel(
        self, receiver_id: str, receiver_mem_alloc_url: str
    ) -> None:
        """
        Initialize zmq connection for memory allocation.
        """
        mem_alloc_socket = self._context.socket(zmq.REQ)

        mem_alloc_socket.connect(get_zmq_path(receiver_mem_alloc_url, protocol="tcp"))

        self._mem_alloc_sockets[receiver_id] = mem_alloc_socket

    def _check_init(self, receiver_info: NixlReceiverInfo):
        receiver_id = receiver_info.receiver_id
        return (
            receiver_id in self._remote_xfer_handlers_dict
            and receiver_id in self._mem_alloc_sockets
        )

    def _init_all_comm(
        self,
        receiver_info: NixlReceiverInfo,
    ):
        """
        Initialize all communication channels with the receiver.
        """
        logger.debug(
            "Initializing all communication channels with receiver %s",
            receiver_info,
        )

        receiver_id = receiver_info.receiver_id
        receiver_host = receiver_info.receiver_host
        receiver_init_port = receiver_info.receiver_init_port
        receiver_alloc_port = receiver_info.receiver_alloc_port

        receiver_init_url = f"{receiver_host}:{receiver_init_port}"
        receiver_mem_alloc_url = f"{receiver_host}:{receiver_alloc_port}"

        # Initialize the nixl sender connection
        self._initialize_nixl_sender_connection(receiver_id, receiver_host, receiver_init_url)

        # Initialize the memory allocation side channel
        self._initialize_mem_alloc_side_channel(receiver_id, receiver_mem_alloc_url)

    def close(self):
        """Close the sender resources."""
        # Wait for the receiver thread to finish with timeout
        # self._sender_thread.join(timeout=3.0)  # 3 second timeout

        # self._running = False
        # if self._sender_thread.is_alive():
        #     logger.warning(
        #         "Sender thread did not shut down cleanly within timeout"
        #     )

        for s in self._mem_alloc_sockets.values():
            s.close()
        self._context.term()

        self._sender_nixl_wrapper.close(self._remote_xfer_handlers_dict)


class NixlReceiver:
    """Handles receiving data through a NixlPipe."""

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: StorageBackendInterface,
    ):
        assert nixl_config.role == NixlRole.RECEIVER, (
            "NixlReceiver should only be initialized with NixlRole.RECEIVER"
        )

        self._backend = backend
        self.memory_allocator = backend.memory_allocator

        self.device = nixl_config.buffer_device
        self._receiver_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.buffer_size,
            page_size=self.memory_allocator.align_bytes,
        )
        self._receiver_cpu_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.cpu_buffer_ptr,
            buffer_size=self.memory_allocator.cpu_buffer_size,
            page_size=self.memory_allocator.align_bytes,
        )

        self._nixl_agent = self._receiver_nixl_wrapper.agent
        self._nixl_cpu_agent = self._receiver_cpu_nixl_wrapper.agent

        self.nixl_config = nixl_config

        receiver_host = nixl_config.peer_host
        receiver_init_port = nixl_config.peer_init_port
        receiver_alloc_port = nixl_config.peer_alloc_port
        receiver_delete_port = os.environ["NIXL_DELETE_PORT"]
        logger.info(f"receiver_delete_port: {receiver_delete_port}")

        receiver_init_url = f"{receiver_host}:{receiver_init_port}"
        receiver_alloc_url = f"{receiver_host}:{receiver_alloc_port}"
        receiver_delete_url = f"{receiver_host}:{receiver_delete_port}"

        self.full_chunk_size = config.chunk_size

        # TODO (Jiayi)" make it async?"
        # Initialize the ZeroMQ context and side channel
        self._context = zmq.Context()  # type: ignore

        self._side_channels = []

        # TODO (Jiayi): have a util func to do this
        # Create/listen initialization side channel
        self._init_side_channel = self._context.socket(zmq.REP)
        self._init_side_channel.bind(get_zmq_path(receiver_init_url, protocol="tcp"))
        self._side_channels.append(self._init_side_channel)

        # Create/listen allocation side channel
        self._alloc_side_channel = self._context.socket(zmq.REP)
        self._alloc_side_channel.bind(get_zmq_path(receiver_alloc_url, protocol="tcp"))
        self._side_channels.append(self._alloc_side_channel)

        # Delete side channel
        self._delete_side_channel = self._context.socket(zmq.REP)
        self._delete_side_channel.bind(get_zmq_path(receiver_delete_url, protocol="tcp"))
        self._side_channels.append(self._delete_side_channel)
        

        # TODO: might be better to put them into one thread
        # and use asyncio to manage.
        # Start the receiver threads
        self._running = True
        self._running_threads = []

        self._mem_alloc_thread = threading.Thread(
            target=self._mem_alloc_loop, daemon=True
        )
        self._mem_alloc_thread.start()
        self._running_threads.append(self._mem_alloc_thread)

        self._init_thread = threading.Thread(target=self._init_loop, daemon=True)
        self._init_thread.start()
        self._running_threads.append(self._init_thread)

        self._stat_thread = threading.Thread(target=self._stat_loop, daemon=True)
        self._stat_thread.start()
        self._running_threads.append(self._stat_thread)

        self._delete_thread = threading.Thread(target=self._mem_delete_loop, daemon=True)
        self._delete_thread.start()
        self._running_threads.append(self._delete_thread)

        self.deleted_reqs_set = set()
        self.deleted_reqs_lock = threading.Lock()

    def _stat_loop(self):
        while self._running:
            try:
                total_allocated_size = self._backend.get_allocated_size()
                max_lifespan = self._backend.get_max_lifespan()
                oldest_req_id = self._backend.get_olddest_req_id()
                put_speed, get_speed = self._backend.stat()
                key_length = self._backend.get_data_key_length()
                valid_obj_num = self._backend.get_num_valid_mem_obj()
                logger.info(
                    f"[Receiver] Total allocated size: {total_allocated_size / (1024*1024):.2f} MB, "
                    f"max lifespan: {max_lifespan * 1000:.1f}ms, "
                    f"req_id with max lifespan: {oldest_req_id}, "
                    f"put_speed: {put_speed:.1f} obj/s, get_speed: {get_speed:.1f} obj/s, "
                    f"key_length: {key_length}, valid_obj_num: {valid_obj_num}"
                )
            except Exception as e:
                logger.exception(f"[Receiver] Exception in stat loop: {e}")
            time.sleep(5)

    def _mem_delete_loop(self):
        torch.cuda.set_device(self.device)
        # TODO: `self._running` might not be safe here
        logger.info(f"Start mem delete loop")
        while self._running:
            try:
                # NOTE: this is a req-reply zmq for now
                # receive alloc request
                delete_req_bytes = self._delete_side_channel.recv()
                delete_req = msgspec.msgpack.decode(delete_req_bytes, type=NixlMsg)
                assert isinstance(delete_req, NixlAllocRequest), (
                    "The request from the remote peer is not a NixlAllocRequest"
                )
                
                assert delete_req.delete, "mem_delete_loop can only receive delete request"
                deleted = self._backend.delete_by_req_id(delete_req.req_id)
                # append deleted req_id to the set
                with self.deleted_reqs_lock:
                    self.deleted_reqs_set.add(delete_req.req_id)
                self._delete_side_channel.send(msgspec.msgpack.encode(
                    NixlAllocResponse(remote_indexes=[])))
                logger.info(
                    "Received delete request for %s, deleted: %s",
                    delete_req.req_id,
                    deleted,
                )   
            except zmq.Again as e:  # type: ignore
                # Handle the timeout when waiting for a message
                logger.debug(
                    "Timeout waiting for a message on the side channel: %s",
                    str(e),
                )
                continue
            except Exception as e:
                logger.error("Failed to process mem alloc loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)

    def _allocate_and_put(self, alloc_request: NixlAllocRequest) -> NixlAllocResponse:
        total_allocs = len(alloc_request.keys)
        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = alloc_request.shape
        alloc_indexes = []

        # max_lifespan_sec = 2
        # self._backend.garbage_collection(max_lifespan_sec)
        keys = []
        mem_objs = []
        for idx, key in enumerate(alloc_request.keys):
            if idx == total_allocs - 1:
                num_alloc_tokens = alloc_request.last_chunk_toks
                token_dim = fmt.token_dim()
                shape[token_dim] = num_alloc_tokens
            else:
                num_alloc_tokens = self.full_chunk_size

            mem_obj = None
            wait_time = 0.2
            decay = 1.5

            while mem_obj is None:
                req_id = alloc_request.req_id
                # check if delete signal has been issued, if so, break the loop
                with self.deleted_reqs_lock:
                    if req_id in self.deleted_reqs_set:
                        logger.info(f"req: {req_id} already deleted by delete thread, stop trying to allocate")
                        break

                if alloc_request.is_cuda:
                    mem_obj = self._backend.allocate(torch.Size(shape), dtype, fmt, req_id=req_id)
                else:
                    mem_obj = self._backend.allocate_cpu(torch.Size(shape), dtype, fmt, req_id=req_id)

                if mem_obj is None:
                    logger.warning(f"Failed to allocate memory object for req: {req_id}, retrying...")
                    time.sleep(wait_time)
                    wait_time *= decay
                
            if mem_obj is None:
                for mem_obj in mem_objs:
                    mem_obj.ref_count_down()
                logger.warning(
                    f"Failed to allocate memory object for req: {req_id}, "
                    "returning empty response.")
                return NixlAllocResponse(remote_indexes=[])

            keys.append(key)
            mem_objs.append(mem_obj)

        for i, key in enumerate(keys):
            mem_obj = mem_objs[i]
            alloc_indexes.append(mem_obj.meta.address)

            self._backend.put(CacheEngineKey.from_string(key), mem_obj)
        if len(keys) > 0: 
            logger.info(f"put {len(keys)} mem_objs for req: {mem_objs[0].req_ids}")

        return NixlAllocResponse(remote_indexes=alloc_indexes)


    # TODO: have a loop wrapper to wrap different loops
    def _mem_alloc_loop(self):
        """ """
        torch.cuda.set_device(self.device)
        # TODO: `self._running` might not be safe here
        while self._running:
            try:
                # NOTE: this is a req-reply zmq for now
                # receive alloc request
                alloc_req_bytes = self._alloc_side_channel.recv()
                alloc_req = msgspec.msgpack.decode(alloc_req_bytes, type=NixlMsg)
                assert isinstance(alloc_req, NixlAllocRequest), (
                    "The request from the remote peer is not a NixlAllocRequest"
                )
                logger.debug(
                    "Received allocation request %s for %s objs",
                    alloc_req.req_id, len(alloc_req.keys),
                )

                # NOTE: it's okay to put the memory objs into the storage backend
                # first because decode vllm will not be able to see the decode
                # request until proxy receives the ack.
                alloc_resp = self._allocate_and_put(alloc_req)
                

                logger.debug(
                    "Replying allocation response for %s objs",
                    len(alloc_resp.remote_indexes),
                )

                # send back response
                self._alloc_side_channel.send(msgspec.msgpack.encode(alloc_resp))

            except zmq.Again as e:  # type: ignore
                # Handle the timeout when waiting for a message
                logger.debug(
                    "Timeout waiting for a message on the side channel: %s",
                    str(e),
                )
                continue
            except Exception as e:
                logger.error("Failed to process mem alloc loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)

    def _init_loop(self):
        local_meta = self._nixl_agent.get_agent_metadata()
        local_cpu_meta = self._nixl_cpu_agent.get_agent_metadata()

        # NOTE: Initialization has to be two stages:
        # (1) Exchanging the metadata.
        # (2) Registering the memory descriptors.
        # Otherwise, there's a chance that nixl got stuck
        # (handle always give "PROC" status) during the first request.
        while self._running:
            try:
                req_bytes = self._init_side_channel.recv()

                logger.debug("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=NixlMsg)

                if isinstance(req, NixlInitRequest):
                    self._nixl_agent.add_remote_agent(req.sender_meta_bytes)
                    self._nixl_cpu_agent.add_remote_agent(req.sender_cpu_meta_bytes)

                    resp = NixlInitResponse(
                        receiver_meta_bytes=local_meta,
                        receiver_cpu_meta_bytes=local_cpu_meta,
                    )

                    logger.debug("Replying initialization response")

                elif isinstance(req, NixlMemRegRequest):
                    is_cuda = req.is_cuda
                    if is_cuda:
                        # Register the memory descriptors for CUDA
                        local_xfer_descs = self._nixl_agent.get_serialized_descs(
                            self._receiver_nixl_wrapper.xfer_descs
                        )
                    else:
                        # Register the memory descriptors for CPU
                        local_xfer_descs = self._nixl_cpu_agent.get_serialized_descs(
                            self._receiver_cpu_nixl_wrapper.xfer_descs
                        )

                    resp = NixlMemRegResponse(
                        receiver_xfer_dlist_bytes=local_xfer_descs,
                    )

                    logger.debug("Replying mem register response")

                self._init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)

    def close(self):
        """Close the receiver resources."""
        self._running = False

        for t in self._running_threads:
            # Wait for the receiver thread to finish with timeout
            t.join(timeout=3.0)  # 3 second timeout

            if t.is_alive():
                logger.warning(
                    "Receiver thread did not shut down cleanly within timeout"
                )
        for side_channel in self._side_channels:
            side_channel.close()
        self._context.term()

        self._receiver_nixl_wrapper.close()


class NixlChannel:
    """Provides the primitives to send the data and process the received data.
    It will have some internal threads to handle the data receiving.
    """

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: StorageBackendInterface,
    ):
        self.nixl_config = nixl_config
        self.role = nixl_config.role

        # Create sender or receiver based on role
        self._sender = None
        self._receiver = None

        self._backend = backend

        if nixl_config.role == NixlRole.SENDER:
            self._sender = NixlSender(nixl_config, config, backend)
        else:
            self._receiver = NixlReceiver(nixl_config, config, backend)

    def _check_sender(self):
        """Check if this channel is configured as a sender."""
        if self._sender is None:
            raise RuntimeError(f"Cannot perform sender operation with role {self.role}")
        return self._sender

    def _check_receiver(self):
        """Check if this channel is configured as a receiver."""
        if self._receiver is None:
            raise RuntimeError(
                f"Cannot perform receiver operation with role {self.role}"
            )
        return self._receiver

    def prepare_send(
        self,
        keys: list[CacheEngineKey],
        mem_objs: list[MemoryObj],
        transfer_spec=None,
    ):
        """Prepare a send transaction by sending the request using
        the side channel.
        """
        sender = self._check_sender()
        sender.prepare_send(keys, mem_objs, transfer_spec)

    def close(self):
        """Close all resources."""
        if self._sender:
            self._sender.close()
        if self._receiver:
            self._receiver.close()


############################################################
# helper functions
############################################################


# TODO (Jiayi): support multiple protocols
def get_zmq_path(url: str, protocol: str = "tcp") -> str:
    """Get the ZeroMQ path for the given base path and suffix."""
    if protocol == "tcp":
        return f"tcp://{url}"
    raise ValueError(f"Unsupported protocol: {protocol}")


@dataclass
class NixlAgentWrapper:
    agent: NixlAgent
    reg_descs: Any
    xfer_descs: Any
    xfer_handler: Any

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        *,
        mem_type: str = "cuda",
    ):
        """
        Initialize the NIXL agent.

        Args:
            buffer_size (int): The size of the buffer.
            buffer_ptr (int): The pointer to the buffer.
            page_size (int): The page size of NIXL and
                the lmcache memory allocator.

        Returns:
            NixlWrapper: The NIXL agent.
            reg_dlist: the registered memory descriptor list.
            xfer_dlist: the local transfer descriptor list.
            prepped_xfer_handler: the prepped transfer handler.
        """
        if NixlAgent is None:
            raise RuntimeError("NIXL is not available")

        # Create a NIXL agent
        nixl_agent = NixlAgent(str(uuid.uuid4()))

        # Register the memory
        memory_desc = [(buffer_ptr, buffer_size, 0, "")]
        # TODO(Jiayi): remove hardcode `mem_type`
        reg_descs = nixl_agent.get_reg_descs(memory_desc, mem_type=mem_type)
        nixl_agent.register_memory(reg_descs)

        # Create xfer handlers
        xfer_desc = []
        for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
            xfer_desc.append((base_addr, page_size, 0))

        xfer_descs = nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = nixl_agent.prep_xfer_dlist("", xfer_descs, mem_type=mem_type)

        self.agent = nixl_agent
        self.reg_descs = reg_descs
        self.xfer_descs = xfer_descs
        self.xfer_handler = xfer_handler

    def close(self, remote_xfer_handlers: Optional[dict[str, Any]] = None):
        self.agent.deregister_memory(self.reg_descs)

        self.agent.release_dlist_handle(self.xfer_handler)

        for remote_xfer_handler in self._remote_xfer_handlers.values():
            self.agent.release_dlist_handle(remote_xfer_handler)

        if remote_xfer_handlers is not None:
            for remote_xfer_handler in remote_xfer_handlers.values():
                self.agent.release_dlist_handle(remote_xfer_handler)
