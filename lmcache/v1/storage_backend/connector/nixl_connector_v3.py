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
from dataclasses import dataclass
from typing import Callable, Optional, Union, Any
from queue import Queue, Empty
import abc
import threading
import time
import uuid

# Third Party
from nixl._api import nixl_agent as NixlAgent
import msgpack
import msgspec
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey, 
    STR_DTYPE_TO_TORCH_DTYPE, 
    TORCH_DTYPE_TO_STR_DTYPE,
    _lmcache_nvtx_annotate
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.connector.nixl_utils import (
    NixlConfigXpYd,
    NixlRole
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.config import LMCacheEngineConfig


logger = init_logger(__name__)


class NixlMsgBase(msgspec.Struct, tag=True):
    """Base class for all nixl-related messages"""
    pass
    
class NixlAllocRequest(NixlMsgBase):
    """
    """
    keys: list[str] # len(keys) indicates num_chunks
    fmt: int
    shape: list[int]  # The shape of the memory objects
    dtype: str
    last_chunk_toks: int

class NixlAllocResponse(NixlMsgBase):
    """
    """
    remote_indexes: list[int]


class NixlInitRequest(NixlMsgBase):
    sender_meta_bytes: bytes  # Metadata from the sender nixl agent

class NixlInitResponse(NixlMsgBase):
    receiver_meta_bytes: bytes  # Metadata from the receiver nixl agent
    receiver_xfer_dlist_bytes: bytes  # Serialized transfer descriptors for the receiver

class NixlProxyNotif(NixlMsgBase):
    """
    """
    req_uuid: str  # The request UUID to notify the proxy

class NixlXferNotif(NixlMsgBase):
    """
    """
    req_uuid: str  # The request UUID to notify the receiver

NixlMsg = Union[
    NixlAllocRequest,
    NixlAllocResponse,
    NixlProxyNotif,
    NixlXferNotif,
]

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
        
        fmt = self.mem_objs[0].fmt
        shape = self.mem_objs[0].meta.shape
        dtype = TORCH_DTYPE_TO_STR_DTYPE(
            self.mem_objs[0].meta.dtype)
        token_dim = fmt.token_dim()
        last_chunk_toks = self.mem_objs[-1].meta.shape[token_dim]
        
        return NixlAllocRequest(
            keys=self.keys, 
            fmt=int(fmt),
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks)
    
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
            "NixlSender should only be initialized"
            " with NixlRole.SENDER")
        
        self.nixl_config = nixl_config
        
        self.memory_allocator = backend.memory_allocator
        
        self._sender_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.buffer_size,
            page_size=self.memory_allocator.align_bytes,
        )
        self._nixl_agent = self._sender_nixl_wrapper.agent

        # Initialize the ZeroMQ context
        self._context = zmq.Context()
        
        self._mem_alloc_sockets: dict[str, zmq.Socket] = {}
        
        self.req_queue = Queue()
        
        self._remote_xfer_handlers_dict = {}

    def prepare_send(
        self, 
        keys: list[CacheEngineKey], 
        mem_objs: list[MemoryObj],
        transfer_spec = None,
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
        
        self.req_queue.put(sender_task)

    
    def _remote_allocate(
        self, 
        receiver_id: str,
        alloc_request: NixlAllocRequest
    ) -> NixlAllocResponse:
        """Send the allocation request to the remote peer and get the response."""
        
        side_channel = self._mem_alloc_sockets[receiver_id]
        
        side_channel.send(msgspec.msgpack.encode(alloc_request))
        msg = side_channel.recv()
        alloc_response = msgspec.msgpack.decode(msg, type=NixlMsg)
        
        assert isinstance(alloc_response, NixlAllocResponse), (
            "The response from the remote peer is not a NixlAllocResponse"
        )
        
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
        """
        notif_msg = NixlXferNotif(req_id)
        notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
        
        handle = self._nixl_agent.make_prepped_xfer(
            "WRITE",
            self._nixl_sender_wrapper.xfer_handler,
            local_indexes,
            self._remote_xfer_handlers_dict[receiver_id],
            remote_indexes,
            notif_msg_bytes,
        )
        
        self._agent.transfer(handle)
        
        while True:
            status = self._nixl_agent.check_xfer_state(handle)
            if status == "ERR":
                logger.error("Error in send operation")
                raise RuntimeError(
                    "Failed to send data to remote peer"
                )
            elif status == "PROC":
                time.sleep(0.001)  # Avoid busy waiting
            assert status == "DONE", (
                f"Transfer status is {status}, expected DONE"
            )
            break
        
    
    def _initialize_nixl_sender_connection(
        self, 
        receiver_id: str,
        receiver_init_url: NixlReceiverInfo,
    ) -> None:
        """
        
        """
                
        init_tmp_socket = self._context.socket(zmq.REQ)
        init_tmp_socket.connect(get_zmq_path(
            receiver_init_url, protocol="tcp"
        ))
        
        nixl_init_req = NixlInitRequest(
            sender_meta_bytes=self._nixl_agent.get_agent_metadata(),
        )

        
        init_tmp_socket.send(msgspec.msgpack.encode(nixl_init_req))
        
        nixl_init_resp_bytes = init_tmp_socket.recv()
        
        init_tmp_socket.close()
        
        nixl_init_resp = msgspec.msgpack.decode(
            nixl_init_resp_bytes, type=NixlMsg)
        
        remote_meta_bytes = nixl_init_resp.receiver_meta_bytes
        remote_agent_name = self._nixl_agent.add_remote_agent(
                remote_meta_bytes
            )
        remote_xfer_dlist_bytes = nixl_init_resp.receiver_xfer_dlist_bytes
        remote_xfer_dlist = self._nixl_agent.deserialize_descs(
                remote_xfer_dlist_bytes
            )
        
        remote_xfer_handlers = self._nixl_agent.prep_xfer_dlist(
                remote_agent_name, remote_xfer_dlist)
        
        self._remote_xfer_handlers_dict[
                receiver_id] = remote_xfer_handlers
    
    def _initialize_mem_alloc_side_channel(
        self, 
        receiver_id: str,
        receiver_mem_alloc_url: str
    ) -> None:
        
        mem_alloc_socket = self._context.socket(zmq.REQ)
        
        mem_alloc_socket.connect(get_zmq_path(
            receiver_mem_alloc_url, protocol="tcp"
        ))
        
        self._mem_alloc_sockets[receiver_id] = mem_alloc_socket
    
    def _check_init(
        self,
        receiver_info: NixlReceiverInfo
    ):
        receiver_id = receiver_info.receiver_id 
        return receiver_id in self._remote_xfer_handlers_dict and \
            receiver_id in self._mem_alloc_sockets 

    def _init_all_comm(
        self,
        receiver_info: NixlReceiverInfo,
    ):
        """Initialize all communication channels with the receiver."""
        receiver_id = receiver_info.receiver_id
        receiver_host = receiver_info.receiver_host
        receiver_port = receiver_info.receiver_port
        
        receiver_init_url = f"{receiver_host}:{receiver_port}"
        receiver_mem_alloc_url = f"{receiver_host}:{receiver_port + 1}"
        
        # Initialize the nixl sender connection
        self._initialize_nixl_sender_connection(
            receiver_id, receiver_init_url)
        
        # Initialize the memory allocation side channel
        self._initialize_mem_alloc_side_channel(
            receiver_id, receiver_mem_alloc_url)
        
    def _sender_loop(self):
        

        while self._running:
            try:
                
                sender_task = self.req_queue.get()
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
                
                alloc_response = self._remote_allocate(
                    receiver_id, alloc_request)
                
                # send kv
                local_indexes = sender_task.get_local_indexes()
                remote_indexes = alloc_response.remote_indexes
                self._blocking_send(
                    req_id, receiver_id,
                    local_indexes, remote_indexes)
                                
                # free local memory
                sender_task.free_mem_objs()
                
            except Exception as e:
                logger.error("Failed to process receiver loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)    

    def close(self):
        """Close the sender resources."""
        for s in self._mem_alloc_sockets.values():
            s.close()
        self._context.term()
        
        self._sender_nixl_wrapper.close(
            self._remote_xfer_handlers_dict)

class NixlReceiver:
    """Handles receiving data through a NixlPipe."""

    def __init__(
        self, 
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: StorageBackendInterface,
    ):
        assert nixl_config.role == NixlRole.RECEIVER, (
            "NixlReceiver should only be initialized"
            " with NixlRole.RECEIVER")
        
        self._backend = backend
        self.memory_allocator = backend.memory_allocator
        
        self._receiver_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.buffer_size,
            page_size=self.memory_allocator.align_bytes,
        )
        
        self._nixl_agent = self._receiver_nixl_wrapper.agent
        
        self.nixl_config = nixl_config
        
        receiver_host = nixl_config.peer_host
        receiver_init_port = nixl_config.peer_init_port
        receiver_alloc_port = nixl_config.peer_alloc_port
        
        receiver_init_url = f"{receiver_host}:{receiver_init_port}"
        receiver_alloc_url = f"{receiver_host}:{receiver_alloc_port}"
        
        proxy_host = nixl_config.proxy_host
        proxy_port = nixl_config.proxy_port
        proxy_url = f"{proxy_host}:{proxy_port}"
        
        self.full_chunk_size = config.chunk_size
        
        
        # TODO (Jiayi)" make it async?"
        # Initialize the ZeroMQ context and side channel
        self._context = zmq.Context()  # type: ignore
        
        self._side_channels = []
        
        # TODO (Jiayi): have a util func to do this
        # Create/listen initialization side channel
        self._init_side_channel = self._context.socket(zmq.REP)
        self._init_side_channel.bind(get_zmq_path(
            receiver_init_url, protocol="tcp"
        ))
        self._side_channels.append(self._init_side_channel)
        
        # Create/listen allocation side channel
        self._alloc_side_channel = self._context.socket(zmq.REP)
        self._alloc_side_channel.bind(get_zmq_path(
            receiver_alloc_url, protocol="tcp"
        ))
        self._side_channels.append(self._alloc_side_channel)
        
        # Connect to proxy side channel
        self._proxy_side_channel = self._context.socket(zmq.PUSH)
        self._proxy_side_channel.connect(get_zmq_path(
            proxy_url, protocol="tcp"
        ))
        self._side_channels.append(self._proxy_side_channel)
       
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
        
        self._init_thread = threading.Thread(
            target=self._init_loop, daemon=True
        )
        self._init_thread.start()
        self._running_threads.append(self._init_thread)
        
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True
        )
        self._recv_thread.start()
        self._running_threads.append(self._recv_thread)
        
    
    def _allocate_and_put(
        self, 
        alloc_request: NixlAllocRequest
    ) -> NixlAllocResponse:
        
        total_allocs = len(alloc_request.keys)
        fmt = MemoryFormat(alloc_request.fmt)
        dtype = torch.dtype(alloc_request.dtype)
        shape = alloc_request.shape
        alloc_indexes = []
        
        for idx, key in enumerate(alloc_request.keys):
            if idx == total_allocs - 1:
                num_alloc_tokens = alloc_request.last_chunk_toks
                token_dim = fmt.token_dim()
                shape[token_dim] = num_alloc_tokens
            else:
                num_alloc_tokens = self.full_chunk_size
            
            mem_obj = self._backend.local_allocate(
                torch.Size(shape), 
                STR_DTYPE_TO_TORCH_DTYPE(dtype),
                fmt
            )
            
            alloc_indexes.append(mem_obj.meta.address)
            
            self._backend.put(CacheEngineKey.from_string(key), mem_obj)
        
        return NixlAllocResponse(
            remote_indexes=alloc_indexes
        )

    # TODO: have a loop wrapper to wrap different loops
    def _mem_alloc_loop(self):
        """
        """
        # TODO: `self._running` might not be safe here
        while self._running:
            try:
                # NOTE: this is a req-reply zmq for now
                # recieve alloc request
                alloc_req_bytes = self._alloc_side_channel.recv()
                alloc_req = msgspec.msgpack.decode(
                    alloc_req_bytes, type=NixlMsg
                )
                assert isinstance(alloc_req, NixlAllocRequest), (
                    "The request from the remote peer is not a NixlAllocRequest")
                # NOTE: it's okay to put the memory objs into the storage backend
                # first because decode vllm will not be able to see the decode
                # request until proxy receives the ack.
                alloc_resp = self._allocate_and_put(alloc_req)
                
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
        while self.running:
            try:
                init_req_bytes = self._init_side_channel.recv()
                init_req = msgspec.msgpack.decode(
                    init_req_bytes, type=NixlMsg
                )
                assert isinstance(init_req, NixlInitRequest), (
                    "The request from the remote peer is not a NixlInitRequest")

                self._receiver_nixl_agent.agent.add_remote_agent(
                    init_req.sender_meta_bytes)
                
                local_meta = self._receiver_nixl_agent.agent.get_agent_metadata()
                
                local_xfer_descs = self._nixl_agent.get_serialized_descs(
                                self._nixl_receiver_wrapper.xfer_descs
                            )
                
                init_resp = NixlInitResponse(
                    receiver_meta_bytes=local_meta,
                    receiver_xfer_dlist_bytes=local_xfer_descs,
                )
                
                self._init_side_channel.send(
                    msgspec.msgpack.encode(init_resp)
                )
                
            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)
    
    
    def _recv_loop(self):
        
        while self._running:
            try:
                notifs = self._nixl_agent.get_new_notifs()
                for remote_agent_name in notifs:
                    for msg_bytes in notifs[remote_agent_name]:
                        msg = msgspec.msgpack.decode(
                            msg_bytes, type=NixlMsg
                        )
                        if isinstance(msg, NixlXferNotif):
                            # send ack to proxy
                            proxy_notif = NixlProxyNotif(req_uuid=msg.req_uuid)
                            self._proxy_side_channel.send(
                                msgspec.msgpack.encode(proxy_notif))
                            
                        else:
                            raise RuntimeError(
                                f"Received unexpected message type: {type(msg)}"
                                f" from remote agent: {remote_agent_name}"
                            )
                            
            except zmq.Again as e:  # type: ignore
                # Handle the timeout when waiting for a message
                logger.debug(
                    "Timeout waiting for a message on the side channel: %s",
                    str(e),
                )
                continue
            except Exception as e:
                logger.error("Failed to process receiver loop: %s", str(e))
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
                    "Receiver thread did not shut down cleanly within timeout")
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

        if nixl_config.role == NixlRole.SENDER:
            self._sender = NixlSender(
                nixl_config, config, backend)
        else:
            self._receiver = NixlReceiver(
                nixl_config, config, backend)

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
    ):
        """Prepare a send transaction by sending the request using
        the side channel.
        """
        sender = self._check_sender()
        sender.prepare_send(keys, mem_objs)

    def local_allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        """Allocate the memory for send."""
        return self._backend.local_allocate(shape, dtype, fmt)

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
def get_zmq_path(
    host: str, port: int, protocol: str = "tcp") -> str:   
    """Get the ZeroMQ path for the given base path and suffix."""
    if protocol == "tcp":
        return f"tcp://{host}:{port}"
    raise ValueError(f"Unsupported protocol: {protocol}")

@ dataclass
class NixlAgentWrapper:
    agent: NixlAgent
    reg_descs: Any
    xfer_descs: Any
    xfer_handler: Any
    
    def __init__(
        self,
        buffer_ptr: int,
        buffer_size:int,
        page_size: int,
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
        reg_descs = nixl_agent.get_reg_descs(memory_desc)
        nixl_agent.register_memory(reg_descs)

        # Create xfer handlers
        xfer_desc = []
        for base_addr in range(buffer_ptr, buffer_ptr + buffer_size,
                            page_size):
            xfer_desc.append((base_addr, page_size, 0))

        xfer_descs = nixl_agent.get_xfer_descs(xfer_desc)
        xfer_handler = nixl_agent.prep_xfer_dlist("", xfer_descs)

        self.nixl_agent = nixl_agent
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
