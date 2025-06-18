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
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.connector.nixl_utils import NixlConfig, NixlRole

logger = init_logger(__name__)


# FIXME: Should use msgspec
class NixlMsgBase(msgspec.Struct, tag=True):
    """Base class for all nixl-related messages"""
    pass
    
class NixlAllocRequest(NixlMsgBase):
    """
    """
    # receiver_id: str
    # req_uuid: str
    keys: list[str] # len(keys) indicates num_chunks
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

# no need to be msgspec
@dataclass
class NixlSenderTask:
    req_id: str
    receiver_zmq_base_path: str
    keys: list[CacheEngineKey]  # The keys to send
    mem_objs: list[MemoryObj]  # The memory objects to send
    
    # FIXME
    def get_alloc_request(self) -> NixlAllocRequest:
        """Get the allocation request for this sender task."""

        return NixlAllocRequest(keys=self.keys, last_chunk_toks=last_chunk_toks)
    
    # FIXME
    def get_local_indexes(self) -> list[int]:
        """
        Get the page indexes of the memory objects.
        This is needed for nixl transfer.
        """
        return
    
    def free_mem_objs(self):
        for mem_obj in self.mem_objs:
            mem_obj.ref_count_down()


# FIXME(Jiayi): Drop this
@dataclass
class NixlRequest:
    """
    A dataclass to represent a request received from the remote peer.
    This can be used to encapsulate the request information.
    """

    keys: list[CacheEngineKey]
    metadatas: list[MemoryObjMetadata]

    @staticmethod
    def encode_custom(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj).__name__} is not serializable")

    @staticmethod
    def decode_custom(d):
        if "__type__" not in d:
            return d
        t = d["__type__"]
        if t == "CacheEngineKey":
            return CacheEngineKey.from_dict(d)
        elif t == "MemoryObjMetadata":
            return MemoryObjMetadata.from_dict(d)
        elif t == "NixlRequest":
            return NixlRequest.from_dict(d)
        else:
            return d

    def to_dict(self):
        return {
            "__type__": "NixlRequest",
            "keys": [k.to_dict() for k in self.keys],
            "metadatas": [m.to_dict() for m in self.metadatas],
        }

    @staticmethod
    def from_dict(d):
        # Note(Kuntai): msgpack will automatically deserialize internal objects,
        # meaning d["keys"] and d["metadatas"] are already deserialized.
        return NixlRequest(keys=d["keys"], metadatas=d["metadatas"])

    def serialize(self) -> bytes:
        return msgpack.packb(self, default=NixlRequest.encode_custom)

    @staticmethod
    def deserialize(s: bytes) -> "NixlRequest":
        return msgpack.unpackb(s, object_hook=NixlRequest.decode_custom)


class NixlPipe:
    """An one-directional pipe to send the data from the sender to the receiver."""

    TRANSFER_BUFFER_SIZE = 128 * 1024 * 1024

    def __init__(
        self,
        nixl_config: NixlConfig,
        side_channel: Union[zmq.sugar.socket.Socket, "SenderSpecificSocket"],  # type: ignore
        sender_meta: Optional[bytes] = None,
    ):
        """
        Initialize the NixlPipe.

        Args:
            nixl_config: The NixlConfig object containing the configuration
                for the NIXL pipe.
            side_channel: The ZeroMQ socket used for communication.
            sender_meta: Optional metadata, will have values when the pipe
                it created on the receiver side and is connected by a
                sender.

        Note:
            We make sure that the receiver will not receive any other messages
            from the sender during __init__, so it will not disturb the main
            receiving loop on the receiver side.
        """
        self.nixl_config = nixl_config
        self.side_channel = side_channel

        if nixl_config.buffer_size > NixlPipe.TRANSFER_BUFFER_SIZE:
            assert nixl_config.buffer_size % NixlPipe.TRANSFER_BUFFER_SIZE == 0, (
                f"Buffer size must be a multiple of {NixlPipe.TRANSFER_BUFFER_SIZE}"
            )

        torch.cuda.set_device(nixl_config.buffer_device)
        self._buffer = torch.empty(
            nixl_config.buffer_size,
            device=nixl_config.buffer_device,
            dtype=torch.uint8,
        )

        self._transfer_buffers = torch.split(
            self._buffer, NixlPipe.TRANSFER_BUFFER_SIZE, dim=0
        )

        # allocator (should be initialized after self._buffer)
        self._allocator = NixlBufferAllocator(self)

        self._agent = nixl_agent(str(nixl_config.role) + str(nixl_config.buffer_device))
        self._reg_descs = self._agent.register_memory(self._transfer_buffers)
        self._local_xfer_descs = self._reg_descs.trim()
        self._remote_xfer_descs = None
        self._local_xfer_handlers = None
        self._remote_xfer_handlers = None

        local_meta = self._agent.get_agent_metadata()
        if nixl_config.role == NixlRole.SENDER:
            self.side_channel.send(local_meta)
            remote_meta = self.side_channel.recv()
            self.peer_name = self._agent.add_remote_agent(remote_meta).decode("utf-8")
        else:
            assert sender_meta is not None, (
                "The sender_meta should be provided on the receiver side"
            )
            self.peer_name = self._agent.add_remote_agent(sender_meta).decode("utf-8")
            self.side_channel.send(local_meta)

        # Exchange the reg_descs
        if nixl_config.role == NixlRole.SENDER:
            msg = self.side_channel.recv()
            self._remote_xfer_descs = self._agent.deserialize_descs(msg)
            logger.info("Received remote transfer descriptors")

            # Prepare the local and remote xfer_dlist_handler
            self._local_xfer_handlers = self._agent.prep_xfer_dlist(
                "", self._local_xfer_descs
            )
            self._remote_xfer_handlers = self._agent.prep_xfer_dlist(
                self.peer_name, self._remote_xfer_descs
            )
        else:
            # Receiver side, send the local descriptors
            self.side_channel.send(
                self._agent.get_serialized_descs(self._local_xfer_descs)
            )
            logger.info("Sent local transfer descriptors to sender")

        # UUID for communication
        self._uuid = None
        if nixl_config.role == NixlRole.RECEIVER:
            # Receiver send an initial uuid to sender
            self._uuid = uuid.uuid4().hex
            self.ack_receive()

    @_lmcache_nvtx_annotate
    def _spin_check_for_ack(self) -> str:
        """
        Spin until receives an ack from the peer.

        Returns:
            The uuid extracted from the ack message.
        """
        receiver_ready = False
        while not receiver_ready:
            notifs = self._agent.get_new_notifs()
            if self.peer_name not in notifs:
                time.sleep(0.001)
                continue

            for notif in notifs[self.peer_name]:
                decoded_uuid = message_to_uuid(notif.decode("utf-8"))
                if decoded_uuid is not None:
                    return decoded_uuid
            time.sleep(0.001)  # Avoid busy waiting

        raise RuntimeError("Failed to receive ACK from remote peer")

    # FIXME: should be in backend
    def local_allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        """
        """
        while True:
            memory_obj = self.allocator.allocate(
                shape, dtype, fmt)
            if memory_obj is not None:
                return memory_obj
            logger.warning(
                "Local NIXL buffer is full. "
                "Waiting for it to be freed."
            )
            time.sleep(0.001)  # Avoid busy waiting
    
    ###########################
    # Sender side functions
    ###########################
        
        # logger.debug(
        #     "Transfer %s completed in %.4f ms, creating the transfer: %.4f ms,"
        #     " transfer time: %.4f ms, pure transfer throughput: %.4f GB/s",
        #     uid,
        #     1000 * (t3 - t1),
        #     1000 * (t2 - t1),
        #     1000 * (t3 - t2),
        #     (write_size / (t3 - t2)) / (2**30),  # GB/s
        # )
        
    
    
    # @_lmcache_nvtx_annotate
    # def _commit_write(self, write_size: int, uid: str):
    #     """A blocking function that ensures the write buffer is delivered to
    #     the receiver.

    #     The transfer is initialized with the uuid.

    #     Args:
    #         write_size: the size of the data that is written into the buffer
    #         uuid: the uuid of the transfer

    #     Raises:
    #         RuntimeError: if the transfer fails
    #     """
    #     # Synchronize the default stream since the transfer happens in another
    #     # stream
    #     torch.cuda.default_stream().synchronize()

    #     # Send the data to the remote peer
    #     num_transfers = (write_size - 1) // NixlPipe.TRANSFER_BUFFER_SIZE + 1
    #     desc_indexes = list(range(num_transfers))
    #     logger.debug(
    #         f"Committing write of {write_size / 1024 / 1024} "
    #         f"MB with {num_transfers} transfers"
    #     )

    #     t1 = time.perf_counter()
    #     handle = self._agent.make_prepped_xfer(
    #         "WRITE",
    #         self._local_xfer_handlers,
    #         desc_indexes,
    #         self._remote_xfer_handlers,
    #         desc_indexes,
    #     )
    #     t2 = time.perf_counter()

    #     self._agent.transfer(handle)  # , uuid_to_message(uid))

    #     # NOTE: Potential optimization we don't immediately need to check
    #     # whether the transfer is done; Instead, we can check it before the
    #     # next time we allocate for write
    #     while (status := self._agent.check_xfer_state(handle)) != "DONE":
    #         if status == "PROC":
    #             time.sleep(0.001)  # Avoid busy waiting
    #         else:
    #             logger.error(
    #                 "Transfer failed with status: %s, handle: %s",
    #                 status,
    #                 handle,
    #             )
    #             raise RuntimeError(
    #                 f"Failed to send data to remote peer: {self.peer_name}, "
    #                 f"status: {status}"
    #             )
    #     t3 = time.perf_counter()

    #     self._agent.send_notif(self.peer_name, uuid_to_message(uid))

    #     logger.debug(
    #         "Transfer %s completed in %.4f ms, creating the transfer: %.4f ms,"
    #         " transfer time: %.4f ms, pure transfer throughput: %.4f GB/s",
    #         uid,
    #         1000 * (t3 - t1),
    #         1000 * (t2 - t1),
    #         1000 * (t3 - t2),
    #         (write_size / (t3 - t2)) / (2**30),  # GB/s
    #     )

    def allocate_for_write(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        """Allocate the memory for write.

        If the buffer is full, it will trigger a flush and then allocate
        the memory from the beginning.
        """
        # NOTE: the flush() is called in the allocator, which is not explicit
        # and may be confusing
        # return self._allocator.allocate(shape, dtype, fmt)

    def batched_remote_allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        """
        """
        pass
    
    @_lmcache_nvtx_annotate
    def flush(self):
        """Flush the buffer to the receiver side.
        Will also reset the allocator's allocated size to 0
        """
        self._uuid = self._spin_check_for_ack()
        logger.debug("Received ACK from remote peer with UUID: %s", self._uuid)
        size = self._allocator.num_bytes_allocated()
        self._commit_write(size, self._uuid)
        self._allocator.reset_allocated_size()

    ###########################
    # Receiver side functions
    ###########################
    
    def ack_receive(self):
        """Send an acknowledgment to the remote peer indicating that
        the transfer was received AND processed successfully.
        """
        self._uuid = uuid.uuid4().hex
        message = uuid_to_message(self._uuid)
        self._agent.send_notif(self.peer_name, message)
        logger.debug("Receiver acked the data with new UUID: %s", self._uuid)

    ###########################
    # Common functions
    ###########################
    def get_allocator(self) -> MemoryAllocatorInterface:
        """Get the underlying allocator for the NIXL pipe"""
        return self._allocator

    def close(self):
        """Close the NIXL pipe"""
        self._agent.deregister_memory(self._reg_descs)
        self._agent.remove_remote_agent(self.peer_name)
        if self._local_xfer_handlers is not None:
            self._agent.release_dlist_handle(self._local_xfer_handlers)
        if self._remote_xfer_handlers is not None:
            self._agent.release_dlist_handle(self._remote_xfer_handlers)





class NixlSender:
    """Handles sending data through a NixlPipe."""

    def __init__(self, nixl_config: NixlConfig):
        self.nixl_config = nixl_config

        # Initialize the ZeroMQ context and side channel
        self._context = zmq.Context()  # type: ignore
        # Change from PAIR to DEALER socket
        self._side_channel = self._context.socket(zmq.DEALER)  # type: ignore
        # Set an identity for this DEALER socket
        self._side_channel.setsockopt(
            zmq.IDENTITY,  # type: ignore
            f"sender-{uuid.uuid4().hex}".encode(),
        )  # type: ignore
        self._side_channel.connect(
            "tcp://{}:{}".format(nixl_config.receiver_host, nixl_config.receiver_port)
        )
        self._side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore

        # Create NIXL Pipe
        self._pipe = NixlPipe(nixl_config, self._side_channel)

        # Send state tracker
        self._during_send = False
        # How may objects are prepared to send
        self._prepared_count = 0
        # How many objects are added to the payload
        self._added_payload_count = 0
        
        self.req_queue = Queue()

    def get_allocator(self) -> MemoryAllocatorInterface:
        """Get the underlying allocator for the NIXL pipe"""
        return self._pipe.get_allocator()

    def prepare_send(
        self, keys: list[CacheEngineKey], mem_objs: list[MemoryObj]
    ):
        """
        Put the sender task into the request queue. 
        """
        
        sender_task = NixlSenderTask(
            keys=keys,
            mem_objs=mem_objs,
        )
        
        self.req_queue.put(sender_task)

    def _initialize_pipe(self):
        pass
    
    def _initialize_side_channel(self):
        pass
    
    def _get_side_channel(self, receiver_id: int):
        pass
    
    def _get_pipe(self, receiver_id: int) -> NixlPipe:
        pass
    
    
    def _remote_allocate(
        self, 
        side_channel, 
        alloc_request: NixlAllocRequest
    ) -> NixlAllocResponse:
        """Send the allocation request to the remote peer and get the response."""
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
        notif_msg = uuid_to_message(req_id)
        
        handle = self._agent.make_prepped_xfer(
            "WRITE",
            self._local_xfer_handlers,
            local_indexes,
            self._remote_xfer_handlers_list[receiver_id],
            remote_indexes,
            notif_msg,
        )
        
        self._agent.transfer(handle)
        
        while True:
            status = self._nixl_wrapper.check_xfer_state(handle)
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
        receiver_id: str
    ) -> None:
        """
        
        """
        
        # FIXME (Jiayi): init_socket here, should be zmq.REQ
        init_socket = 
        
        nixl_init_req = NixlInitRequest(
            sender_meta_bytes=self._nixl_sender_agent.get_agent_metadata(),
        )

        
        init_socket.send(msgspec.msgpack.encode(nixl_init_req))
        
        nixl_init_resp_bytes = init_socket.recv()
        nixl_init_resp = msgspec.msgpack.decode(
            nixl_init_resp_bytes, type=NixlMsg)
        
        remote_meta_bytes = nixl_init_resp.receiver_meta_bytes
        remote_agent_name = self._nixl_sender_agent.add_remote_agent(
                remote_meta_bytes
            )
        remote_xfer_dlist_bytes = nixl_init_resp.receiver_xfer_dlist_bytes
        remote_xfer_dlist = self._nixl_sender_agent.deserialize_descs(
                remote_xfer_dlist_bytes
            )
        
        remote_xfer_handlers = self._nixl_sender_agent.prep_xfer_dlist(
                remote_agent_name, remote_xfer_dlist)
        
        self._remote_xfer_handlers_list[
                receiver_id] = remote_xfer_handlers

        
    
    
    def _sender_loop(self):
        
        # FIXME: create sender (request) zmq socket in init

        while self._running:
            try:
                # FIXME: need to handle establish connection request
                # get sender request from queue
                sender_task = self.req_queue.get()
                receiver_zmq_base_path = sender_task.receiver_zmq_base_path
                req_id = sender_task.req_id
                
                # NOTE (Jiayi): Currently, a sender needs to connect to
                # 3 side channels:
                # (1) _init_side_channel (ad-hoc-established and destroyed
                # after nixl connection is established),
                # (2) _alloc_side_channel (ad-hoc-established),
                # (3) _proxy_side_channel (pre-established).
                
                # NOTE (Jiayi): A sender also needs to initialize nixl 
                # connection.
                
                # NOTE (Jiayi): `_init_all_comm` checks and initializes
                # _alloc_side_channel and nixl connection.
                
                self._init_all_comm(receiver_zmq_base_path)
            
                
                # use remote alloc
                alloc_request = sender_task.get_alloc_request()
                alloc_response = self._remote_allocate(
                    alloc_request)
                
                # send kv
                local_indexes = sender_task.get_local_indexes()
                remote_indexes = alloc_response.remote_indexes
                self._blocking_send(
                    req_id, receiver_zmq_base_path,
                    local_indexes, remote_indexes)
                                
                # free local memory
                sender_task.free_mem_objs()
                
            except Exception as e:
                logger.error("Failed to process receiver loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)    

    # FIXME
    def close(self):
        """Close the sender resources."""
        self._side_channel.close()
        self._context.term()
        self._pipe.close()


class NixlReceiver:
    """Handles receiving data through a NixlPipe."""

    # FIXME
    def __init__(self, nixl_config: NixlConfig):
        self.nixl_config = nixl_config

        # Initialize the ZeroMQ context and side channel
        self._context = zmq.Context()  # type: ignore
        # Change from PAIR to ROUTER socket
        self._side_channel = self._context.socket(zmq.ROUTER)  # type: ignore
        self._side_channel.bind(
            "tcp://{}:{}".format(nixl_config.receiver_host, nixl_config.receiver_port)
        )
        self._side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore
        # Add a timeout for the side channel
        self._side_channel.setsockopt(
            zmq.RCVTIMEO,  # type: ignore
            5000,  # Set a timeout for receiving to avoid blocking
        )

        # Track pipes for each sender
        self._sender_pipes: dict[bytes, NixlPipe] = {}

        # Observers
        self._observers: list[NixlObserverInterface] = []

        # Start the receiver thread
        self._running = True
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, daemon=True
        )
        self._receiver_thread.start()
        
        
    
    def _initialize_pipe(self):
        pass
    
    def _initialize_side_channel(self):
        pass
    
    def _get_side_channel(self, receiver_id: int):
        pass
    
    def _get_pipe(self, receiver_id: int) -> NixlPipe:
        pass
    
    
    # FIXME
    def _allocate_and_put(
        self, 
        alloc_request: NixlAllocRequest
    ) -> NixlAllocResponse:
        
        total_allocs = len(alloc_request.keys)
        alloc_indexes = []
        
        for key in alloc_request.keys:
            # FIXME
            mem_obj = self._backend.local_allocate(
                shape, 
                dtype,
                fmt
            )
            
            alloc_indexes.append(mem_obj.meta.address)
            
            # FIXME: need str to CacheEnginekey
            self._backend.put(key, mem_obj)
        
        return NixlAllocResponse(
            remote_indexes=alloc_indexes
        )

    # FIXME: have a loop wrapper to wrap different loops
    def _mem_alloc_loop(self):
        """
        """
        # FIXME: `self._running` might not be safe here
        while self._running:
            try:
                # FIXME: need to handle establish connection request
                # for both side channel and nixl pipe
                
                # FIXME
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
                logger.error("Failed to process receiver loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)
    
    def _recv_init_loop(self):
        while self.running:
            try:
                init_req_bytes = self._init_side_channel.recv()
                init_req = msgspec.msgpack.decode(
                    init_req_bytes, type=NixlMsg
                )
                assert isinstance(init_req, NixlInitRequest), (
                    "The request from the remote peer is not a NixlInitRequest")

                self._receiver_nixl_agent.add_remote_agent(
                    init_req.sender_meta_bytes)
                
                local_meta = self._receiver_nixl_agent.get_agent_metadata()
                
                local_xfer_descs = self._nixl_wrapper.\
                            get_serialized_descs(self._local_xfer_dlist)
                
                init_resp = NixlInitResponse(
                    receiver_meta_bytes=local_meta,
                    receiver_xfer_dlist_bytes=local_xfer_descs,
                )
                
                self._init_side_channel.send(
                    msgspec.msgpack.encode(init_resp)
                )
                
            except:
                pass
    
    
    def _recv_loop(self):
        
        while self._running:
            try:
                notifs = self._nixl_wrapper.get_new_notifs()
                for remote_agent_name in notifs:
                    for msg_bytes in notifs[remote_agent_name]:
                        msg = msgspec.msgpack.decode(
                            msg_bytes, type=NixlMsg
                        )
                        if isinstance(msg, NixlXferNotif):
                            # send ack to proxy
                            # FIXME: This channel can be push-pull
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

    def register_receive_observer(self, observer: NixlObserverInterface):
        """Register a new receive observer

        Args:
            observer: The observer to register
        """
        self._observers.append(observer)

    def close(self):
        """Close the receiver resources."""
        self._running = False
        # Wait for the receiver thread to finish with timeout
        self._receiver_thread.join(timeout=3.0)  # 3 second timeout
        if self._receiver_thread.is_alive():
            logger.warning("Receiver thread did not shut down cleanly within timeout")

        # Close all pipes
        for sender_id, pipe in self._sender_pipes.items():
            logger.info(f"Closing pipe for sender: {sender_id.decode()}")
            pipe.close()

        self._side_channel.close()
        self._context.term()


# Helper class to route messages to specific senders
class SenderSpecificSocket:
    """A wrapper around a ROUTER socket that only communicates with a specific
    sender.
    """

    def __init__(
        self,
        router_socket: zmq.Socket,  # type: ignore
        sender_id: bytes,
    ):
        self.router_socket = router_socket
        self.sender_id = sender_id

    def send(self, data: bytes):
        """Send data to the specific sender."""
        self.router_socket.send_multipart([self.sender_id, data])

    def recv(self) -> bytes:
        """Receive data from the specific sender.

        This is a simplified implementation that assumes messages are only
        coming from the specific sender. In a real implementation, you would
        need to filter messages by sender_id.
        """
        frames = self.router_socket.recv_multipart()
        if frames[0] == self.sender_id:
            return frames[1]
        else:
            logger.warning(f"Received message for wrong sender: {frames[0].decode()}")
            return b""


class NixlChannel:
    """Provides the primitives to send the data and process the received data.
    It will have some internal threads to handle the data receiving.
    """

    def __init__(self, nixl_config: NixlConfig):
        self.nixl_config = nixl_config
        self.role = nixl_config.role

        # Create sender or receiver based on role
        self._sender = None
        self._receiver = None

        if nixl_config.role == NixlRole.SENDER:
            self._sender = NixlSender(nixl_config)
        else:
            self._receiver = NixlReceiver(nixl_config)

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

    def get_allocator(self) -> MemoryAllocatorInterface:
        """Get the underlying allocator for the NIXL pipe"""
        sender = self._check_sender()
        return sender.get_allocator()

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

    def finish_send(self):
        """Finish the send transaction by flushing the buffer."""
        sender = self._check_sender()
        sender.finish_send()



    def register_receive_observer(self, observer: NixlObserverInterface):
        """Register a new receive observer"""
        receiver = self._check_receiver()
        receiver.register_receive_observer(observer)

    def close(self):
        """Close all resources."""
        if self._sender:
            self._sender.close()
        if self._receiver:
            self._receiver.close()


############################################################
# helper functions
############################################################
def uuid_to_message(uid: str) -> str:
    """Convert the uuid to the message"""
    return f"NIXL_TRANSFER_{uid}"


def message_to_uuid(message: str) -> Optional[str]:
    """Convert the message to the uuid"""
    if not message.startswith("NIXL_TRANSFER_"):
        return None
    return message[len("NIXL_TRANSFER_") :]


def get_zmq_path(base_path: str, role)


def init_nixl_agent(
    buffer_size: int,
    buffer_ptr: int,
    nixl_page_size: int = 4096,
) -> tuple[NixlAgent, Any, Any, Any]:
    """
    Initialize the NIXL agent.

    Args:
        buffer_size (int): The size of the buffer.
        buffer_ptr (int): The pointer to the buffer.
        nixl_page_size (int, optional): The page size of NIXL. Defaults to 4096.

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
    reg_descs = nixl_agent.get_reg_descs(memory_desc, mem_type="DRAM")
    nixl_agent.register_memory(reg_descs)

    # Create xfer handlers
    xfer_desc = []
    for base_addr in range(buffer_ptr, buffer_ptr + buffer_size,
                           nixl_page_size):
        xfer_desc.append((base_addr, nixl_page_size, 0))

    xfer_descs = nixl_agent.get_xfer_descs(xfer_desc, mem_type="DRAM")
    xfer_handler = nixl_agent.prep_xfer_dlist("", xfer_descs, mem_type="DRAM")

    return nixl_agent, reg_descs, xfer_descs, xfer_handler
