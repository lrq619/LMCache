# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Any, Optional, Union
import copy
import threading
import time
import uuid

# Third Party
import msgspec
import torch
import zmq
from zmq.utils.monitor import parse_monitor_message

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
from lmcache.v1.storage_backend.connector.nixl_utils import NixlConfigXpYd, NixlRole

import socket

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
from concurrent.futures import Future
import os
import nvtx

if TYPE_CHECKING:
    # Third Party
    from nixl._api import NixlAgent

    # First Party
    from lmcache.v1.storage_backend.nixl_backend_v3 import NixlBackend
    
_MAX_INFLIGHT_ALLOC_PER_RECV = 128     # 每个 receiver 在途 alloc 上限
_MAX_INFLIGHT_XFER_GLOBAL     = 256   # 全局在途 xfer 上限
_POLL_CHECK_HANDLES_PER_TICK  = 128    # 每 tick 检查的 handle 数
_IDLE_BACKOFF_SEC_MIN         = 0.0003
_IDLE_BACKOFF_SEC_MAX         = 0.002

logger = init_logger(__name__)


class NixlMsgBase(msgspec.Struct, tag=True):
    """Base class for all nixl-related messages"""

    pass


class NixlAllocRequest(NixlMsgBase):
    """Nixl allocation request message"""

    keys: list[str]  # len(keys) indicates num_chunks
    fmt: int
    shape: list[int]  # The shape of the memory objects
    dtype: str
    last_chunk_toks: int
    req_id: str
    is_cuda: bool = True  # Whether the memory objects are on CUDA
    delete: bool = False


class NixlAllocResponse(NixlMsgBase):
    """Nixl allocation response message"""

    # Indexes (local) of already sent memory objects
    already_sent_indexes: list[int]

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
    transfer_spec: Any = None

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
        logger.info(f"in send task, keys is {self.keys}")
        keys = [key.to_string() for key in self.keys]
        logger.info(f"send key_str is {keys}")

        return NixlAllocRequest(
            keys=keys,
            fmt=fmt.value,
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks,
            req_id=self.req_id
        )

    # TODO (Jiayi): reduce for loop
    def get_local_indexes(
        self,
        already_sent_indexes: list[int],
    ) -> list[int]:
        """
        Get the page indexes of the memory objects.
        This is needed for nixl transfer.
        """
        local_indexes = []
        for idx, mem_obj in enumerate(self.mem_objs):
            if idx in already_sent_indexes:
                continue
            local_indexes.append(mem_obj.meta.address)
        return local_indexes

    def free_mem_objs(self):
        for mem_obj in self.mem_objs:
            mem_obj.ref_count_down()

_MSG_ENC = msgspec.msgpack.Encoder()
_MSG_DEC = msgspec.msgpack.Decoder(type=NixlMsg)

@dataclass
class _TaskCtx:
    req_id: str
    receiver_id: str
    task: NixlSenderTask
    future: Future
    local_indexes: Optional[List[int]] = None
    remote_indexes: Optional[List[int]] = None
    start_ts_ms: Optional[float] = None
    start_ac_ms: Optional[float] = None
    end_ac_ms: Optional[float] = None

class NixlSender:
    """Handles sending data through a NixlPipe."""

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: "NixlBackend",
        tp_rank: int,
    ):
        assert nixl_config.role == NixlRole.SENDER, (
            "NixlSender should only be initialized with NixlRole.SENDER"
        )

        self.device = nixl_config.buffer_device
        self._dst_device_str = str(nixl_config.buffer_device)

        self.nixl_config = nixl_config
        self._backend = backend
        self.memory_allocator = backend.memory_allocator

        self._sender_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.nixl_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.nixl_allocator.buffer_size,
            page_size=self.memory_allocator.nixl_allocator.align_bytes,
            tp_rank=tp_rank,
            mem_type="cuda"
        )
        self._sender_cpu_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.nixl_allocator.cpu_buffer_ptr,
            buffer_size=self.memory_allocator.nixl_allocator.cpu_buffer_size,
            page_size=self.memory_allocator.nixl_allocator.align_bytes,
            tp_rank=tp_rank,
            mem_type="DRAM"
        )
        
        self._nixl_agent = self._sender_nixl_wrapper.agent
        self._nixl_cpu_agent = self._sender_cpu_nixl_wrapper.agent

        # Initialize the ZeroMQ context
        self._context = zmq.Context()

        self._mem_alloc_sockets: dict[str, zmq.Socket] = {}

        self.req_queue: Queue[NixlSenderTask] = Queue()

        self._remote_xfer_handlers_dict: dict[
            str, NixlAgent.nixl_prepped_dlist_handle
        ] = {}
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

        self._proxy_side_channel = self._context.socket(zmq.PUSH)
        self._proxy_side_channel.connect(get_zmq_path(proxy_url, protocol="tcp"))

        self.tp_rank = tp_rank
        
        self.notified_req_set = set()
        self.notified_req_lock = threading.Lock()
        
        # for async io
        self._send_q: Queue[NixlSenderTask] = Queue(maxsize=512)

        self._io_thr: Optional[threading.Thread] = None
        self._stop = False

        self._zmq_ctx = zmq.Context.instance()
        self._poller = zmq.Poller()

        self._alloc_peers: dict[str, dict] = {}

        # in-flight 传输：handle -> (agent, _TaskCtx, mem_objs)
        self._inflight_xfers: dict[object, tuple[object, _TaskCtx, list[MemoryObj]]] = {}

        # partial prefill 的 2s 错误定时器 & 锁
        self._partial_err_timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

        # receiver -> alloc endpoint（用于异步 DEALER；同步通道继续用 _mem_alloc_sockets）
        self._alloc_endpoint_of: dict[str, str] = {}
        self.mem_obj_map = {}
        self.mem_obj_map_lock = threading.Lock()
        self._stat_thread = threading.Thread(target=self._stat_loop, name="nixl-sender-stat-loop", daemon=True)
        self._stat_thread.start()

    def _stat_loop(self):
        while self._running:
            try:
                total_allocated_size, total_allocated_size_cpu = self._backend.get_allocated_size()
                instance_uuid = os.environ.get("INSTANCE_UUID", "unkown")
                max_lifespan = 0
                oldest_req_id = ""
                now = time.time()
                with self.mem_obj_map_lock:
                    for req_id, mem_objs in self.mem_obj_map.items():
                        mem_obj = mem_objs[0]
                        lifespan = now - mem_obj.allocated_ts 
                        if lifespan > max_lifespan:
                            max_lifespan = lifespan
                            oldest_req_id = req_id

                logger.info(
                    f"[SenderStat], uuid:{instance_uuid}, tp_rank: {self.tp_rank}, "
                    f"total gpu allocated size: {total_allocated_size / (1024*1024):.2f} MB, "
                    f"total cpu allocated size: {total_allocated_size_cpu / (1024*1024):.2f} MB, "
                    f"max lifespan: {max_lifespan * 1000:.1f}ms, "
                    f"req_id with max lifespan: {oldest_req_id}, "
                )
            except Exception as e:
                logger.exception(f"[Sender] Exception in stat loop: {e}")
            time.sleep(1)

    def _start_io_thread(self):
        self._io_thr = threading.Thread(target=self._io_loop, name="nixl-async-io", daemon=True)
        self._io_thr.start()
    
    def _io_loop(self):
        try:
            if self._dst_device_str.startswith("cuda"):
                if ":" in self._dst_device_str:
                    idx = int(self._dst_device_str.split(":")[1])
                else:
                    idx = torch.cuda.current_device()
                torch.cuda.set_device(idx)
                torch.cuda.synchronize() 
        except Exception as e:
            logger.error(f"[NixlBackend] failed to set CUDA device in sender thread: {e}")
            raise
        backoff = _IDLE_BACKOFF_SEC_MIN
        while not self._stop:
            progressed = False

            progressed |= self._pump_new_tasks()

            if self._alloc_peers:
                events = dict(self._poller.poll(timeout=0))
                progressed |= self._pump_alloc_writable(events)
                progressed |= self._pump_alloc_readable(events)

            progressed |= self._poll_transfers()

            if not progressed:
                time.sleep(backoff)
                backoff = min(backoff * 1.2, _IDLE_BACKOFF_SEC_MAX)
            else:
                backoff = _IDLE_BACKOFF_SEC_MIN

        logger.info("nixl-async-io thread exiting")
        
    def _ensure_peer(self, receiver_id: str, endpoint: str):
        peer = self._alloc_peers.get(receiver_id)
        if peer is not None:
            return peer
        sock = self._zmq_ctx.socket(zmq.DEALER)
        # sock.setsockopt(zmq.TCP_NODELAY, 1)
        # sock.setsockopt(zmq.IMMEDIATE, 1)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 0)
        logger.info(f"connecting alloc DEALER socket to {endpoint} for receiver {receiver_id}")
        sock.connect(endpoint)
        self._poller.register(sock, zmq.POLLIN | zmq.POLLOUT)
        peer = dict(sock=sock, outbox=deque(), pending={}, inflight=0)
        self._alloc_peers[receiver_id] = peer
        
        mon = sock.get_monitor_socket(zmq.EVENT_CONNECTED | zmq.EVENT_HANDSHAKE_SUCCEEDED)
        timeout_sec = 5.0
        mon.setsockopt(zmq.RCVTIMEO, int(timeout_sec * 1000))  # 超时就算了
        connected = False
        try:
            while not connected:
                evt = parse_monitor_message(mon.recv_multipart())  # 阻塞直到事件或超时
                if evt["event"] in (zmq.EVENT_HANDSHAKE_SUCCEEDED, zmq.EVENT_CONNECTED):
                    connected = True
        except zmq.Again:
            # 超时也行：后续 send(NOBLOCK) 可能会 Again，由 outbox 兜底
            logger.warning(f"timeout waiting for connect to {endpoint} for receiver {receiver_id}")
            pass
        
        return peer
    
    @_lmcache_nvtx_annotate
    def _pump_new_tasks(self) -> bool:
        progressed = False
        for _ in range(64):  # 每 tick 吸一小撮，避免长时间占用循环
            try:
                task = self._send_q.get_nowait()
            except Exception:
                break

            req_id = task.req_id
            rid = task.receiver_info.receiver_id

            if not self._check_init(task.receiver_info):
                self._init_all_comm(task.receiver_info)

            # 准备 DEALER peer
            endpoint = self._alloc_endpoint_of[rid]
            peer = self._ensure_peer(rid, endpoint)

            # 构造 alloc 请求帧
            alloc_req = task.get_alloc_request()
            alloc_req.is_cuda = self._remote_xfer_handlers_is_cuda_dict.get(rid, True)
            payload = _MSG_ENC.encode(alloc_req)
            mv = memoryview(payload)

            # 提前算好 local_indexes，回包后要立刻送 xfer
            local_indexes = task.get_local_indexes(already_sent_indexes=[])
            fut = task._async_future

            ctx = _TaskCtx(req_id=req_id, receiver_id=rid, task=task,
                           future=fut, local_indexes=local_indexes)

            # 受 in-flight 限制；能发就发，不能发就入 outbox
            if peer["inflight"] >= _MAX_INFLIGHT_ALLOC_PER_RECV:
                peer["outbox"].append((req_id, mv))
                peer["pending"][req_id] = ctx
            else:
                try:
                    logger.info(f"alloc try-send NOW req={req_id} rid={rid} sock={peer['sock']} on tp_rank: {self.tp_rank}")
                    peer['sock'].send_multipart([req_id.encode(), mv], flags=zmq.NOBLOCK, copy=False)
                    logger.info(f"alloc sent NOW req={req_id} rid={rid} sock={peer['sock']} peer is {peer} on tp_rank: {self.tp_rank}")
                    ctx.start_ac_ms = time.time() * 1000
                    peer["pending"][req_id] = ctx
                    peer["inflight"] += 1
                except zmq.Again as e:
                    logger.info(f"alloc send WOULD-BLOCK req={req_id} rid={rid} sock={peer['sock']} on tp_rank: {self.tp_rank}")
                    import traceback
                    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    logger.error("failed:\n%s", tb)
                    peer["outbox"].append((req_id, mv))
            progressed = True
        return progressed
    
    @_lmcache_nvtx_annotate
    def _pump_alloc_writable(self, events: dict) -> bool:
        progressed = False
        for rid, peer in self._alloc_peers.items():
            sock = peer['sock']
            if sock not in events or not (events[sock] & zmq.POLLOUT):
                continue
            while peer["outbox"] and peer["inflight"] < _MAX_INFLIGHT_ALLOC_PER_RECV:
                req_id, mv = peer["outbox"][0]
                try:
                    sock.send_multipart([req_id.encode(), mv], flags=zmq.NOBLOCK, copy=False)
                    peer["inflight"] += 1
                    peer["outbox"].popleft()
                    progressed = True
                except zmq.Again:
                    break
        return progressed
    
    @_lmcache_nvtx_annotate
    def _pump_alloc_readable(self, events: dict) -> bool:
        progressed = False
        for rid, peer in self._alloc_peers.items():
            sock = peer['sock']
            if sock not in events or not (events[sock] & zmq.POLLIN):
                continue
            while True:
                try:
                    req_id, msg_b = sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

                req_id = str(req_id.decode())
                logger.info(f"alloc got-reply req_id={req_id} on tp_rank: {self.tp_rank}")

                # logger.info(f"pending add: rid={rid}; pending_keys={list(peer['pending'].keys())}, peer is {peer}")
                ctx: _TaskCtx = peer["pending"].pop(req_id, None)
                peer["inflight"] = max(0, peer["inflight"] - 1)
                ctx.end_ac_ms = time.time() * 1000
                logger.info(f"alloc DONE for req_id={ctx.req_id}, duration: {ctx.end_ac_ms - ctx.start_ac_ms} ms on tp_rank: {self.tp_rank}")

                if ctx is None:
                    logger.warning("alloc reply for unknown req_id=%s", req_id)
                    continue

                # 解析回包并校验
                resp = _MSG_DEC.decode(memoryview(msg_b))
                if getattr(resp, "__class__", type(resp)).__name__ != "NixlAllocResponse":
                    self._fail_ctx(ctx, RuntimeError(f"bad alloc response type: {type(resp)}"))
                    progressed = True
                    continue

                remote_indexes = getattr(resp, "remote_indexes", None)
                if not remote_indexes:
                    self._fail_ctx(ctx, RuntimeError("remote allocate returned empty indexes"))
                    progressed = True
                    continue

                ctx.remote_indexes = list(remote_indexes)

                # 分配成功 -> 立即提交 xfer（非阻塞）
                self._submit_send_nb(ctx)
                progressed = True
        return progressed
    
    @_lmcache_nvtx_annotate
    def _submit_send_nb(self, ctx: _TaskCtx):
        start = time.perf_counter()
        
        receiver_id = ctx.receiver_id
        is_cuda = self._remote_xfer_handlers_is_cuda_dict.get(receiver_id, True)
        agent = self._nixl_agent if is_cuda else self._nixl_cpu_agent
        wrapper = self._sender_nixl_wrapper if is_cuda else self._sender_cpu_nixl_wrapper
        remote_hdl = self._remote_xfer_handlers_dict[receiver_id]

        transfer_start = time.perf_counter()
        handle = agent.make_prepped_xfer(
            "WRITE",
            wrapper.xfer_handler,
            ctx.local_indexes,
            remote_hdl,
            ctx.remote_indexes,
        )
        transfer_end = time.perf_counter()
        logger.info(f"making handler takes: {transfer_end-transfer_start}")
        
        # 提交（不要等待）
        logger.info("submitting xfer handle=%s for req_id=%s", handle, ctx.req_id)
        ctx.start_ts_ms = time.time() * 1000 
        with nvtx.annotate("launch transfer", color="yellow"):
            transfer_start = time.perf_counter()
            try:
                agent.transfer(handle)
            except Exception as e:
                logger.error(f"Transfer failed for handle={handle}: {e}")
                raise e
            transfer_end = time.perf_counter()
            logger.info(f"transfer call takes: {transfer_end-transfer_start} on tp_rank: {self.tp_rank}")

        # 控制全局在途上限（可选：如超过则下个 tick 再检查）
        if len(self._inflight_xfers) >= _MAX_INFLIGHT_XFER_GLOBAL:
            logger.debug("global inflight xfer at cap=%d", _MAX_INFLIGHT_XFER_GLOBAL)

        # 记录在途
        mem_objs = ctx.task.mem_objs
        self._inflight_xfers[handle] = (agent, ctx, mem_objs)
        
        end = time.perf_counter()
        logger.info(f"_submit_send_nb call takes: {transfer_end-transfer_start} on tp_rank: {self.tp_rank}")
        
    @_lmcache_nvtx_annotate
    def _poll_transfers(self) -> bool:
        progressed = False
        if not self._inflight_xfers:
            return progressed

        for handle in list(self._inflight_xfers.keys()):
            agent, ctx, mem_objs = self._inflight_xfers.get(handle, (None, None, None))
            if agent is None:
                continue
            try:
                st = agent.check_xfer_state(handle)  # 非阻塞
            except Exception as e:
                self._inflight_xfers.pop(handle, None)
                self._fail_ctx(ctx, e)
                progressed = True
                continue

            if st == "PROC":
                continue

            progressed = True
            self._inflight_xfers.pop(handle, None)

            if st == "DONE":
                ctx.end_ts_ms = time.time() * 1000
                logger.info(f"transfer DONE for req_id={ctx.req_id}, duration: {ctx.end_ts_ms - ctx.start_ts_ms} ms on tp_rank: {self.tp_rank}")
                self._finish_ctx_ok(ctx)
            else:
                self._fail_ctx(ctx, RuntimeError(f"send error, status={st}"))

        return progressed
    
    def _finish_ctx_ok(self, ctx: _TaskCtx):
        req_id = ctx.req_id
        task = ctx.task

        with self.mem_obj_map_lock:
            assert req_id in self.mem_obj_map, f"req {req_id} not in self.mem_obj_map"
            del self.mem_obj_map[req_id]
        logger.info(f"req {req_id} finished transfer without any error on tp_rank: {self.tp_rank}")
        # last prefill -> 立即通知 & 取消 2s 错误定时器
        if task.transfer_spec.is_last_prefill:
            self._send_notify_msg(req_id)
            # with self._lock:
            #     t = self._partial_err_timers.pop(req_id, None)
            # if t:
            #     t.cancel()

        fut = ctx.future
        if fut and not fut.done():
            fut.set_result(True)

        try:
            task.free_mem_objs()
        except Exception:
            pass

    def _fail_ctx(self, ctx: _TaskCtx, err: Exception):
        logger.error(f"req_id={ctx.req_id} failed with error: {err}", exc_info=True)
        req_id = ctx.req_id
        with self.mem_obj_map_lock:
            assert req_id in self.mem_obj_map, f"req {req_id} not in self.mem_obj_map"
            del self.mem_obj_map[req_id]
        try:
            self._send_error_msg(req_id)
        except Exception:
            pass

        fut = ctx.future
        if fut and not fut.done():
            fut.set_exception(err)

        try:
            ctx.task.free_mem_objs()
        except Exception:
            pass

        # 取消 2s 定时器
        with self._lock:
            t = self._partial_err_timers.pop(req_id, None)
        if t:
            t.cancel()
    
    def prepare_send_async(
        self,
        keys: list[CacheEngineKey],
        mem_objs: list[MemoryObj],
        transfer_spec=None,
    ) -> Future:
        assert transfer_spec is not None
        receiver_info = copy.deepcopy(transfer_spec.receiver_info)
        logger.info(f"receiver_info is {receiver_info}, tp rank is {self.tp_rank}")
        receiver_info.receiver_init_port = (
            transfer_spec.receiver_info.receiver_init_port[self.tp_rank]
        )
        receiver_info.receiver_alloc_port = (
            transfer_spec.receiver_info.receiver_alloc_port[self.tp_rank]
        )
        receiver_info.receiver_id = transfer_spec.receiver_info.receiver_host + str(
            receiver_info.receiver_init_port
        )
        
        sender_task = NixlSenderTask(
            req_id=transfer_spec.req_id,
            receiver_info=receiver_info,
            keys=keys,
            mem_objs=mem_objs,
            transfer_spec=transfer_spec
        )

        req_id = sender_task.req_id
        rid = sender_task.receiver_info.receiver_id
        logger.debug("prepare_send_async: enqueue req=%s -> receiver=%s (%d objs)",
                     req_id, rid, len(keys))

        with self.mem_obj_map_lock:
            self.mem_obj_map[req_id] = mem_objs

        fut = Future()

        # if not transfer_spec.is_last_prefill:
        #     timer = threading.Timer(2.0, self._send_error_msg, args=[req_id])
        #     with self._lock:
        #         self._partial_err_timers[req_id] = timer
        #     timer.start()

        self._send_q.put(sender_task)

        if self._io_thr is None:
            self._start_io_thread()

        sender_task._async_future = fut
        return fut

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
        logger.info(f"Notified kv ready for req: {req_id} on tp_rank: {self.tp_rank}")
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

        receiver_info = copy.deepcopy(transfer_spec.receiver_info)
        receiver_info.receiver_init_port = (
            transfer_spec.receiver_info.receiver_init_port[self.tp_rank]
        )
        receiver_info.receiver_alloc_port = (
            transfer_spec.receiver_info.receiver_alloc_port[self.tp_rank]
        )
        receiver_info.receiver_id = transfer_spec.receiver_info.receiver_host + str(
            receiver_info.receiver_init_port
        )

        sender_task = NixlSenderTask(
            req_id=transfer_spec.req_id,
            receiver_info=receiver_info,
            keys=keys,
            mem_objs=mem_objs,
        )

        logger.info(
            "Preparing to send %s objs with request ID: %s to receiver: %s on tp_rank: %d",
            len(sender_task.keys),
            sender_task.req_id,
            receiver_info,
            self.tp_rank
        )

        # self.req_queue.put(sender_task)

        req_id = sender_task.req_id
        receiver_id = receiver_info.receiver_id

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
        if not self._check_init(receiver_info):
            self._init_all_comm(receiver_info)

        # use remote alloc
        alloc_request = sender_task.get_alloc_request()

        alloc_response = self._remote_allocate(receiver_id, alloc_request)

        # send kv
        local_indexes = sender_task.get_local_indexes(
            alloc_response.already_sent_indexes
        )
        remote_indexes = alloc_response.remote_indexes

        # NOTE (vladnosiv): len(local_indexes) may be zero
        # if the requests in the batch have a large common prefix
        if not local_indexes:
            logger.debug(
                "Sending objs with request ID: %s is not required: "
                "all indexes already sent",
                sender_task.req_id,
            )
        else:
            self._blocking_send(req_id, receiver_id, local_indexes, remote_indexes)

        logger.debug(f"transfer spec: {transfer_spec}")
        if transfer_spec.is_last_prefill:
            # Notify the proxy that the transfer is done
            notif_msg = NixlProxyNotif(req_id=req_id)
            notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
            self._proxy_side_channel.send(notif_msg_bytes)

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

        handle = self._nixl_agent.make_prepped_xfer(
            "WRITE",
            self._sender_nixl_wrapper.xfer_handler,
            local_indexes,
            self._remote_xfer_handlers_dict[receiver_id],
            remote_indexes,
            # notif_msg_bytes,
        )

        # NOTE (Jiayi): cannot make this transfer in another thread,
        # giving error: `UCX  ERROR cuCtxGetDevice(&key.cu_device)
        # failed: invalid device context`
        self._nixl_agent.transfer(handle)

        # TODO (Jiayi): offload the following to another thread
        # TODO (Jiayi) tune hyperparameters
        wait_time = 0.0007
        decay = 1.1
        while True:
            status = self._nixl_agent.check_xfer_state(handle)
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
        logger.info(f"receiver_info.receiver_init_port {receiver_info.receiver_init_port}, tp_rank: {self.tp_rank}")
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
        
        alloc_endpoint = get_zmq_path(receiver_mem_alloc_url, protocol="tcp")
        self._alloc_endpoint_of[receiver_id] = alloc_endpoint

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
        self._sender_cpu_nixl_wrapper.close(self._remote_xfer_handlers_dict)


class NixlReceiver:
    """Handles receiving data through a NixlPipe."""

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: "NixlBackend",
        tp_rank: int,
    ):
        assert nixl_config.role == NixlRole.RECEIVER, (
            "NixlReceiver should only be initialized with NixlRole.RECEIVER"
        )

        self._backend = backend
        self.memory_allocator = backend.memory_allocator
        self.tp_rank = tp_rank

        self.device = nixl_config.buffer_device
        self._receiver_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.nixl_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.nixl_allocator.buffer_size,
            page_size=self.memory_allocator.nixl_allocator.align_bytes,
            tp_rank=tp_rank,
            mem_type="cuda",
        )
        self._receiver_cpu_nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=self.memory_allocator.nixl_allocator.cpu_buffer_ptr,
            buffer_size=self.memory_allocator.nixl_allocator.cpu_buffer_size,
            page_size=self.memory_allocator.nixl_allocator.align_bytes,
            tp_rank=tp_rank,
            mem_type="DRAM"
        )

        self._nixl_agent = self._receiver_nixl_wrapper.agent
        self._nixl_cpu_agent = self._receiver_cpu_nixl_wrapper.agent

        self.nixl_config = nixl_config

        receiver_host = nixl_config.peer_host
        receiver_init_port = nixl_config.peer_init_port[tp_rank]
        receiver_alloc_port = nixl_config.peer_alloc_port[tp_rank]
        receiver_delete_ports = os.environ["NIXL_DELETE_PORT"].split(",")
        receiver_delete_port = receiver_delete_ports[tp_rank]
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
        # self._alloc_side_channel = self._context.socket(zmq.REP)
        # self._alloc_side_channel.bind(get_zmq_path(receiver_alloc_url, protocol="tcp"))
        # self._side_channels.append(self._alloc_side_channel)
        self._alloc_side_channel = self._context.socket(zmq.ROUTER)
        self._alloc_side_channel.setsockopt(zmq.IMMEDIATE, 1)
        self._alloc_side_channel.setsockopt(zmq.LINGER, 0)
        self._alloc_side_channel.bind(get_zmq_path(receiver_alloc_url, "tcp"))
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
                total_allocated_size, total_allocated_size_cpu = self._backend.get_allocated_size()
                max_lifespan = self._backend.get_max_lifespan()
                oldest_req_id = self._backend.get_olddest_req_id()
                put_speed, get_speed = self._backend.stat()
                key_length = self._backend.get_data_key_length()
                valid_obj_num = self._backend.get_num_valid_mem_obj()
                instance_uuid = os.environ.get("INSTANCE_UUID", "unkown")
                logger.info(
                    f"[ReceiverStat], uuid:{instance_uuid}, tp_rank: {self.tp_rank}, "
                    f"total gpu allocated size: {total_allocated_size / (1024*1024):.2f} MB, "
                    f"total cpu allocated size: {total_allocated_size_cpu / (1024*1024):.2f} MB, "
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
                    NixlAllocResponse(already_sent_indexes=[], remote_indexes=[])))
                logger.info(
                    "Received delete request for %s on tp_rank %d, deleted: %s",
                    delete_req.req_id,
                    self.tp_rank,
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

        keys = []
        mem_objs = []
        for idx, key_str in enumerate(alloc_request.keys):
            # logger.info(f"receive key_str is {key_str}")
            key = CacheEngineKey.from_string(key_str)
            # logger.info(f"receive key is {key}")

            if idx == total_allocs - 1:
                num_alloc_tokens = alloc_request.last_chunk_toks
                token_dim = fmt.token_dim()
                shape[token_dim] = num_alloc_tokens
            else:
                num_alloc_tokens = self.full_chunk_size

            mem_obj = None
            wait_time = 0.05
            decay = 1.1
            while mem_obj is None:
                req_id = alloc_request.req_id
                # check if delete signal has been issued, if so, break the loop
                with self.deleted_reqs_lock:
                    if req_id in self.deleted_reqs_set:
                        logger.info(f"req: {req_id} already deleted by delete thread on tp_rank: {self.tp_rank}, stop trying to allocate")
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
                return NixlAllocResponse(already_sent_indexes=[], remote_indexes=[])
            keys.append(key)
            mem_objs.append(mem_obj)

        for i, key in enumerate(keys):
            mem_obj = mem_objs[i]
            alloc_indexes.append(mem_obj.meta.address)

            self._backend.put(key, mem_obj)
            
        if len(keys) > 0: 
            logger.info(f"put {len(keys)} mem_objs for req: {mem_objs[0].req_ids} for tp_rank: {self.tp_rank}")

        return NixlAllocResponse(
            already_sent_indexes=[], remote_indexes=alloc_indexes
        )

    # TODO: have a loop wrapper to wrap different loops
    def _mem_alloc_loop(self):
        """ """
        torch.cuda.set_device(self.device)
        # TODO: `self._running` might not be safe here
        while self._running:
            try:
                # NOTE: this is a req-reply zmq for now
                # receive alloc request
                # alloc_req_bytes = self._alloc_side_channel.recv()
                # alloc_req = msgspec.msgpack.decode(alloc_req_bytes, type=NixlMsg)
                frames = self._alloc_side_channel.recv_multipart()
                if len(frames) != 3:
                    logger.warning("alloc: bad frame count=%d", len(frames))
                    continue
                routing_id, req_id_b, msg_b = frames
                alloc_req = msgspec.msgpack.decode(msg_b, type=NixlMsg)
                assert isinstance(alloc_req, NixlAllocRequest), (
                    "The request from the remote peer is not a NixlAllocRequest"
                )

                logger.debug(
                    "Received allocation request for %s objs",
                    len(alloc_req.keys),
                )

                # NOTE: it's okay to put the memory objs into the storage backend
                # first because decode vllm will not be able to see the decode
                # request until proxy receives the ack.
                alloc_resp = self._allocate_and_put(alloc_req)

                logger.debug(
                    "Replying allocation response for %s objs for %s req",
                    len(alloc_resp.remote_indexes), alloc_req.req_id
                )

                # send back response
                # self._alloc_side_channel.send(msgspec.msgpack.encode(alloc_resp))
                resp_b = msgspec.msgpack.encode(alloc_resp)
                self._alloc_side_channel.send_multipart([routing_id, req_id_b, resp_b])

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
        self._receiver_cpu_nixl_wrapper.close()


class NixlChannel:
    """Provides the primitives to send the data and process the received data.
    It will have some internal threads to handle the data receiving.
    """

    def __init__(
        self,
        nixl_config: NixlConfigXpYd,
        config: LMCacheEngineConfig,
        backend: "NixlBackend",
    ):
        self.nixl_config = nixl_config
        self.role = nixl_config.role

        # Create sender or receiver based on role
        self._sender = None
        self._receiver = None

        self._backend = backend

        # Third Party
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        tp_rank = get_tensor_model_parallel_rank()

        if nixl_config.role == NixlRole.SENDER:
            self._sender = NixlSender(nixl_config, config, backend, tp_rank)
        else:
            self._receiver = NixlReceiver(nixl_config, config, backend, tp_rank)

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
        sender.prepare_send_async(keys, mem_objs, transfer_spec)

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
    agent: "NixlAgent"
    reg_descs: Any
    xfer_descs: Any
    xfer_handler: Any

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        tp_rank: int,
        backends: Optional[list[str]] = None,
        mem_type: str = "cuda"
    ):
        """
        Initialize the NIXL agent.

        Args:
            buffer_size (int): The size of the buffer.
            buffer_ptr (int): The pointer to the buffer.
            page_size (int): The page size of NIXL and
                the lmcache memory allocator.
            tp_rank (int): The tensor parallel rank.

        Returns:
            NixlWrapper: The NIXL agent.
            reg_dlist: the registered memory descriptor list.
            xfer_dlist: the local transfer descriptor list.
            prepped_xfer_handler: the prepped transfer handler.
        """
        try:
            # Third Party
            from nixl._api import nixl_agent as NixlAgent
            from nixl._api import nixl_agent_config
        except ImportError as err:
            raise RuntimeError("NIXL is not available") from err

        # Handle None backends by setting default to ["UCX"]
        # if backends is None:
        #     backends = ["UCX"]

        # Create a NIXL agent
        # nixl_agent = NixlAgent(
        #     str(uuid.uuid4()),
        #     nixl_agent_config(backends=backends),
        # )
        nixl_agent = NixlAgent(str(uuid.uuid4()))

        # Register the memory
        memory_desc = [(buffer_ptr, buffer_size, tp_rank, "")]
        # TODO(Jiayi): remove hardcode `mem_type`
        reg_descs = nixl_agent.get_reg_descs(memory_desc, mem_type=mem_type)
        nixl_agent.register_memory(reg_descs)

        logger.info(f"page size is {page_size / 1024 / 1024} MB")
        # Create xfer handlers
        xfer_desc = []
        for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
            xfer_desc.append((base_addr, page_size, tp_rank))

        xfer_descs = nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = nixl_agent.prep_xfer_dlist("", xfer_descs, mem_type=mem_type)

        self.agent = nixl_agent
        self.reg_descs = reg_descs
        self.xfer_descs = xfer_descs
        self.xfer_handler = xfer_handler

    def close(self, remote_xfer_handlers: Optional[dict[str, Any]] = None):
        self.agent.deregister_memory(self.reg_descs)

        self.agent.release_dlist_handle(self.xfer_handler)

        for remote_xfer_handler in self.agent._remote_xfer_handlers_dict.values():
            self.agent.release_dlist_handle(remote_xfer_handler)

        if remote_xfer_handlers is not None:
            for remote_xfer_handler in remote_xfer_handlers.values():
                self.agent.release_dlist_handle(remote_xfer_handler)
