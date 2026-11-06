"""Аренда единственного загруженного шлюза."""

from __future__ import annotations

from models.observe import TraceEvent
from ports.gateways import ModelGateway
from ports.trace import TraceSink


class GpuLease:
    """Гарантирует не больше одного загруженного шлюза.

    Attributes:
        trace_sink: Приёмник событий смены аренды.
        current: Текущий загруженный шлюз или None.
        acquire_order: Журнал имён acquire для тестов.
        release_count: Число release.
    """

    def __init__(self, trace_sink: TraceSink) -> None:
        """Сохраняет TraceSink.

        Args:
            trace_sink: Приёмник событий.
        """
        self.trace_sink = trace_sink
        self.current: ModelGateway | None = None
        self.acquire_order: list[str] = []
        self.release_count = 0

    def acquire(self, gateway: ModelGateway) -> None:
        """Выгружает предыдущий шлюз и загружает новый.

        Args:
            gateway: Шлюз для загрузки.
        """
        if self.current is gateway and gateway.loaded:
            return
        if self.current is not None and self.current is not gateway:
            previous_name = self.current.name
            self.current.unload()
            self.trace_sink.emit(
                TraceEvent(
                    stage="gpu_lease",
                    status="ok",
                    counters={"unloaded": 1},
                    debug_payload=previous_name,
                )
            )
            self.current = None
        gateway.load()
        self.current = gateway
        self.acquire_order.append(gateway.name)
        self.trace_sink.emit(
            TraceEvent(
                stage="gpu_lease",
                status="ok",
                counters={"loaded": 1},
                debug_payload=gateway.name,
            )
        )

    def release(self) -> None:
        """Выгружает текущий шлюз, если есть."""
        self.release_count += 1
        if self.current is None:
            return
        name = self.current.name
        self.current.unload()
        self.current = None
        self.trace_sink.emit(
            TraceEvent(
                stage="gpu_lease",
                status="ok",
                counters={"released": 1},
                debug_payload=name,
            )
        )
