"""Перевод чисел в little-endian float16 и обратно."""

from __future__ import annotations

import math
import struct

from gateways.client import LlamaServerError


class Float16Codec:
    """Кодирует конечные числа в IEEE 754 binary16.

    Attributes:
        max_finite: Наибольшее конечное значение float16.
    """

    max_finite = 65504.0

    def encode(self, values: list[float]) -> bytes:
        """Переводит список чисел в little-endian float16.

        Args:
            values: Компоненты одного вектора.

        Returns:
            Байты длины dim * 2.

        Raises:
            LlamaServerError: Пустой вектор, нечисло или выход за float16.
        """
        if not values:
            raise LlamaServerError("пустой вектор")
        chunks: list[bytes] = []
        for value in values:
            chunks.append(self._encode_one(value))
        return b"".join(chunks)

    def decode(self, payload: bytes) -> list[float]:
        """Читает little-endian float16 в обычные числа.

        Args:
            payload: Байты одного вектора, длина чётная.

        Returns:
            Список компонент.

        Raises:
            ValueError: Длина не кратна двум.
        """
        if len(payload) % 2 != 0:
            raise ValueError("длина float16 не кратна 2")
        count = len(payload) // 2
        values: list[float] = []
        for index in range(count):
            start = index * 2
            number = struct.unpack_from("<e", payload, start)[0]
            values.append(float(number))
        return values

    def _encode_one(self, value: float) -> bytes:
        """Кодирует одно число с округлением struct к ближайшему чётному.

        Args:
            value: Компонента.

        Returns:
            Два байта little-endian.

        Raises:
            LlamaServerError: Нечисло, inf, nan или модуль больше 65504.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LlamaServerError("нечисловое значение")
        number = float(value)
        if not math.isfinite(number) or abs(number) > self.max_finite:
            raise LlamaServerError("число вне float16")
        return struct.pack("<e", number)
