"""HTTP-клиент llama-server на стандартной библиотеке."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class LlamaServerError(Exception):
    """Сбой загрузки, выгрузки или infer у llama-server.

    Attributes:
        status_code: HTTP-код, если ответ сервера был. Иначе None.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Сохраняет текст и код.

        Args:
            message: Текст для фазы и лога.
            status_code: Код HTTP или None при обрыве и таймауте.
        """
        super().__init__(message)
        self.status_code = status_code


class LlamaHttpClient:
    """Отправляет JSON в llama-server и возвращает объект ответа.

    Attributes:
        base_url: Адрес без завершающего слэша.
        timeout_sec: Таймаут одного сокета.
        poll_interval_sec: Пауза между опросами готовности пресета.
    """

    def __init__(
        self,
        base_url: str,
        timeout_sec: float,
        poll_interval_sec: float,
    ) -> None:
        """Сохраняет адрес и таймауты.

        Args:
            base_url: Базовый URL без завершающего слэша.
            timeout_sec: Таймаут одного вызова.
            poll_interval_sec: Пауза опроса GET /models.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec

    def request(self, method: str, path: str, payload: dict | None) -> dict:
        """Выполняет один HTTP-вызов и разбирает JSON-объект.

        Args:
            method: GET или POST.
            path: Путь, начиная со слэша.
            payload: Тело JSON или None для GET.

        Returns:
            Разобранный объект ответа при HTTP 200.

        Raises:
            LlamaServerError: Сеть, таймаут или код не 200.
        """
        url = f"{self.base_url}{path}"
        data: bytes | None = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        http_request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_sec) as response:
                raw = response.read()
        except TimeoutError as error:
            raise LlamaServerError("таймаут") from error
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise self.error_for_status(error.code, body) from error
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", None)
            if isinstance(reason, TimeoutError):
                raise LlamaServerError("таймаут") from error
            raise LlamaServerError("обрыв соединения") from error
        return self._object_from_bytes(raw)

    def wait(self, seconds: float) -> None:
        """Ждёт перед следующим опросом готовности.

        Args:
            seconds: Пауза в секундах. Ноль и меньше не ждут.
        """
        if seconds > 0:
            time.sleep(seconds)

    def error_for_status(self, status: int, body: str) -> LlamaServerError:
        """Собирает ошибку по коду и усечённому телу.

        Args:
            status: HTTP-код.
            body: Тело ответа.

        Returns:
            LlamaServerError с кодом. Неизвестная модель помечена фразой
            «нетот модели».
        """
        clipped = body[:500]
        message = f"HTTP {status}: {clipped}"
        if status == 404 and "model" in body.lower():
            message = f"нетот модели {message}"
        return LlamaServerError(message, status_code=status)

    def _object_from_bytes(self, raw: bytes) -> dict:
        """Разбирает тело 200 как JSON-объект.

        Args:
            raw: Байты ответа.

        Returns:
            Словарь JSON.

        Raises:
            LlamaServerError: Тело не объект.
        """
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise LlamaServerError("ответ llama-server не JSON") from error
        if not isinstance(parsed, dict):
            raise LlamaServerError("ответ llama-server не объект")
        return parsed
