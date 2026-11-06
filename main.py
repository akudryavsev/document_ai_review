"""Запуск приложения из корня репозитория."""

from __future__ import annotations

import sys

from entry import CliApp


class Main:
    """Стартует процесс и возвращает код выхода.

    Attributes:
        argv: Аргументы командной строки без имени интерпретатора.
    """

    def __init__(self, argv: list[str]) -> None:
        """Сохраняет аргументы запуска.

        Args:
            argv: Аргументы командной строки без имени интерпретатора.
        """
        self.argv = argv

    def run(self) -> int:
        """Делегирует разбор argv и прогон в CliApp.

        Returns:
            Код выхода процесса.
        """
        return CliApp().main(self.argv)


if __name__ == "__main__":
    raise SystemExit(Main(sys.argv[1:]).run())
