"""Подделки портов для тестов без GPU."""

from fakes.artifacts import InMemoryArtifactStore
from fakes.gateways import FakeEmbeddingGateway, FakeLlmGateway, FakeRerankerGateway
from fakes.pipeline import (
    FakeAnswerGenerator,
    FakeAnswerVerifier,
    FakeChunker,
    FakeCleaner,
    FakeContextAssembler,
    FakeExtractor,
    FakeHybridRetriever,
    RecordingResultSink,
)
from fakes.trace import RecordingTraceSink

__all__ = [
    "FakeAnswerGenerator",
    "FakeAnswerVerifier",
    "FakeChunker",
    "FakeCleaner",
    "FakeContextAssembler",
    "FakeEmbeddingGateway",
    "FakeExtractor",
    "FakeHybridRetriever",
    "FakeLlmGateway",
    "FakeRerankerGateway",
    "InMemoryArtifactStore",
    "RecordingResultSink",
    "RecordingTraceSink",
]
