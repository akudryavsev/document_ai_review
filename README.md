# document_ai_review

Локальный разбор документов по фазам A1 → A2 → B. Извлечение DOCX/PDF, очистка, чанкинг, гибридный поиск, сборка контекста и ответ без верификатора реализованы. Артефакты — JSONL + `vectors.f16`, без OCR и без parquet.

Веса считает отдельный контейнер `inference` (`llama-server`, одна GGUF на GPU). Процесс приложения в него ходит по HTTP и сам модели не грузит. `eval` и очередь Redis ещё не реализованы: `python main.py eval` завершается с ошибкой.

Код лежит в `src`. Импорт совпадает с именем папки (`import phases` читает `src/phases`). Команды `docker compose` набираются на хосте из корня репозитория.

## Контейнер приложения

Образ — Python 3.11. Зависимости: `pydantic`, `pyyaml`, `pytest`, `python-docx`, `pypdf`, `pymorphy3`. Репозиторий монтируется в `/app`, `PYTHONPATH=/app/src`.

```bash
docker compose up -d --build
docker compose exec app python main.py --help
```

Если оболочка уже открыта в контейнере (Dev Containers: Reopen in Container), Docker не нужен:

```bash
python main.py --help
```

## CLI

В корне лежит пример `config.yaml`: пути прогона в `./experiments/` (в git не входит), `extraction.debug_dump=true`. Разбор только на чанки — это команда с `--phase A1`, а не флаги `phases.run_*` в YAML.

- `experiments/documents/` — `.docx` / `.pdf`
- `experiments/debug/` — дампы extract/clean
- `experiments/emb/<doc_id>/chunks.jsonl` — чанки A1
- `experiments/results/` — сводка и ответы
- `experiments/index/`, `experiments/contexts/` — артефакты A2
- `experiments/questions.json` — для прогонов с A2/B

Список фаз задаёт `--phase`. Без флага это `A1,A2,B`. Поля `phases.run_a1` / `run_a2` / `run_b` в YAML на этот список не влияют.

`--embedding-model`, `--reranker-model` и `--answer-model` — имена секций в `deploy/llama/models.ini` (`embed`, `rerank`, `answer`), не идентификаторы Hugging Face. Другое имя допустимо, только если такая секция есть в ini.

```bash
mkdir -p experiments/documents
python main.py run --config config.yaml --phase A1
python main.py run --config config.yaml --phase A1,A2
python main.py run --config config.yaml
python main.py run --config config.yaml --phase B --force-phase B --answer-model answer
python main.py eval --config config.yaml --ground-truth ./gt.json --predictions ./experiments/results
```

Для `--phase A1` файл вопросов не читается. Как только в списке фаз есть A2 или B, `questions.json` обязателен и проверяется до старта фаз. Полный прогон и фазы A2/B требуют запущенный контейнер `inference` и три файла GGUF, см. ниже.

Без подкоманды `main.py` печатает справку и завершается с ненулевым кодом. Команда `eval` пока возвращает ошибку: `EvaluationRun` отложен до реализации спецификации метрик. `Application` извлекает текст реальными `FormatAwareExtractor`, `Cleaner` и `Chunker`, ищет реальным `HybridSearcher`, контекст собирает реальным `ContextAssembler`, ответ пишет реальный `AnswerGenerator`, артефакты пишет в `FileArtifactStore`. Эмбеддер, реранкер и LLM — `LlamaEmbeddingGateway`, `LlamaRerankerGateway` и `LlamaLlmGateway`: они ходят по HTTP в контейнер `inference` (`llama-server`). Фаза A1 пишет `vectors.f16` из байтов float16. Верификатор в граф не входит. Пустой поиск не вызывает LLM: в ответе фраза «В поиске по этому вопросу ничего не найдено.» При `retrieval.debug_dump=true` фаза A2 дополнительно пишет `{cache_dir}/debug/{doc_id}/{question_id}.retrieval.json` и `{cache_dir}/debug/{doc_id}/{question_id}.context.md`. Имена моделей в конфиге — секции `embed`, `rerank`, `answer` из `deploy/llama/models.ini`. PDF без текстового слоя помечается `no_text_layer`; OCR в этой итерации нет.

## Тесты

Каталог `tests/` покрывает раздел 10 спецификации верхних классов и каталоги извлечения, очистки, чанкинга, хранилища, гибридного поиска, сборки контекста, ответа и шлюзов llama-server (`T-EXT`, `T-CLN`, `T-CHK`, `T-STO`, `T-RET`, `T-RNK`, `T-CTX`, `T-GEN`, `T-GW`).

Из корня репозитория на хосте:

```bash
docker compose exec app python -m pytest tests
```

Внутри контейнера:

```bash
python -m pytest tests
```

## Модели llama.cpp

Каталог `models/` на хосте монтируется в контейнер как `/models`. Файлы в git не входят. Приложение путь к GGUF не открывает: его читает только `llama-server` по `deploy/llama/models.ini`.

| Секция ini и поле `model` в `config.yaml` | Файл по умолчанию | Назначение |
|---|---|---|
| `embed` | `/models/embed.gguf` | Эмбеддинги, фаза A1. В секции `embedding = true` |
| `rerank` | `/models/rerank.gguf` | Реранк, фаза A2. В секции `embedding = true`, `reranking = true`, `pooling = rank` |
| `answer` | `/models/answer.gguf` | Ответ, фаза B. Чат-модель, флагов эмбеддинга нет |

Имя секции менять не нужно. В `config.yaml` остаются `embed`, `rerank`, `answer`. Какой именно GGUF лежит по пути секции, выбирает оператор: одна модель должна помещаться на GPU контейнера. Окно контекста всех пресетов — `ctx-size = 4096` в секции `[*]`.

На хосте для боевого прогона нужны драйвер NVIDIA и NVIDIA Container Toolkit. Образ сервера — `ghcr.io/ggml-org/llama.cpp:server-cuda`, в контейнере одновременно не больше одной модели (`--models-max 1`).

Без профиля стартует только `app`. Профиль `poc` поднимает ещё и `inference`. Из контейнера `app` адрес уже прописан: `http://inference:8080`.

```bash
docker compose --profile poc up -d
```

Замена байтов GGUF при том же имени секции сама по себе фазу не пересчитывает. После смены файла эмбеддера нужен `--force-phase A1`, после смены реранкера — `--force-phase A2`, после смены модели ответа — `--force-phase B`.

### Скачать GGUF

Качать на хосте, в каталог `models/`. В образе `app` клиента Hugging Face нет. Ниже `huggingface-cli` из пакета `huggingface_hub` (`pip install -U huggingface_hub` на хосте). Репозиторий и имя файла — те, которые вы кладёте на эту машину; проект конкретную карточку не фиксирует.

Скачать один файл и сразу положить под именем, которое уже ждёт ini:

```bash
mkdir -p models
huggingface-cli download ORG/EMBED-GGUF embed-file.gguf --local-dir /tmp/gguf
cp /tmp/gguf/embed-file.gguf models/embed.gguf

huggingface-cli download ORG/RERANK-GGUF rerank-file.gguf --local-dir /tmp/gguf
cp /tmp/gguf/rerank-file.gguf models/rerank.gguf

huggingface-cli download ORG/ANSWER-GGUF answer-file.gguf --local-dir /tmp/gguf
cp /tmp/gguf/answer-file.gguf models/answer.gguf
```

Так ini править не надо. На место `embed` годится только GGUF эмбеддера, на `rerank` — модель, которую llama-server умеет скорить как reranker, на `answer` — чат.

### Уже скачанные файлы

Два равноправных способа.

Оставить ini как есть и назвать файлы так, как в нём написано. Путь в ini — путь внутри контейнера, поэтому на хосте это каталог `models/`:

```bash
cp /path/to/already/my-embed.gguf models/embed.gguf
cp /path/to/already/my-reranker.gguf models/rerank.gguf
cp /path/to/already/my-answer.gguf models/answer.gguf
```

Либо сохранить исходные имена и поправить только строку `model` в нужной секции `deploy/llama/models.ini`. Секцию не переименовывать и это имя не писать в `config.yaml`.

```bash
cp /path/to/already/Qwen3-8B-Q4_K_M.gguf models/Qwen3-8B-Q4_K_M.gguf
```

```ini
[answer]
model = /models/Qwen3-8B-Q4_K_M.gguf
```

То же для `[embed]` и `[rerank]`: слева секция `embed` или `rerank`, справа `/models/` и имя файла, как оно лежит на хосте в `models/`. Флаги `embedding`, `reranking` и `pooling` у этих секций не снимать. После правки ini контейнер `inference` нужно пересоздать, чтобы он перечитал пресет: `docker compose --profile poc up -d inference`.
