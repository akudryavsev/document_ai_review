# document_ai_review

Локальные ответы по вашим документам. На сервер кладутся DOCX и PDF с текстовым слоем, список вопросов и три файла моделей. Веса считает видеокарта в отдельном контейнере, наружу запросы не уходят.

Команды `docker compose` набираются на хосте из корня репозитория. Каталоги `experiments/` и `models/` в git не входят: после копирования репозитория их нужно заполнить самим.

Краткий маршрут:

1. Поставить Docker Engine с Compose v2 и NVIDIA Container Toolkit.
2. Собрать контейнер приложения.
3. Положить документы в `experiments/documents/`.
4. Записать вопросы в `experiments/questions.json` и проверить файл.
5. Скачать GGUF в `models/` и вписать имена этих файлов в списки `config.yaml`.
6. Поднять контейнер с моделями и запустить прогон. Ответы лежат в `experiments/results/`.

## 1. Поставить Docker и доступ к видеокарте

На хосте нужны Docker Engine и плагин Compose v2: команда называется `docker compose`. Этой версии нужен и сервис с моделью, у него в compose-файле указано `gpus: all`.

Установка Engine: [docs.docker.com/engine/install](https://docs.docker.com/engine/install/). После неё пользователь, от которого вы запускаете команды, входит в группу `docker` (`sudo usermod -aG docker $USER`, затем новый вход в систему).

Драйвер NVIDIA стоит на хосте, не в образе. `nvidia-smi` на хосте должен показать карту до старта контейнеров.

Дальше — NVIDIA Container Toolkit, иначе Docker не отдаст GPU контейнеру с моделью: [install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). В конце настройки runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверка:

```bash
docker compose version
nvidia-smi
```

В контейнере одновременно загружена одна модель. На карту должна помещаться самая большая из трёх GGUF при окне контекста 4096.

## 2. Собрать контейнер приложения

Из корня репозитория:

```bash
docker compose up -d --build
docker compose exec app python main.py --help
```

Образ приложения — Python 3.11. Репозиторий смонтирован в `/app`. Клиента Hugging Face в этом образе нет: модели качаются на хосте.

Если оболочка уже открыта внутри контейнера `app` (Dev Containers), приставку `docker compose exec app` к командам `python` не добавляйте. Контейнер `inference` всё равно поднимается с хоста.

## 3. Положить документы

```bash
mkdir -p experiments/documents
```

Сюда кладутся файлы `.docx` и `.pdf`. Обход рекурсивный, вложенные папки тоже читаются. Другие расширения пропускаются.

Имя файла становится идентификатором документа, поэтому имена во всей папке должны быть уникальны. Ответ по файлу `dogovor.docx` потом будет в `experiments/results/dogovor.docx.jsonl`.

PDF нужен с текстовым слоем: скан без текста помечается `no_text_layer` и в ответы не идёт. OCR нет.

Пример конфига `config.yaml` уже указывает на `experiments/documents`. Путь менять не обязательно.

## 4. Записать и проверить вопросы

Файл по умолчанию — `experiments/questions.json`. Пока в списке фаз есть поиск или ответ, прогон читает его до старта и останавливается на пустом списке, дубликате `id`, дубликате `category` и невалидном JSON.

В `category` пишите сам вопрос: его получит модель, и по нему будет формулироваться ответ.

В `synonyms` пишите, как этот факт может выглядеть в документе: абзац или две-три фразы в тех словах, в каких пункт встречается в договоре, политике или отчёте. Несколько синонимов — несколько таких заготовок, если в разных файлах одна и та же мысль записана по-разному. Поиск идёт по каждой заготовке отдельно и берёт лучшее совпадение. Модель при этом отвечает на один текст — `category`.

```json
{
  "version": "1.0",
  "questions": [
    {
      "id": "Q001",
      "category": "Какой срок хранения персональных данных?",
      "synonyms": [
        "Персональные данные хранятся в течение пяти лет с момента достижения цели обработки, после чего подлежат уничтожению.",
        "Срок хранения сведений о субъекте — 5 (пять) лет. По истечении срока носители уничтожаются по акту."
      ],
      "expected_answerable": true
    }
  ]
}
```

`id` и `category` уникальны внутри файла. `expected_answerable` прогон для отбора вопросов не использует, поле можно оставить `true`.

Контейнер `app` из шага 2 уже должен быть запущен. Проверка:

```bash
docker compose exec app python -c "from ports.file_sources import FileQuestionSource; loaded = FileQuestionSource('experiments/questions.json').load(); print('вопросов:', len(loaded.questions))"
```

Строка `вопросов: N` значит, что схема принята.

## 5. Скачать модели

Нужны файлы GGUF трёх ролей. В одном прогоне их может быть несколько: прогон идёт по очереди, в каждый момент загружена одна модель.

| Роль | Список в `config.yaml` | Что это за модель |
|---|---|---|
| Эмбеддинги, индексация | `embedding.models` | Модель эмбеддингов. В секции ini будет `embedding = true` |
| Реранк, поиск | `reranker.models` | Модель, которую llama-server считает реранкером. В секции: `embedding = true`, `reranking = true`, `pooling = rank` |
| Ответ | `answer.models` | Обычная чат-модель. Флагов эмбеддинга в секции нет |

Имя в списке — имя файла в `models/`, с суффиксом `.gguf`. Одно и то же имя нельзя поставить дважды и нельзя отдать двум ролям. `python main.py run` до старта фаз проверяет все три списка и останавливается, если какого-то файла нет. Это относится и к `--phase A1`: индексация считает эмбеддинги кусков файлом из `embedding.models`, а файлы реранкера и ответа к этому моменту уже лежат в `models/`, потому что проверка и `models.ini` собираются по всему конфигу.

Пример в репозитории — по одному файлу:

```yaml
embedding:
  models:
    - embed.gguf
reranker:
  models:
    - rerank.gguf
answer:
  models:
    - answer.gguf
```

Несколько моделей — несколько строк в списке. Прогон перебирает эмбеддеры, для каждого — реранкеры, для каждой пары — модели ответа. В строке ответа остаются имена этой тройки: `embedding_model`, `reranker_model`, `answer_model`.

Проект конкретный репозиторий на Hugging Face не фиксирует. Качается на хосте, пакет `huggingface_hub`:

```bash
pip install -U huggingface_hub
mkdir -p models
```

Если команда `huggingface-cli` не находится, тот же вызов делает `hf download`.

### Имена из примера

`config.yaml` и `deploy/llama/models.ini` уже согласованы на `embed.gguf`, `rerank.gguf` и `answer.gguf`. Каталог `/models` внутри контейнера — это `models/` на хосте.

```bash
huggingface-cli download ORG/EMBED-GGUF embed-file.gguf --local-dir /tmp/gguf
cp /tmp/gguf/embed-file.gguf models/embed.gguf

huggingface-cli download ORG/RERANK-GGUF rerank-file.gguf --local-dir /tmp/gguf
cp /tmp/gguf/rerank-file.gguf models/rerank.gguf

huggingface-cli download ORG/ANSWER-GGUF answer-file.gguf --local-dir /tmp/gguf
cp /tmp/gguf/answer-file.gguf models/answer.gguf
```

### Оставить понятные имена

Файл кладётся в `models/` под своим именем, и это имя пишется в список роли:

```bash
cp /path/to/bge-m3-q8.gguf models/bge-m3-q8.gguf
cp /path/to/bge-reranker.gguf models/bge-reranker.gguf
cp /path/to/qwen-reranker.gguf models/qwen-reranker.gguf
cp /path/to/Qwen3-8B-Q4_K_M.gguf models/Qwen3-8B-Q4_K_M.gguf
```

```yaml
embedding:
  models:
    - bge-m3-q8.gguf
reranker:
  models:
    - bge-reranker.gguf
    - qwen-reranker.gguf
answer:
  models:
    - Qwen3-8B-Q4_K_M.gguf
```

Ini руками не правят. Команда перезаписывает `deploy/llama/models.ini` по спискам: заголовок секции и поле `model` равны имени файла, путь внутри контейнера — `/models/<имя>`.

```bash
docker compose exec app python main.py write-models-ini --config config.yaml
docker compose --profile poc up -d inference
```

`run` сравнивает ini со списками и останавливается, если текст другой. После правки списков снова `write-models-ini` и пересоздание `inference`. Окно контекста задаёт `ctx-size` в секции `[*]`, сейчас `4096`.

## 6. Запустить прогон

Файлы GGUF уже лежат в `models/`. Поднять приложение и сервер моделей:

```bash
docker compose --profile poc up -d
```

Образ сервера — `ghcr.io/ggml-org/llama.cpp:server-cuda`, он скачивается при первом запуске и занимает заметное место. В `docker compose ps` сервис `inference` должен быть запущен. Журнал: `docker compose logs inference`. Из контейнера `app` адрес сервера уже прописан в `config.yaml`: `http://inference:8080`.

Полный прогон — индексация, поиск и ответы:

```bash
docker compose exec app python main.py run --config config.yaml
```

Только индексация, с пересчётом уже готовых кусков:

```bash
docker compose exec app python main.py run --config config.yaml --phase A1 --force-phase A1
```

Для имён из `config.yaml` до этой команды в `models/` уже лежат `embed.gguf`, `rerank.gguf` и `answer.gguf`. A1 загружает `embed.gguf` и пишет векторы кусков в `experiments/emb/embed.gguf/<имя документа>/`.

Список фаз задаёт `--phase`. Без флага выполняются `A1,A2,B`. Поля `phases.run_a1`, `run_a2` и `run_b` в `config.yaml` на список не влияют. Файл вопросов читается, когда в списке есть A2 или B.

В конце процесса печатается строка `run complete documents=… questions=… error_rate=…`.

Ответы:

- `experiments/results/<имя файла>.jsonl` — строки пар «документ × вопрос». Одна тройка моделей даёт свою строку; в ней `embedding_model`, `reranker_model` и `answer_model`. Текст ответа в поле `answer`, признак в `found`, страницы и цитата в `citations` (`page_from`, `page_to`, `section_path`, `quote`).
- `experiments/results/run_summary.json` — сколько документов и вопросов было в прогоне, счётчики фаз `computed`, `skipped`, `failed` и список `combinations` по тройкам.

Повторный расчёт той же тройки и того же вопроса заменяет прежнюю строку. Строки других троек остаются. Сводка каждый раз пишется заново.

На один запуск можно оставить одну модель роли флагом `--embedding-model`, `--reranker-model` или `--answer-model`. Имя — файл из `models/`. Остальные списки берутся из конфига.

Если поиск ничего не нашёл, модель не вызывается, а в `answer` будет фраза «В поиске по этому вопросу ничего не найдено.»

`failed` у фазы A1 в сводке обычно значит, что файл не разобрался: чаще всего это PDF без текстового слоя.

Первый прогон считает эмбеддинги всех кусков и может идти долго. Следующий запуск с теми же файлами фазы пропускает: в сводке это `skipped`.

## 7. Повторить прогон после правок

Новый файл с новым именем подхватывается следующим запуском сам.

Повторное использование артефактов смотрит на настройки, а не на байты документов, вопросов и GGUF. После замены содержимого при том же имени файла, после правки `questions.json` и после смены файла эмбеддера нужен пересчёт всех трёх фаз:

```bash
docker compose exec app python main.py run --config config.yaml \
  --force-phase A1 --force-phase A2 --force-phase B
```

После смены только реранкера достаточно `--force-phase A2 --force-phase B`. После смены только модели ответа — `--force-phase B`.

Флаги можно повторять. Один `--force-phase A1` обновляет индекс и не переписывает уже сохранённые ответы.

## Ограничения этого запуска

Точка входа — команда `python main.py run`. Верификатор в прогон не входит. `python main.py eval` группирует уже записанные ответы по тройке моделей и печатает JSON-массив отчётов. Формулы Recall, nDCG и exact match в этот отчёт ещё не входят.

## Тесты

Из корня на хосте, при запущенном контейнере `app`:

```bash
docker compose exec app python -m pytest tests
```

Внутри контейнера: `python -m pytest tests`.
