# dnd-skill-router

`dnd-skill-router` - Локальный OpenAI-compatible proxy для Obsidian Copilot и LM Studio. Proxy принимает chat-запросы, выбирает один DnD-oriented skill, обогащает prompt правилами проекта и отправляет итоговый запрос в локальную модель LM Studio

Проект полезен, если вы ведете или готовите DnD-кампанию в Obsidian и хотите, чтобы локальная модель отвечала не "вообще как чат-бот", а в одном из понятных рабочих режимов: сцена, анализ, шаблон, лор или правила.

## Кому подойдет

Подойдет, если вы собираетесь:

- подключить Obsidian плагин Copilot к локальным моделям через OpenAI-compatible API;
- автоматически выбирать стиль ответа под текущую задачу мастера;
- хранить skill-ы в отдельных markdown-файлах и менять их без правки кода;
- использовать LM Studio как backend для генерации;
- сохранить приватность заметок: запросы идут в локальный proxy и локальный LM Studio server.

Проект не поддерживает следующие функции:

- **RAG по всему Obsidian vault**: Не реализовано в proxy. Может быть добавлено, например, через Obsidian плагин Copilot (плагин поддерживает эмбеддинг модели для поиска по заметкам) или LM Studio через MCP.
- **Совмещение скиллов**: На один запрос выбирается только один skill

## Как это работает

Основной поток:

```text
Obsidian Copilot или другой OpenAI-compatible клиент
  -> FastAPI proxy dnd-skill-router
  -> LangGraph pipeline выбора skill
  -> LM Studio /v1/chat/completions или /v1/responses
  -> ответ отправляется обратно клиенту
```

На каждый запрос router смотрит на последнее `user`-сообщение и выбирает один skill:

- `story` - сцены, NPC, диалоги, сюжетные ходы, описания.
- `analysis` - противоречия, логические дыры, проверка согласованности.
- `template` - структуры, таблицы, повторно используемые форматы.
- `lore` - мир, фракции, культуры, история, география.
- `rules` - правила DnD, механики, проверки, rulings, статблоки.

Если в начале последнего сообщения есть manual command, например `!story`, LLM-router не вызывается: skill выбирается напрямую. Если команды нет, proxy вызывает router model в LM Studio, получает ранжированный список skills и сравнивает уверенность с порогом из `config.yaml`.

Если уверенность ниже порога, main model не вызывается. Вместо этого клиент получает уточняющий ответ с предложением выбрать skill вручную.

## Быстрый старт

Требования:

- Python 3.11 или новее;
- запущенный LM Studio server с OpenAI-compatible endpoint;
- одна или две загруженные модели в LM Studio: router model для классификации и main model для ответа.

Установка:

```bash
cd dnd-skill-router
pip install -e .
```

Создайте локальный `.env`, если хотите хранить рядом пример локальных переменных окружения:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Проверьте `config.yaml` и укажите реальные имена моделей из LM Studio:

```yaml
lm_studio:
  base_url: "http://127.0.0.1:1234/v1"
  api_key: "lm-studio"
  router_model: "qwen/qwen3.5-9b"
  main_model: "your-main-model-name"
```

Запуск proxy:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Проверка:

```text
http://127.0.0.1:8000/health
```

OpenAI-compatible base URL:

```text
http://127.0.0.1:8000/v1
```

## Настройка Obsidian Copilot

В Obsidian Copilot выберите OpenAI-compatible provider и укажите:

```text
Base URL: http://127.0.0.1:8000/v1
Model: dnd-skill-router
API key: any non-empty value
```

Proxy умеет работать с:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/responses
```

## Примеры запросов

Автоматический выбор skill:

```text
Придумай сцену в таверне, где NPC пытается скрыть важную улику.
```

Ручной выбор skill:

```text
!analysis Проверь эту сцену на противоречия с прошлой сессией.
!template Сделай шаблон заметки для фракции в Obsidian.
!lore Опиши портовое королевство и его конфликт с магократией.
!rules Как обработать погоню по крышам в DnD?
```

Manual commands работают только в начале последнего пользовательского сообщения:

```text
!story Придумай тревожное описание заброшенного храма.
```

Поддерживаемые команды:

```text
!story
!analysis
!template
!lore
!rules
```

## Где лежат инструкции skills

Skills хранятся как markdown-файлы в директории `skills/`:

```text
skills/story.md
skills/analysis.md
skills/template.md
skills/lore.md
skills/rules.md
```

Общие инструкции:

```text
skills/_shared/answer_rules.md
skills/_shared/consistency_lens.md
```

При генерации proxy добавляет system messages в таком порядке:

1. базовый системный prompt проекта;
2. общие правила ответа из `answer_rules.md`;
3. prompt выбранного skill;
4. `consistency_lens.md` для skills, которым нужна проверка согласованности;
5. оригинальную историю сообщений клиента.

История сообщений не удаляется. Текущая задача определяется последним `user`-сообщением.

## Wiki: `.env`

В проекте есть `.env.example`:

```env
APP_ENV=local
CONFIG_PATH=./config.yaml
```

Приложение читает переменные из окружения через `os.getenv`. Сам файл `.env` сейчас не загружается автоматически кодом приложения, хотя зависимость `python-dotenv` установлена. Поэтому значения из `.env` начнут влиять на запуск только если вы экспортировали их в shell или запускаете приложение через инструмент, который сам подгружает `.env`.

`CONFIG_PATH` - путь к YAML-конфигу. Это единственная переменная окружения, которая сейчас напрямую влияет на запуск приложения. Если переменная не задана, используется `./config.yaml`.

Примеры:

```env
CONFIG_PATH=./config.yaml
CONFIG_PATH=./config.local.yaml
CONFIG_PATH=D:/Configs/dnd-skill-router.yaml
```

PowerShell-пример без `.env`:

```powershell
$env:CONFIG_PATH="./config.local.yaml"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`APP_ENV` - служебная переменная для обозначения окружения. В текущем коде она не используется для ветвления логики, но может быть полезна для локальных скриптов или будущих настроек.

Файл `.env` не должен попадать в git. Он уже добавлен в `.gitignore`.

## Wiki: `config.yaml`

`config.yaml` - главный файл настройки проекта. Схема строгая: неизвестные поля считаются ошибкой, поэтому лучше добавлять параметры только после поддержки в коде.

### `server`

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  enable_cors: true
```

- `host` - адрес, на котором должен слушать proxy.
- `port` - порт proxy.
- `enable_cors` - включает CORS `*`. Полезно для клиентов, которые обращаются к API из браузерного окружения.

Важно: при запуске через `uvicorn --host ... --port ...` фактический bind задает команда запуска. Значения `server.host` и `server.port` используются приложением для логов и как проектная настройка.

### `lm_studio`

```yaml
lm_studio:
  base_url: "http://127.0.0.1:1234/v1"
  api_key: "lm-studio"
  router_model: "qwen/qwen3.5-9b"
  main_model: "your-main-model-name"
  request_timeout_seconds: 600
```

- `base_url` - OpenAI-compatible URL LM Studio.
- `api_key` - ключ для OpenAI SDK/httpx. LM Studio обычно принимает любое непустое значение.
- `router_model` - модель, которая классифицирует последнее сообщение по skills.
- `main_model` - модель, которая пишет итоговый ответ пользователю.
- `request_timeout_seconds` - timeout обычных запросов к LM Studio. Timeout не ретраится автоматически, чтобы не ставить повторные тяжелые генерации в очередь.

### `routing`

```yaml
routing:
  confidence_threshold: 80
  max_ranked_skills: 3
  use_manual_commands: true
```

- `confidence_threshold` - минимальная уверенность router model от 0 до 100. Ниже порога proxy просит уточнить skill.
- `max_ranked_skills` - сколько вариантов skills сохранять из ответа router model.
- `use_manual_commands` - проектная настройка для manual commands. В текущем pipeline команды поддерживаются через отдельный узел маршрутизации.

### `skills`

```yaml
skills:
  directory: "./skills"
  default_skill: "story"
  shared_answer_rules: "./skills/_shared/answer_rules.md"
  consistency_lens: "./skills/_shared/consistency_lens.md"
```

- `directory` - папка с markdown-файлами skills.
- `default_skill` - запасной skill в настройках. В текущем pipeline выбор обычно идет через manual command, router model или clarification.
- `shared_answer_rules` - общие правила ответа, добавляются к каждому skill.
- `consistency_lens` - дополнительная инструкция для проверки согласованности. Сейчас добавляется для `story`, `lore`, `template` и `analysis`.

### `generation`

```yaml
generation:
  router_temperature: 0.0
  router_max_tokens: 600
  main_temperature: 0.1
  main_max_tokens: 50000
  stream: false
```

- `router_temperature` - температура router model. Обычно лучше держать `0.0`, чтобы классификация была стабильной.
- `router_max_tokens` - лимит ответа router model.
- `main_temperature` - температура main model.
- `main_max_tokens` - лимит ответа main model для Chat Completions.
- `stream` - проектное значение по умолчанию. Фактический streaming для запроса берется из payload клиента и дополнительно ограничивается `streaming.enabled`.

Если клиент передает безопасные параметры генерации (`temperature`, `max_tokens`, `top_p`, `stop` и другие поддерживаемые поля), proxy передает их в LM Studio для main model.

### `logging`

```yaml
logging:
  level: "INFO"
  log_file: "./logs/router.log"
  debug_full_payload: false
```

- `level` - уровень логирования.
- `log_file` - путь к файлу логов. Директория создается автоматически.
- `debug_full_payload` - если `true`, в routing log попадает полный исходный payload. По умолчанию выключено, чтобы не логировать весь контекст Obsidian.

### `streaming`

```yaml
streaming:
  enabled: true
  mode: "real"
  fallback_to_fake_streaming: false
  lm_studio_timeout_seconds: 600
  send_done_on_disconnect: true
```

- `enabled` - разрешает streaming-ответы proxy, если клиент запросил `stream: true`.
- `mode` - режим streaming. Сейчас основной рабочий режим - реальный passthrough stream от LM Studio.
- `fallback_to_fake_streaming` - проектная настройка для fallback-поведения.
- `lm_studio_timeout_seconds` - timeout streaming-запросов к LM Studio.
- `send_done_on_disconnect` - настройка поведения при разрывах stream.

### `responses_api`

```yaml
responses_api:
  enabled: true
  proxy_to_lm_studio_responses: true
  support_streaming: true
  support_previous_response_id_passthrough: true
  store_previous_responses: false
  unsupported_tools_policy: "ignore"
```

- `enabled` - включает endpoint `POST /v1/responses`.
- `proxy_to_lm_studio_responses` - указывает, что Responses API проксируется в LM Studio `/responses`.
- `support_streaming` - разрешает streaming для Responses API.
- `support_previous_response_id_passthrough` - позволяет передавать `previous_response_id` дальше в LM Studio.
- `store_previous_responses` - хранение предыдущих responses. В MVP состояние не сохраняется.
- `unsupported_tools_policy` - что делать с неподдерживаемыми полями tools: `ignore` или `reject`.

#### `responses_api.reasoning`

```yaml
reasoning:
  mode: "think_block"
  stream_insertion_strategy: "transform_reasoning_events"
  preserve_usage: true
  strip_reasoning_from_completed: true
  log_presence: true
  log_raw_reasoning: false
```

- `mode` - как обрабатывать reasoning events LM Studio:
  - `drop` - не показывать reasoning пользователю;
  - `think_block` - показывать reasoning как текст внутри `<think>...</think>`;
  - `plain` - показывать reasoning как обычный текст без тегов;
  - `pass_through` - пропускать raw reasoning events как есть, удобно для отладки.
- `stream_insertion_strategy` - текущая стратегия трансформации reasoning stream.
- `preserve_usage` - сохранять usage, включая reasoning tokens, если они пришли от LM Studio.
- `strip_reasoning_from_completed` - удалять reasoning из финального completed-события.
- `log_presence` - логировать факт наличия reasoning.
- `log_raw_reasoning` - логировать сырой reasoning. Включайте осторожно: это может записывать чувствительный текст.

#### `responses_api.diagnostics`

```yaml
diagnostics:
  enabled: false
  placement: "end"
  format: "visible_block"
  include_source_api: true
  include_reasoning_mode: true
  include_streaming_strategy: true
  include_selected_skill: true
  include_confidence: true
  include_manual_skill: true
```

- `enabled` - добавлять диагностический блок в Responses API ответ.
- `placement` - куда вставлять диагностику: `start`, `end` или `both`.
- `format` - формат блока: видимый текст `visible_block` или `html_comment`.
- `include_*` - какие поля маршрутизации и режима ответа включать в диагностику.

## Проверка API вручную

Chat Completions:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dnd-skill-router",
    "messages": [
      {"role": "user", "content": "!story Опиши вход в древний храм"}
    ],
    "stream": false
  }'
```

Responses API streaming:

```bash
curl -N http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dnd-skill-router",
    "input": "!story Напиши два предложения, описывающие жару",
    "stream": true
  }'
```

## Тесты

Обычные тесты не требуют запущенного LM Studio:

```bash
python -m pytest
```

Live-интеграция с LM Studio вынесена отдельно:

```powershell
$env:RUN_LM_STUDIO_INTEGRATION="1"
$env:LM_STUDIO_LIVE_TIMEOUT="900"
$env:LM_STUDIO_LIVE_MAX_OUTPUT_TOKENS="80"
python -m pytest tests/test_lm_studio_live_integration.py -m lm_studio_live -s
```

Перед live-тестами убедитесь, что LM Studio server запущен, endpoints доступны, а `router_model` и `main_model` совпадают с именами моделей в LM Studio.

## Ограничения MVP

- На один запрос выбирается только один skill.
- Proxy не хранит состояние между запросами.
- Obsidian Copilot должен отправлять историю диалога в каждом запросе.
- Routing зависит от качества JSON, который возвращает router model.
- При низкой уверенности main model не вызывается.
- Manual commands должны быть в начале последнего пользовательского сообщения.
- Полный Obsidian context не логируется по умолчанию.
- RAG, long-term memory, автоматическое редактирование заметок и multi-skill pipeline не реализованы.
