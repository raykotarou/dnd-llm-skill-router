# dnd-skill-router

Локальный OpenAI-compatible proxy для Obsidian Copilot и LM Studio. Proxy принимает chat-запросы, выбирает один DnD-oriented skill, обогащает prompt правилами проекта и отправляет итоговый запрос в локальную модель LM Studio.

## Назначение

Проект помогает направлять запросы по подготовке кампании в один сфокусированный режим:

- `story` - сцены, NPC, диалоги, сюжетные ходы.
- `analysis` - противоречия, логические дыры, проверки согласованности.
- `template` - структуры, таблицы, повторно используемые форматы.
- `lore` - мир, фракции, культуры, география.
- `rules` - механики DnD, проверки, rulings, статблоки.

## Архитектура

Основной поток запроса:

```text
Obsidian Copilot
  -> FastAPI OpenAI-compatible proxy
  -> LangGraph router pipeline
  -> LM Studio OpenAI-compatible /v1/chat/completions
```

Узлы pipeline:

```text
parse_request
extract_latest_user_message
detect_manual_skill_command
rank_skills
select_skill_or_clarify
load_skill_prompt
enrich_prompt
call_lm_studio
format_response
```

Исходные сообщения Obsidian сохраняются. Инструкции proxy добавляются как system messages перед оригинальной историей диалога.

## Установка

Используйте Python 3.11 или новее.

```bash
cd dnd-skill-router
pip install -e .
```

При необходимости создайте локальный env-файл:

```bash
cp .env.example .env
```

В Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

## Запуск

Минимальный запуск:

```bash
pip install -e .
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Если `uvicorn` не находится в `PATH`, используйте:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Проверка состояния:

```text
http://127.0.0.1:8000/health
```

OpenAI-compatible base URL:

```text
http://127.0.0.1:8000/v1
```

## Настройка LM Studio

Запустите локальный сервер LM Studio с OpenAI-compatible endpoint.

Конфиг по умолчанию ожидает:

```yaml
lm_studio:
  base_url: "http://127.0.0.1:1234/v1"
  api_key: "lm-studio"
  router_model: "qwen-router-8b"
  main_model: "qwen-main-35b"
  request_timeout_seconds: 600
```

Измените имена моделей в `config.yaml`, чтобы они совпадали с моделями, загруженными в LM Studio. Router model используется для классификации skill. Main model пишет итоговый ответ.

Большие локальные модели могут отвечать долго. `request_timeout_seconds` задаёт timeout ожидания ответа LM Studio. Timeout долгой генерации не ретраится автоматически, чтобы proxy не ставил повторные тяжёлые запросы в очередь LM Studio.

## Настройка Obsidian Copilot

Настройте Obsidian Copilot на OpenAI-compatible provider:

```text
Base URL: http://127.0.0.1:8000/v1
Model: dnd-skill-router
API key: any non-empty value
```

Proxy предоставляет:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/responses
```

## Manual Commands

Manual commands имеют приоритет над LLM routing и работают только в начале последнего пользовательского сообщения:

```text
!story Придумай сцену в таверне
!analysis Проверь сцену на несостыковки
!template Сделай шаблон NPC
!lore Опиши королевство
!rules Как работает grapple
```

Команда может быть удалена перед prompt enrichment, чтобы итоговая модель получила пользовательский запрос без технического префикса.

## Confidence Threshold

`config.yaml` управляет порогом уверенности routing:

```yaml
routing:
  confidence_threshold: 80
  max_ranked_skills: 3
  use_manual_commands: true
```

Правило:

```text
confidence >= threshold -> selected skill
confidence < threshold  -> clarification response
```

Clarification возвращается как обычное OpenAI-compatible сообщение ассистента, поэтому Obsidian Copilot отображает его как стандартный ответ.

## Примеры

Story:

```text
Придумай сцену в таверне, где NPC пытается скрыть важную улику.
```

Analysis:

```text
!analysis Проверь эту сцену на противоречия с прошлой сессией.
```

Template:

```text
!template Сделай шаблон заметки для фракции в Obsidian.
```

Lore:

```text
Опиши историю портового королевства и его конфликт с магократией.
```

Rules:

```text
!rules Как обработать погоню по крышам в DnD?
```

## Responses API и Reasoning

`POST /v1/responses` работает как дополнительный внешний формат для клиентов, которым нужен OpenAI-compatible Responses API. Внутри используется тот же Skill Router: запрос адаптируется в messages, выбирается skill, собирается enriched prompt, затем proxy вызывает LM Studio `/v1/responses`.

Reasoning events LM Studio для Responses API управляются настройкой:

```yaml
responses_api:
  reasoning:
    mode: "drop"
```

Режимы:

- `drop` - reasoning не показывается пользователю, но usage с `reasoning_tokens` сохраняется.
- `think_block` - reasoning стримится как обычный output text внутри `<think>...</think>`.
- `plain` - reasoning стримится как обычный output text без тегов.
- `pass_through` - raw reasoning events от LM Studio проходят как есть; режим для отладки.

Для отладки можно включить diagnostics:

```yaml
responses_api:
  diagnostics:
    enabled: true
    placement: "start"      # start, end или both
    format: "visible_block" # visible_block или html_comment
```

Streaming-проверка:

```bash
curl -N http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dnd-skill-router",
    "input": "!story Напиши два предложения, описывающие жару",
    "stream": true
  }'
```

Non-streaming-проверка:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dnd-skill-router",
    "input": "Придумай короткую сцену в таверне",
    "stream": false
  }'
```

## Live-тесты LM Studio

Обычный `python -m pytest` не вызывает LM Studio. Для проверки реальной интеграции есть отдельный live-набор, который может выполняться долго из-за JIT loading моделей:

```powershell
$env:RUN_LM_STUDIO_INTEGRATION="1"
$env:LM_STUDIO_LIVE_TIMEOUT="900"
$env:LM_STUDIO_LIVE_MAX_OUTPUT_TOKENS="80"
python -m pytest tests/test_lm_studio_live_integration.py -m lm_studio_live -s
```

Перед запуском убедитесь, что LM Studio server запущен, OpenAI-compatible endpoints доступны, а `router_model` и `main_model` в `config.yaml` совпадают с именами моделей в LM Studio. Эти тесты проверяют `/v1/models`, router-call через `/v1/chat/completions`, non-streaming `/v1/responses` и streaming `/v1/responses` с history, где есть `assistant`-сообщение.

## Ограничения MVP

- Streaming поддерживается для `/v1/chat/completions` и `/v1/responses`, но proxy не хранит состояние между запросами.
- Skill routing зависит от качества JSON, который возвращает router model.
- На один запрос используется только один основной skill.
- Manual commands должны находиться в начале последнего пользовательского сообщения.
- Proxy не хранит состояние диалога; Obsidian Copilot отправляет историю в каждом запросе.
- Подсчёт токенов в локальных fallback-ответах приблизительный или равен нулю.
- LM Studio должен быть запущен отдельно.

## Запрещено в MVP

В MVP нельзя реализовывать:

- multi-skill pipeline;
- persistent routing state;
- long-term memory;
- RAG по Obsidian vault;
- автоматическое редактирование заметок;
- автоматический critic-pass;
- автоматическое управление выгрузкой моделей;
- сложный context filtering;
- UI вне Obsidian Copilot.

Также нельзя:

- удалять историю сообщений из payload;
- выбирать skill по всей истории вместо последнего user message;
- запускать main model при низкой уверенности;
- вызывать router model при manual command;
- логировать полный Obsidian context по умолчанию;
- подменять задачу пользователя собственной цепочкой reasoning;
- сохранять состояние между запросами.
