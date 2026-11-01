# Gemma 4 Developer Agent: куда двигаться после prompt tuning

**Вердикт на 1 ноября 2026:** прекратить искать следующий удачный prompt. Лучший первый эксперимент — **детерминированный skill для локализации и безопасной правки**, второй — **принудительный короткий цикл coder → независимый reviewer → finalizer через SequentialAgent**, третий, с отдельным техническим допуском, — **LoRA на коротких проверенных agent trajectories**. Это ранжирование по вероятности получить проверяемый результат за оставшийся месяц, а не обещание роста LB. У LoRA выше потенциал изменить способности модели, но хуже техническая готовность и уже есть отрицательный публичный результат. Полноценный ParallelAgent/best-of-N пока не ставить на критический путь.

**Важнейшая новая находка:** первичный dataset Idris доступен через API, если кодировать `/` в имени файла как `%2F`. Его лучший опубликованный v24 — обычный single agent **без LoRA/subagents**; v11 — настоящий SequentialAgent, но ledger даёт ему 0.12. На одних и тех же 46 dev tasks v20 с `adapter: distill` решил **13**, v20_base без адаптера — **18**; парные исходы: **3 выигрыша / 8 проигрышей** адаптера. Prompt, sampling и eval_config побайтово одинаковы. Следовательно, «лидеры наверняка уже обучили LoRA/собрали swarm» — неподтверждённая гипотеза, а не основание инвестировать недели. Источники и способ воспроизведения ниже.

**Факт / оценка / неизвестно:** «факт» далее означает прочитанный исходник или публичный артефакт; сообщения участника обозначены отдельно и не превращены в подтверждение staff. «Оценка» — инженерный прогноз/план, не измеренный uplift. Текущие 24/4334 и 0.22 берутся из [локальной проверки LB](leaderboard_check.md); единичный best не считается средним solve rate. Размер public set около 58 — оценка по сетке score, не раскрытый точный N. Прочитаны также TL;DR [noise_and_history.md](noise_and_history.md) и [README shell-first](../submission_bold_shell_first/README.md).

## Ранжирование направлений

Все сроки ниже — **оценки активной работы одного инженера**, GPU-время отдельно. Численного прогноза hidden score доказательства не позволяют; ожидаемый эффект указан как механизм и условие успеха.

| Приоритет | Направление и минимальный эксперимент | Реализуемость / 12 часов | Стоимость | Ожидаемый эффект и доказательства | Риск / критерий остановки |
|---|---|---|---|---|---|
| **1** | **Executable skills:** bounded lexical/AST locator + guarded edit + короткий verification/diff helper; LLM принимает смысловые решения | Поддержано исходниками. Один вызов запускает несколько детерминированных операций в sandbox, время остаётся общим. Без дополнительного LLM-прохода | 3–5 дней, smoke и 2–3 парных GPU-сессии | Устранение повторных поисков, malformed edits, лишних чтений; экономию можно потратить на решение. Это механизм, не доказанный uplift; обычный skill с ещё одним текстом не решает задачу | Ошибки locator, неправильный patch, skill не вызывается. Остановить, если не сокращает time-to-first-edit/ошибки или не даёт новых solves |
| **2** | **SequentialAgent coder → reviewer → finalizer**, при подтверждённой проблеме поиска отдельная ветка locator → coder | Прямо компилируется. Общий бюджет 4.5 мин/28 calls/80 turns, а не отдельный каждому. Sequential избегает установленного риска Parallel | 2–4 дня, 2–3 парных сессии | Структура обеспечивает сам факт передачи, который prompt лишь просит; reviewer ищет конкретный пропущенный контракт/соседний путь. Доказательства эффективности в этом соревновании пока нет; v11 Idris — отрицательное предостережение | Reviewer повторяет ошибку, съедает время, портит правильный patch. Не развивать, если нет выигрышей по свежим failure cases |
| **3, условно** | **LoRA SFT на коротких успешных действиях**, сначала один адаптер rank 8/16 и доказательство его активности | Формат разрешён. Обучение отделено от 12h inference; inference получает startup/KV/latency риск. Практичнее начать GPU smoke с 12B либо ограниченных слоёв 31B | 7–12 дней включая данные/совместимость; GPU-часы до профиля неизвестны | Единственный из трёх меняет веса и распределение ошибок, а не только маршрутизацию. Публичный v20 проиграл базе 13:18; улучшение не доказано, меньший val loss недостаточен | Silent no-op адаптера, OOM, quantization gap, переобучение, отрицательный transfer. Технический gate за 48h, затем прекращать при отсутствии положительного paired сигнала |
| 4 | **12B с большей глубиной поиска/проверки**, тот же общий wall budget | Staff разрешает; скорость надо измерять в актуальном wheelhouse. Все агенты одного размера | 1–3 дня сверх общего eval; 1–2 сессии | Обмен способности на большее число попыток. Сам по себе меньший размер не uplift: опубликованный Idris v22b 9/46 — слабый результат | Локальная скорость не переносится на scorer; правки хуже, преимуществ throughput нет |
| 5 | **Best-of-2 + selection**, последовательно, затем только при исправном runtime параллельно | Только через разрешённые агенты/skills; schema не содержит готового best-of-N/изолированных workspaces/selector. Требует собственного хранения кандидатов в sandbox | 4–7 дней | Возможен выигрыш, только если кандидаты достаточно различаются и selector распознаёт правильный. Oracle pass@2 полезен как верхняя граница, но не как submit metric | Вдвое меньше времени кандидату, общий workspace, weak selector, Parallel bug; full best-of-2 при 4.5 мин на попытку слишком дорог |
| 6 | **LoopAgent self-review**, максимум 2 итерации | Schema/runtime поддерживают. Без подтверждённого механизма выхода использовать только малый max_iterations | 1–2 дня, но ниже приоритетом, чем явные три стадии | Может исправлять ошибки после теста; преимущество над обычным агентом не измерено | Повторная неверная гипотеза, циклы; YAML не даёт произвольного Python stop callback |
| Не брать | Python callbacks / собственный host-side tool registry / смешивание 31B coder + 12B locator | Через исследованный ZIP путь недоступно либо запрещено | Не тратить время | Наличие класса в SDK не равно доступному рычагу submission | Потребовалось бы изменение организаторского harness |

## Что действительно поддержано: проверка исходников

### Точные источники и воспроизводимость

Оба официальных wheel заново получены **read-only**, распакованы в памяти, не устанавливались и не исполнялись:

- **[A: adk_submission 0.2.13](https://www.kaggle.com/api/v1/datasets/download/metric/gemma-4-developer-agent-wheelhouse/adk_submission-0.2.13-py3-none-any.whl)**, 66 528 bytes, SHA-256 `f156f0bd84580ccb9334726d114d7a5481f16d3a6e2887cc4d276ced0ef8d99d`.
- **[S: swegemma 0.2.11](https://www.kaggle.com/api/v1/datasets/download/metric/gemma-4-developer-agent-wheelhouse/swegemma-0.2.11-py3-none-any.whl)**, 119 670 bytes, SHA-256 `9ad876581f3acde8ec3c369858f0834e2b9f41a3cffce3ca42f09cb090aea9d3`.

Ссылки `A:path:line` и `S:path:line` ниже относятся к файлам **внутри этих ZIP/wheel**, строки нумеруются с 1. Это не ссылки на отсутствующие локальные файлы. Получить те же номера можно через `zipfile.ZipFile(io.BytesIO(urlopen(url).read())).read(path).decode().splitlines()`. [Read-only listing wheelhouse](https://www.kaggle.com/api/v1/datasets/list/metric/gemma-4-developer-agent-wheelhouse) возвращает для этих файлов время добавления 2026-10-12; одна версия пакета может перепубликовываться, поэтому фиксировать хеши, а не только номер. Тождественность закрытому scorer сейчас **не установлена**.

| Возможность | Факт из исходника | Практическая граница |
|---|---|---|
| Агент как tool | `A:adk_submission/schema.py:84–137`, `resolvers/tools.py:38–127`: `agent_tool: {config_path: ..., skip_summarization: true/false}` | Это вызываемый подагент, не произвольный Python tool. `skip_summarization` управляет обработкой ответа, не гарантирует отдельный task budget |
| LlmAgent children | `A:schema.py:277–299`: `sub_agents`, `output_key`, `include_contents: default/none`, transfer flags | Ролевое разделение/передача управления; само наличие children не принуждает модель вызвать нужного child |
| Детерминированный workflow | `A:schema.py:326–361`, `builders/workflows.py:40–142`: Sequential/Parallel/Loop; Loop `max_iterations >=1` | Sequential вызывает список по порядку; Loop повторяет список. Никакого YAML `if`, произвольной stop-функции или встроенного patch selector здесь нет |
| Межстадийный результат | `A:builders/llm.py:77–96` передаёт `output_key`, `include_contents`, children в ADK | `output_key` даёт место в state; корректность обращения к нему в instruction и передачи issue надо проверять runtime smoke, особенно при `include_contents: none` |
| Skills | `A:schema.py:287–323`; `builders/llm.py:62–69`; `resolvers/tools.py:129–263` загружает skill и создаёт `SkillToolset` с `code_executor`, `script_timeout`, additional_tools | Отдельные файлы `SKILL.md`, resources и `.py` script; инструментальный workflow исполняется sandbox executor, а не в host процессе |
| Custom tools | `A:resolvers/tools.py:64–65` ищет строковое имя в закрытом tool registry; `schema.py:287` допускает строку либо AgentToolEntry | Прямой `my_tool.py:function`/`CodeConfig` не поддержан. Собственную детерминированную функцию упаковывать как skill script; интерфейсом остаётся `run_skill_script` |
| Callbacks | `A:schema.py:252–254,295–299`; `resolvers/callbacks.py:83–95` возвращает пустые kwargs, если registry нет | **В исследованном S runner callback registry не передаётся:** `S:harness/agent_runner.py:616–628`; `A:compiler.py:63` default `None`. Следовательно, новые имена в YAML в этом пути не создадут callback. Публичного перечня доступных scorer callbacks не найдено |
| Registry | `A:registry.py:1–6,217–240`: заполняется организатором; callbacks pre-approved | Нельзя загрузить собственный Python registry из ZIP. SDK-примеры register(...) — API организатора, не разрешение участнику |
| Ограничения | `S:config.py:19–68` | Общий **распакованный** размер до 3 GiB включительно по проверке `>` в `A:discovery.py:115`; YAML/skill по 50 MiB; 10 000 файлов; 500 агентов; depth50; skills1000; Loop500. Это предельные caps, не разумный проект |
| Расширения | `S:config.py:20–31,40–57` | `.yaml,.yml,.md,.txt,.py,.json,.safetensors`; общий SDK допускает больше расширений, но соревнование сужает список. Поэтому shell-скрипт как отдельный `.sh` не предполагать разрешённым; Python script может вызвать subprocess |
| Sampling | `A:schema.py:202–213`; `S:config.py:60–66` | Поддержанные поля включают seed, penalties, output/thinking; **не** произвольный `extra_body` или server configuration. Это уже исчерпанная ветка, не новый проект |

**Центральный budget:** `S:harness/agent_runner.py:601–628` компилирует весь workflow с одним context/executor; `:713–722,788–813,932–957` применяет общий time/turn/tool ceiling. Subagents не умножают 28 calls и 4.5 минуты. `LoopAgent.max_iterations` — число циклов workflow, `max_turns` — отдельное ограничение LLM calls. Гарантировать собственный жёсткий бюджет каждой стадии только текстом нельзя. Если coder расходует всё время, reviewer не спасёт структуру.

### Модели и ParallelAgent

**Факт:** [актуальная официальная секция model rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/prizes) разрешает ровно четыре QAT варианта: `gemma-4-31b-it-qat-w4a16-ct`, `gemma-4-12b-it-qat-w4a16-ct`, `gemma-4-e4b-it-qat-w4a16-ct`, `gemma-4-e2b-it-qat-w4a16-ct`. Все агенты/subagents одного submission используют **один и тот же** base variant, адаптеры могут различаться. Старый индексированный Overview всё ещё показывает только31B: это устаревший снимок, не причина игнорировать staff.

**Staff:** в [объявлении новых моделей](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287) подтверждены те же условия вычисления; для обучения предлагаются unquantized variants на странице моделей. Там же подтверждено исправление проблемы загрузки 12B в свежем wheelhouse. Участник указывает на ParallelAgent bug ADK1.36, staff признаёт риск перехода на v2; гарантии backport не найдено. Tool calls остаются последовательными даже при параллельных LLM branches.

**Уточнение:** [upstream issue #7324](https://github.com/google/adk-python/issues/7324) касается function-response без соответствующего function-call при compaction и out-of-order parallel events; обсуждение связывает исправление с **ADK2.5.0**, а не просто любым v2. Это не доказательство падения каждого короткого ParallelAgent, но достаточно, чтобы не ставить его в основную ветку до smoke со сжатием контекста. Править SDK у себя не означает исправить scorer.

### LoRA: формат, загрузка, размеры, обучение

**Факт исходника:** `adapter: name` у LlmAgent выбирает обнаруженный adapter (`A:resolvers/models.py:46–79`). `A:discovery.py:384–480` группирует файлы PEFT checkpoint; использовать `adapters/name/adapter_config.json` + `adapter_model.safetensors`. В swegemma маршрут меняет LiteLlm model на `openai/name` с сохранением prefix (`S:harness/agent_runner.py:64–94`), а не сливает веса в YAML.

**Факт исходника:** `A:server.py:759–800` собирает обнаруженные адаптеры, вычисляет максимальный rank с округлением до поддерживаемого vLLM значения и запускает `--enable-lora --max-loras <число> --max-lora-rank <rank> --lora-modules name=path`. Это dynamic adapters поверх общего base model; несколько специалистов могут иметь разные LoRA. Найденный builder не подтверждает старые советы «всегда максимум8 адаптеров/rank128»: число определяется manifest. Полного договорного максимума rank на scorer не установлено; staff сообщает об успешных rank64, это наблюдение, не cap.

Размер ограничивается **всем unpacked bundle <=3 GiB**, а не отдельной квотой LoRA. Rank влияет на память, но размер файла сам по себе не предсказывает startup. `VllmConfig.extra_args`/GPU utilization — host API, отсутствуют в submission agent schema. Нельзя обещать исправление OOM добавлением собственных server flags в ZIP.

**Runtime evidence:** в [Question about LoRA](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749191) участники воспроизводят OOM, сокращение KV-cache и timeout; staff указывает `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` и сообщает об успешных rank64 submissions. В отдельном [отчёте о silent wipe](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743508) участник описывает alias decoder layers, повторный `reset_lora` и одинаковые logprobs base/adapter. Содержательный ответ staff и доказательство исправления именно текущего файла не извлечены: статус **неизвестен**, не «все адаптеры точно сломаны».

**Обучение на Kaggle GPU — разрешённый исследовательский путь, техническая готовность условна.** Staff прямо предлагает fine-tuning меньших моделей. L4x4 дают суммарно96GB, но память не становится автоматически единым пулом. 31B BF16 weights около62GB, плюс activations, optimizer/adapter state и временные буферы: обычный `device_map=auto` не заменяет корректное distributed training. За48h проверить один forward/backward/export на выбранном backend, затем измерить tokens/s/peak VRAM; не обещать full31B long-context обучение за одну сессию. QLoRA требует совместимости тренирующего quantization backend; готовый inference QAT checkpoint нельзя без проверки считать trainable bitsandbytes checkpoint. Указанные память и сроки — оценки.

**Gate до большого обучения:** (1) ненулевой маленький adapter экспортируется в PEFT; (2) официальный VllmServer его обслуживает; (3) фиксированные prompts при temperature0 дают измеримое изменение logprobs относительно base; (4) корректны module names/shapes и tokenizer; (5) часовой mixed-length smoke без OOM/очереди/исчезающих adapters; (6) неизменённая архитектура base vs adapter на валидных dev tasks. Нулевой адаптер подходит для startup/control, но не доказывает активность trained weights.

### Какие данные использовать

- **129 public dev tasks:** [официальное описание](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent/input) содержит issue, gold patch и test_patch. Это задачи, **не готовые агентные trajectories**. На train split можно получить успешные traces с собственным Gemma, выполнить отбор проверкой; gold полезен для проверки, не подмешивается в inference prompt. Сначала навсегда отделить tuning/holdout IDs и желательно related issues/commits; нельзя обучить на всех129 и затем назвать dev score обобщением.
- **[SWE-rebench OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories):** первичная карточка Nebius, примерно67.1k строк, CC-BY4.0, traces Qwen3-Coder/OpenHands, поля tools/trajectory/model_patch/resolved. Взять небольшой отфильтрованный successful subset, Python, короткие tool sequences; убрать дубль issues/commits с holdout. Простое переименование OpenHands tool JSON недостаточно: семантику действий/наблюдений надо преобразовать и replay-проверить.
- **[SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith):** первичная карточка SWE-bench, примерно59.1k задач, MIT. Поля patch/problem/tests дают материал для генерации traces; эта таблица сама по себе не multi-turn trajectory corpus. Начать с небольшого совместимого subset, а не перегенерации всего датасета.

**Предложение:** assistant-action-only SFT на успешных коротких traces с маской user/tool observations; отдельно удержать reasoning/replay формат. Не обучать модель печатать ответы инструментов. Начать с 200–1000 качественных последовательностей, rank8/16, коротких окон, одного checkpoint. Это гиперпараметры пилота, **не проверенный рецепт**; не делать одновременно GRPO, новый scaffold и новую модель.

**Граница правил:** public dev и LoRA официально предусмотрены; для сторонних данных нужны лицензия/provenance и применимые условия конкурса. [Rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules) не отдали читаемый текст в этом исследовании. [Вопрос об external proprietary teachers](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745961) содержит лишь ссылку участника на другой thread, не прочитанное разрешение staff. Поэтому коммерческий teacher API не объявляю однозначно разрешённым; минимальный пилот обходится public released trajectories и собственным Gemma. Ничего организаторам не отправлялось.

## Что реально раскрывают сильные команды

### Idris: теперь первичные файлы, а не пересказ реплики

[Dataset автора](https://www.kaggle.com/datasets/muhammedidris28/gemma4-agent-runs), Apache2.0. Прочитаны README, ledger, configs, runs и training metrics. Read-only URLs:

- [README](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/README.md) описывает наборы и происхождение результатов.
- [leaderboard.json](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/leaderboard.json): v11=.12, v17=.17/.13, v19=.18, v24=.24. Это авторский ledger, не подписанный scorer manifest.
- [v24 agent.yaml](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/agents%2Fv24%2Fagent.yaml): single31B, шесть workspace tools, **нет adapters/skills/subagents**; [sampling](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/agents%2Fv24%2Fconfigs%2Fsampling.yaml): T1/p.95/k64, output8192, thinking4096/replayON. Это уже исследованная нами ветка, не новая архитектура.
- [v11 agent.yaml](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/agents%2Fv11%2Fagent.yaml): **SequentialAgent locator → coder**. Оба child имеют codegraph skill и все9tools: [locator](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/agents%2Fv11%2Fsub_agents%2Flocator.yaml). Это не строгая read-only изоляция locator.
- [runs.jsonl](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/runs.jsonl): 574 task-run строки, SHA-256 `428c8702f10766324108b5868ca8c1ef0d7a8fbaeae487c9a682a294b592cd42`. Пересчёт: v11=6/40; v13=8/40; v14=8/40; v15=4/40; v20=13/46, v20_base=18/46; v21=9/46; v22b=9/46. Подмножества и версии разные — не соединять в общий ranking.
- [v20](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/agents%2Fv20%2Fagent.yaml) отличается от [v20_base](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/agents%2Fv20_base%2Fagent.yaml) строкой adapter; сравнение трёх общих prompt/sampling/eval файлов дало byte equality. На matched46 задачах adapter-only3, base-only8, net−5. Это один прогон на плечо, **не статистическое доказательство вреда LoRA вообще**, и не известно, был ли adapter полностью активен в той версии runtime.
- [training adapter1](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/training%2Fadapter1_swe_smith_all_output.metrics.json), [adapter2](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/training%2Fadapter2_swe_smith_actions_only.metrics.json), [adapter3](https://www.kaggle.com/api/v1/datasets/download/muhammedidris28/gemma4-agent-runs/training%2Fadapter3_swe_rebench_actions_only.metrics.json): реальные TPU training metrics, rank16, action-only и all-output варианты. Checkpoint q/o layers30–59 около28.7MB; loss уменьшается. Это подтверждает существование эксперимента и export checks, **не GPU throughput и не рост solve rate**. Привязка каждого metric-файла к конкретному evaluated adapter не доказана.

[Реплика cosmorin](https://www.kaggle.com/code/cosmorin/gemma-4-idris-v24-e07-faithful-replica) v1 независимо доступна через kernels/pull; явно пишет, что сама ещё не submitted. Она полезна для упаковки, но не заменяет первичный результат. Аналогично [Hanh v8](https://www.kaggle.com/code/honghanhhh/gemma-4-baseline-lb-0-24) при read-only pull показывает наш V10 lineage, один агент/9tools; название .24 не доказывает качество current ZIP.

### Yurnero и остальные .24

| Команда | Что удалось установить | Что не установлено |
|---|---|---|
| Yurnero .29 | [Публичный профиль code](https://www.kaggle.com/samson8/code) не показывает Gemma architecture notebook; в [discussion746008](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746008) есть комментарий о public/private compute | Ни prompt, ни LoRA, ни multi-agent bundle, ни writeup, связанный с .29, не найден. Нельзя объявлять причиной лидерства обучение |
| Yike Xu, SAVE, chizhu, pluris, Alberth Moreira, Pınar Aksoy | Их best .24 подтверждён локальным LB snapshot; сделан поиск имён + competition в публичных code/discussion | Проверяемого раскрытия scored architecture в прочитанных материалах нет. Отсутствие в поиске не доказывает отсутствие публикации вообще |
| tammam alasha | В staff-thread о новых моделях задаёт вопросы о hardware, unquantized training base и актуальности README | Эти вопросы не раскрывают архитектуру его .24 |
| Idris | Есть первичные bundles/ledger/training/runs, описанные выше | v24 не является доказательством, что .24 — стабильный per-run результат |

**Вывод:** лучший найденный открытый рецепт .24 — single agent. Архитектурного секрета .29 из публичных источников не извлечено. Поиск не охватывает закрытые notebooks/личные сообщения; переписка с командами не велась.

## Как сделать три первых направления конкретными

### 1. Skills: перенести повторяемую работу из генерации в исполняемый код

**Предлагаемая архитектура:** один31B solver, один небольшой skill. `locate` получает issue identifiers, ограниченно ищет текст и AST Python definitions/imports/call sites, возвращает path:line и короткие source regions. Не выполнять полный графовый embedding pipeline на каждом task. `apply` принимает точный path/old/new, проверяет путь, уникальность совпадения и protected paths, сохраняет исходную кодировку/не трогает остальное; при ошибке возвращает диагностику без изменений. `check` показывает diff/compile и запускает только явно выбранную короткую проверку с timeout. Не генерировать корректность патча эвристикой.

Это **предложение**, scripts здесь не написаны. Права/защита путей должны быть в коде helper, а не только в prompt. Read/search-only locator не получает edit/write/submit tools; если дать ему `run_command`, read-only остаётся договорённостью, поскольку shell может писать. Для настоящего ограничения лучше dedicated locator script с узким входом и без произвольной команды.

Skill script не получает скрытых тестов/золота и не обращается к сети. Не прятать artifacts внутри repository: scratch под sandbox `/tmp`, чтобы не попасть в diff. Обычная YAML skills декларация не гарантирует вызов script; на smoke проверить tool schema, script arguments, `__file__`/resources и рабочий каталог. Уточнить что именно helper убрал, измеряя failed edits, reading calls, first_edit_s и оставшееся время. Если проблема — семантически неправильный fix, а не операции, этот рычаг не обязан помочь.

### 2. Sequential review: одна дополнительная независимая проверка, один писатель

Минимум: root SequentialAgent с тремя уникальными children. Coder делает первоначальную правку и короткий handoff; reviewer получает issue+diff, проверяет точные требования и соседние реализации, возвращает **один конкретный дефект либо accept**; finalizer выполняет обоснованное исправление/submit. Reviewer не получает mutation tools; shell-free просмотр diff можно обеспечить через skill. Не позволять каждой стадии начинать поиск заново.

В первом smoke проверить, завершает ли coder `submit_patch` весь harness или только возвращает результат: не предполагать, что уже submitted patch можно потом автоматически заменить. До проверки submit оставлять finalizer. Установить явный небольшой Loop max, если он нужен, вместо неограниченной самооценки. Пер-стадийное время, написанное в prompt, является soft target; доказательство, что reviewer реально запускается, — traces.

Localization-first сравнить отдельно только на задачах с неверным первым файлом/символом, затем на общем30. Не строить planner+locator+coder+tester+critic сразу: именно такой overhead может съесть весь эффект. Ранее локально уже существовали analyzer variants без надёжного результата; новая работа должна проверять enforced workflow и качество handoff, а не повторять старое переименование ролей.

### 3. LoRA: менять поведение, а не имитировать длинный чужой стиль

Сначала доказать работающий inference adapter. Затем один baseline-preserving fine-tune, train/holdout по репозиторию или группе связанных issue, assistant actions only, tool-schema replay check. Цель — больше правильных коротких локализаций/правок и меньше invalid tool calls. Не считать хорошую teacher-forced loss доказательством: evaluated adapter обязан выиграть по полноценным trajectories, оставляя prompt/tools/budget неизменными.

Если31B training stack не заработал за48h, перейти на12B technical pilot; это отдельная модельная ветка, сравнивать adapter12B с **base12B**, а затем winner12B с31B на одинаковом wall budget. Нельзя применять12B LoRA к31B или делать маленького planner при31B root. Не расходовать две недели на общий RL с ненадёжным grader.

## Best-of-N: что возможно и где реальная стоимость

**Предложение реализации:** bounded sequential candidate A/B, scripts сохраняют unified diffs под `/tmp`, восстанавливают только изменения кандидата относительно исходного snapshot, передают selector issue+два patch+доступную диагностику; только выбранный patch остаётся в рабочем дереве. Сначала read-only proposals, один applier, без конкурирующих edits. Не делать безусловный `git reset --hard` без известного исходного состояния. Изолированные worktrees, перенос skills и команды вне `/workspace` требуют runtime проверки, их готовность не подтверждена одной schema.

На dev сначала измерить oracle union и затем actual selector: если union двух коротких попыток не превосходит одну длинную, selector не нужен. Формула `1-(1-p)^2` — только иллюстрация независимых попыток с прежним p: здесь попытки коррелированы и каждую приходится сокращать, потому численным прогнозом она не служит. Hidden validation не вызывается selector: доступны лишь repository tests/самостоятельные проверки. Модельное голосование само по себе не равно test evidence.

**Расчёт бюджета (оценка, N≈120):** 120×4.5min=9h, остаётся3h setup/startup/прочих затрат; 120×8min=16h до overhead. Для двух полных4.5min кандидатов получится18h. Значит вариант должен делить нынешние4.5min между A/B/selector, либо доказать выигрыш меньшего размера модели. По [официальной оценке](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview) setup включён в12h, финальная patch validation исключена; in-agent tests входят в время агента. ParallelAgent не создаёт отдельные GPU и не распараллеливает hidden tasks по желанию ZIP.

## План 2–4 недели до 2 декабря

### 2–8 ноября: сначала достоверный стенд и skill

1. Сохранить хеши wheel/model/data и неизменённый `reduced_015` как контроль; .22 держать как исторический best, не целевой mean. Прогнать **gold/empty patch control вне agent inference** на планируемых dev tasks. Если gold не проходит или empty уже проходит — task нельзя интерпретировать как обычное evidence качества repair. Список исключений/причин фиксировать до A/B.
2. Прочитать 10–15 failure traces и разделить bad localization, wrong patch, empty patch, tool errors, budget, environment. Выбрать skill, который убирает наблюдаемый повторяемый сбой. 3–5 задач smoke, включая edit mismatch и ошибка test command; затем замороженная valid30 cohort.
3. Парный запуск baseline vs skills на30, затем общий новый sampling seed на тех же задачах. Go/no-go: новые solves без роста crashes/timeouts, плюс ожидаемый механизм в traces. Это operational screen, не доказательство hidden uplift.

**Почему controls обязательны:** [первичный thread747248](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747248) описывает gold failures и empty false positives, shadowing installed package; [thread744370](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744370) различает Docker и Kaggle subprocess. Staff сообщает, что исправляет public tasks, private repos не совпадают с training. Значит даже paired30 без controls может выбирать агента по сломанной среде. Не менять grader только у candidate; repaired-local и unmodified-official результаты хранить раздельно.

### 9–15 ноября: bounded workflow и LoRA technical gate

- Реализовать трёхстадийный Sequential smoke, проверить state/issue handoff, budget/submission semantics и реальный запуск reviewer. Два30-task paired runs против замороженного контроля; skill и workflow сначала сравнить по отдельности.
- До большого обучения —48h LoRA gate. Если активность/запуск/экспорт не доказаны, записать blocker и остановить ветку. Если прошёл — собрать200–1000 фильтрованных short traces вне holdout и один pilot checkpoint. При paper track отдельный дедлайн12 ноября; не жертвовать воспроизводимой основной веткой ради него.

### 16–22 ноября: один обученный кандидат и независимый holdout

- Paired base vs adapter **одной модели/архитектуры**, с тем же budget; сравнить malformed tools, first edit, timeouts и actual solves. Loss использовать только для выбора checkpoints на training validation.
- Не запускать бесконечный checkpoint search по одному30-task cohort. Сильнейший кандидат сравнить на новых30 untouched issues, убрать связанные train issues; повторить seed при сохранённом task list. При данных только из четырёх public repos перенос на private считать ограниченным.
- Если маленькая модель даёт устойчивую экономию времени, проверить best-of-2 **только** при достаточном oracle headroom. Иначе оставить эту ветку закрытой.

### 23–29 ноября: подтвердить победителя и время всего набора

- Не комбинировать два слабых эффекта в надежде на чудо; максимум один подтверждённый skill + один подтверждённый workflow/adapter, затем отдельный тест комбинации.
- Smoke официального compiler на **точном** ZIP, полный прогон максимально доступного набора с timing setup/inference отдельно от grader. Оценить wall upper bound для≈120 задач, желательно запас≥15–20% от12h; это инженерная цель, не гарантия N.
- Только после paired evidence расходовать submissions по решению координатора/пользователя. Этот research не отправляет ничего. Старый стабильный ZIP сохранить для rollback/final selection.

### 30 ноября – 2 декабря: заморозить артефакт

Выбрать версию по воспроизводимым результатам, зафиксировать точные hashes, provenance данных/адаптера и необходимые licenses. Финальный smoke без новых prompt/sampling правок. Дедлайн **2026-12-02 23:59 UTC**, то есть **3 декабря02:59 Москвы**; оставить минимум сутки на platform failures. Entry/merge25 ноября, paper12 ноября — отдельные даты ([официальный timeline](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)).

## Как использовать существующий dev_eval без самообмана

Прочитан [gemma4_variant_dev_eval.ipynb](../dev_eval/gemma4_variant_dev_eval.ipynb). **Сейчас** он сравнивает `reduced_015` vs `reproduce_fix_verify`, выбирает30 задач seed2026, добавляет общий sampling seed, чередует порядок плеч, сохраняет `manifest.json`, `task_metrics.csv`, `paired_comparison.csv` и raw logs. Использует общий31B vLLM server и `--sandbox subprocess`, concurrency1. Считает paired-only outcomes и exact McNemar. Не прочитанные/не завершённые пары не превращает в defeats.

**Необходимые будущие изменения стенда (в этом research не выполнены):**

- Включить skills и sub_agent YAML/scripts в `BUNDLES`, обновить provenance/hash registry; общий override сейчас меняет только `configs/sampling.yaml`, а независимые sampling configs children тоже надо зафиксировать. Убедиться, что полный bundle hash учитывает каждый новый файл.
- Для LoRA нынешний ручной vLLM command **не содержит** `--enable-lora/--lora-modules`: одного `adapter:` в YAML недостаточно. Нужен server с manifest/официальным VllmServer и соответствующая маршрутизация endpoint; проверить `/v1/models`. Весовые файлы лучше читать с mounted dataset, не встраивать бинарник в notebook JSON. Сравнение без адаптера на сервере с LoRA и base-only server различать, потому memory/KV overhead различается.
- Для12B нынешние model alias/discovery и один31B server не подходят: отдельные model runs с теми же task IDs, hardware, wall budget, без одновременного размещения31B+12B. Это не просто новое имя в VARIANTS.
- Зафиксировать валидный cohort после controls; на нём30 **полных пар**, одинаковые task IDs/seeds/limits, неизменный harness. При повторе менять sampling seed, сохраняя cohort; holdout выбирается отдельно. Общий seed не делает разные workflows детерминированно эквивалентными.
- Из manifest сохранить ещё wheel file hashes, adapter SHA/rank/modules, training split/provenance. По traces: first_edit_s, target files/symbols, reviewer invocation/изменения, invalid-tool/edit failures, compactions. Существующий CSV не обещает все эти поля автоматически.

**Ресурс:** при4.5min/arm agent-only30×2×4.5=4.5h. Notebook дополнительно резервирует verification/setup **для каждого плеча**, в отличие от scoring12h; проверять число фактически завершённых пар. Если30 не помещаются, продолжать точный запланированный cohort второй сессией, не сравнивать удобные первые18 как будто это полные30.

**Решение по результату:** смотреть W=candidate-only и L=baseline-only, net=W−L; разность aggregate score скрывает регрессии. Предварительный go-сигнал — положительный net в двух seed runs, без роста runtime failures, затем подтверждение fresh30. Практический ориентир +3 net/30 — **наша оценка полезности**, не статистическая значимость: даже W3/L0 даёт McNemar p=.25. Повторы одного task по seeds зависимы; не складывать их как60 независимых issues. Несколько кандидатов создают selection bias, поэтому окончательное решение делается на untouched holdout. Не переводить +3/30 напрямую в +0.10 hidden.

## Границы выполненной работы

Изменён только этот отчёт; официальные wheels и public datasets читались по HTTP в памяти. Не запускались inference/training/GPU/grading, не создавались submissions, не менялись notebooks/ZIP/config, не выполнялись git commits. Schema подтверждена source references; реальная компиляция новой архитектуры и её эффект ещё не проверены.

Не установлены: архитектура .29, средний score топ-команд, точное текущее содержимое закрытого scorer и полнота LoRA/Parallel fixes. Это главные риски; следующий расход времени должен покупать runtime proof и paired evidence, а не ещё одну конфигурацию с убедительным названием.
