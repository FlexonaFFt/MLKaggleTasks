# Bold Shell First — новый эксперимент

Готовый четырёхфайловый bundle, а не повтор версии 0.15 с несколькими дополнительными правилами. Основная гипотеза: короткий контракт и один интерфейс для поиска/правки сократят зацикливание на чтении и неуспешных `edit_file`. Настройки sampling взяты из опубликованного v24 Idris, а не выбраны случайно. **Score этого варианта неизвестен.**

Kaggle commit name: `--experiment --bold-shell-first`.

## Как использовать

Открыть `gemma4_bold_shell_first.ipynb`, CPU, internet off, Run All. На Kaggle результат — `/kaggle/working/submission.zip`. Локально можно задать `GEMMA_OUTPUT_ROOT`; без него используется `gemma4_output` в текущей директории. В репозитории уже лежит проверенный `submission.zip`.

Notebook содержит только упаковку и проверки; никаких установок, inference, отправок на Kaggle или git commits. Первая упаковка использует stdlib; последняя ячейка проверяет YAML через PyYAML. Notebook не зависит от исследовательских web-источников во время выполнения.

## Что изменено относительно базы 0.15

Точная база: Git `115dc0e`, `gemma4_budget_aware_time_budget_4p5.ipynb`. Извлечены именно литералы четырёх runtime-файлов, существующий notebook не исполнялся и не менялся. Полный unified diff: [bundle_vs_015.diff](bundle_vs_015.diff).

| Файл | Было → стало | Основание и ограничение |
|---|---|---|
| `agent.yaml` | 9 → 4 tools: `run_command`, `read_file`, `get_status`, `submit_patch`; удалены три graph tools, `edit_file`, `write_file` | Единый shell-интерфейс для правок устраняет отдельный цикл exact-match tool failures. Это новая гипотеза по диагнозу [Sundararaj](https://www.jayaprakash.net/blog/2026-09-24-gemma-4-developer-agent-on-kaggle/), а не доказанный выигрыш; Python/heredoc тоже может быть сформирован неверно. |
| `configs/sampling.yaml` | temperature **0.2 → 1.0**, добавлен **top_k: 64** | Точный sampling-файл совпадает по SHA-256 с [репликой Idris v24](https://www.kaggle.com/code/cosmorin/gemma-4-idris-v24-e07-faithful-replica). Автор реплики связывает v19 0.18 → v24 0.24 именно с этими двумя полями; мы его скор не воспроизводили. |
| `eval_config.yaml` | **Побайтово без изменений**: command timeout 240 s, calls 28, time 4.5 min, turns 80 | Не повторяем опасный 8-минутный потолок и не увеличиваем вызовы после предыдущего TimeLimit. |
| `prompts/system.md` | **7,363 → 2,061 bytes**, 330 слов, полный новый текст | Ранняя обоснованная правка к 8 charged calls / 90 s вместо фаз до call 12; нет обязательной pre-edit reproduction; guarded Python replacement через shell, одна короткая проверка и ограниченное исправление. Короткий edit-first workflow поддержан [публичным обзором](https://www.kaggle.com/code/hitarthjain0/gemma-4-agent-what-took-lb-0-12-0-24), но точный новый текст не тестировался моделью. |

Сохраняются 31B, top_p .95, output 8192, thinking budget 4096 и `include_thoughts: true`. Нет subagents, LoRA, skills, callbacks или новых schema keys помимо проверенного `top_k`. Не копировался чужой system prompt. Сохранены attribution Matterhorn3838 и нашей бюджетной базы; источник идеи sampling — MUHAMMED IDRIS BIN YACOOB, диагностики — Jayaprakash Sundararaj.

## Что сейчас делают публичные сильные решения

Исследовано 2026-10-31 через web search и **read-only Kaggle kernels/pull API**. Заголовок notebook, best score страницы, team-best LB и score конкретного ZIP — разные факты. Привязку к закрытым отправкам лидеров получить нельзя; обещать «секрет текущего топа» было бы выдумкой.

- [Idris v24 / E07 replica](https://www.kaggle.com/code/cosmorin/gemma-4-idris-v24-e07-faithful-replica), версия 1: самый конкретный найденный >=0.2 reference. Реплика содержит четыре embedded файла и их хеши: **6 tools без graph**, T=1.0, p=.95, k=64, out=8192, thinking4096/replay ON; **8 min /28 calls /80 turns /240 s**, prompt 3,240 bytes, edit by call12, один check/repair. Заявлены авторский ledger v24=0.24, v19=0.18 и официальная компиляция реплики; сама реплика на момент публикации не была отправлена. [Первичный dataset автора](https://www.kaggle.com/datasets/muhammedidris28/gemma4-agent-runs) указан источником; получить его отдельные файлы прямым unauthenticated download не удалось (404), поэтому ledger не верифицирован независимо.
- [Hanh Tran, Baseline LB 0.24](https://www.kaggle.com/code/honghanhhh/gemma-4-baseline-lb-0-24), текущая версия 8: **название заявляет .24, поисковый снимок страницы показывает best .18 V3**. Извлечённый текущий source прямо основан на нашем .22 bundle: девять tools всё ещё зарегистрированы, graph запрещён prompt; T=.2, 8 min/28 calls, длинный prompt с защитой scratch/test paths и parallel implementations. Это не доказательство выигрыша short prompt или удаления graph tools.
- [Apex Silver](https://www.kaggle.com/code/hitarthjain0/gemma-4-apex-silver-agent-lb-0-24), v1: contract-first single agent, те же 8min/28call; title .24–.26+ и target silver — **не подтверждённый результат этого ZIP**. Автор кредитует Hanh Tran; новая независимая оценка не найдена.
- [Shape of Doubt](https://www.kaggle.com/code/prvsiyan/gemma-and-the-shape-of-doubt): search snapshot показывал best .20 V32, но current pull **v49 сам пишет best measured .18** и воспроизводит наш V10, 9 tools с graph prohibition, T=.2, 8min/28call. Его обзор полезен как карта refs, не как доказательство .20 current bundle.
- [Hitarth Jain, What Took .12→.24](https://www.kaggle.com/code/hitarthjain0/gemma-4-agent-what-took-lb-0-12-0-24), v5: это обзор изменений public frontier, **не его собственный .24**. Полезные идеи — один агент, ранняя правка, буквальный issue contract, чистый diff; часть утверждений о графах и timeout устарела. Его пересказ [Ayush Thakur ablations](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746250) даёт T=.2 48/129, .4 и1.0 51/129, thinking off23/129; это отдельные single runs, а не доказательство выигрыша T1.0.

Официальный [leaderboard](https://www.kaggle.com/competitions/gemma-4-developer-agent/leaderboard) не отдаёт читаемые строки через web extraction. Сторонний [CLIST snapshot](https://clist.by/standings/google-the-gemma-4-developer-agent-competition-coding-artificial-intelligence-custom-metric-70662440/) показывает верх .29, затем .24 и .22, но не раскрывает bundles; .29 решение не установлено. Нельзя переносить старую фразу «топ .24» на текущий LB или считать public notebook названием подтверждённой отправки.

## Схема: что проверено и какие рычаги остаются

Прочитан официальный [adk_submission 0.2.13 wheel](https://www.kaggle.com/api/v1/datasets/download/metric/gemma-4-developer-agent-wheelhouse/adk_submission-0.2.13-py3-none-any.whl), SHA-256 `f156f0bd84580ccb9334726d114d7a5481f16d3a6e2887cc4d276ced0ef8d99d`. `schema.py:202–213` определяет разрешённые generation fields; `resolvers/generation.py:35–45` передаёт их в GenAI config. Это проверка исходников schema, **не локальный запуск официального compiler**.

Также свежескачанный официальный [swegemma 0.2.11 wheel](https://www.kaggle.com/api/v1/datasets/download/metric/gemma-4-developer-agent-wheelhouse/swegemma-0.2.11-py3-none-any.whl), SHA-256 `9ad876581f3acde8ec3c369858f0834e2b9f41a3cffce3ca42f09cb090aea9d3`, побайтово совпал с прочитанными cached `config.py`, `harness/agent_runner.py`, `tools/execution.py`, `tools/base.py`. Публичные notebooks могут описывать другую версию.

| Рычаг | Что действительно поддержано | Что делаем |
|---|---|---|
| temperature, top_p, top_k | schema float>=0, p 0..1, k>=1; runtime пример k64 в v24 | T1.0/k64, p сохранён; top_k уже встречался в ранних локальных variants, но не в стабильной .15 базе |
| max_output_tokens, thinking_config | out>=1; swegemma constraint1..32768; thinking_budget0..32768, include_thoughts, thinking_level | Не режем output/thinking после слабых прошлых результатов; бюджет thinking и сохранение мыслей — разные настройки |
| presence_penalty, frequency_penalty, stop_sequences, response_mime_type, seed | Поля есть в restricted schema; произвольные `extra_body`, `min_p`, repetition_penalty отсутствуют, extra forbidden | Penalties/stop/MIME ранее не найдены в registry; **не используем**: schema acceptance не доказывает желаемое действие vLLM. Seed bridge в source есть, но seed не означает GPU determinism |
| tools | Только официальные имена/agent_tool, shell разрешён; native editing не обязателен | Четыре официальных tools, правки через run_command |
| max_turns, max_tool_calls, max_time_minutes, timeout_seconds | Проверенные четыре поля eval_config и EvalConfig mapping | Без изменений. 80 turns — отдельный потолок, не 80 обычных tools |
| max_llm_calls, verification_timeout_seconds | Есть в Python EvalConfig/CLI, verification default300 отдельно от command timeout | **Не добавлять в bundle на этом основании**: парсер hosted eval_config не проверен для этих полей |
| cost_usd, total_tokens, stdout/file caps, compaction/cache, concurrency/shards | Внутренние dataclass/API возможности swegemma | Это не доказанные submission.yaml knobs; не используем |
| output_key, include_contents, transfer flags; Sequential/Loop/ParallelAgent; skills/adapters | Есть в restricted ADK schema | Без нужды не включаем; callbacks требуют registry, произвольный Python callback запрещён. ParallelAgent имеет [обсуждаемый ADK1.36 дефект](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287) |

Source bridge positive `thinking_budget` включает thinking и передаёт thinking_token_budget; `include_thoughts` управляет replay отдельно. Поэтому старые советы «include_thoughts false = thinking off» неприменимы без версии. Harness `tools/base.py` и decorators в execution tools исключают get_status/submit_patch из charged call count, но они всё равно требуют model turns/time. Поэтому prompt говорит **charged calls**, а не обещает считать каждый status одним из28.

## Empty patch против wrong patch

**Наши hidden failures: доля неизвестна.** Score .12 или .15 сообщает только решённые задачи; без patch sizes, exit reasons и traces нельзя разделить empty, wrong, timeout и crash. Нельзя объявить большинство наших проигрышей пустыми патчами.

[Публичные эксперименты yiyu0716](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747696) показывают на31 dev tasks: v2 — 4/5 empty при13/11 solves, v3 —5/5 empty при13/11 solves, v6.1 —3 empty при13 solves. Среди нерешённых это примерно **17–28% empty**, остальные **72–83% — non-empty failures и/или crashes/timeouts**, не автоматически wrong patch. Другие конфигурации давали10 empty из31 при11 solves (**50% нерешённых**); патологический workflow v6.4 дал21 empty при4 solves (**78% нерешённых**). Это чужие dev runs с иными бюджетами/железом, не оценка нашего hidden set.

[Sundararaj, first-hand traces](https://www.jayaprakash.net/blog/2026-09-24-gemma-4-developer-agent-on-kaggle/) сообщает48 неуспешных edit_file из77 в одном run и39 повторов ошибочной команды в другом. Это объясняет, почему стоит проверить новый editing interface, **не даёт популяционной доли empty patches**. Более поздний [отчёт Pilkwang Kim](https://pilkwangkim.github.io/posts/Gemma-4-Developer-Agent-Part-2-Measuring-an-Agent-While-Its-Evaluator-Moved/) показывает слабую исполняемость длинного test-first workflow; потому здесь убрана обязательная pre-edit reproduction.

В swegemma `harness/agent_runner.py:1047–1104` есть сбор unsubmitted working-tree diff после исчерпания бюджета. Ранняя правка поэтому сохраняет шанс на проверку даже без submit; пустой diff не помогает. Правило «edit by8/90s» остаётся soft instruction, не инструментальная гарантия.

## Время и честное ожидание

При120 tasks agent ceilings: **120×4.5=540min=9h**; остаётся180min на общий startup, sandbox setup и другие учитываемые расходы. При129tasks —580.5min и139.5min резерва. По [правилам](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview), финальная patch validation исключена из12h, sandbox setup включён. Это запас при заданном N, **не измеренная гарантия завершения**: неизвестны setup costs, фактическое N и scorer version. 240s отдельной команды не отменяет asyncio session timeout; post-edit тест ограничен в prompt30s.

Ожидаемый механизм — более ранние реальные source edits, меньше exact-match tool loops, другая sampling trajectory. Обоснованного численного прогноза score нет; **достижение .20–.24 не обещается**, возможен спад ниже .15. Shell может проиграть native edit_file из-за quoting/encoding ошибок, T1.0 может увеличить случайные неверные изменения, сокращение prompt — убрать полезные подсказки. Меняются несколько факторов осознанно: это поиск новой ветки, не причинный A/B отдельных knobs.

## Альтернатива №2 — не реализована

**Idris-style compact native editor, 4.5min**: шесть workspace tools, T1.0/p.95/k64, output8192, thinking4096/replayON, короткий operational prompt около3KB, calls28/turns80/timeout240, первая правка by8–12. Сохранить edit_file/write_file, shell fallback после первого mismatch. Это ближе к конкретному заявленному .24 reference и менее рискованно по quoting, но слабее отличается от наших прошлых native-editor variants. Если shell-first даёт больше malformed commands/empty patches, перейти сюда; второй notebook или ZIP не создавался.

## Проверки и артефакты

- `nbformat.validate`: PASS; все четыре Python-ячейки исполнены сверху вниз в локальном Python3.14, включая YAML includes и exact values.
- Повторное полное исполнение сохранило тот же ZIP; notebook сохранён с execution counts и компактными outputs.
- ZIP: ровно4 файла, CRC и содержимое каждого member совпадают с literals; SHA-256 `a7b8727b94dfbe6fc3f9fb17c00a8f371466a60d0f22a2ef7796c0c95fca9175`.
- [validation.json](validation.json) сохраняет hashes/размеры/status, [bundle_vs_015.diff](bundle_vs_015.diff) — точный diff.
- Локальный execution не запускал Gemma, Docker grading или официальный ADK compiler; Pydantic/ADK отсутствуют в использованном packaging env. Проверка schema сделана чтением официального wheel. Визуальный notebook viewer не проверялся; outputs компактные, ячейки без графиков.
- Kaggle submission, push и git commit не выполнялись. Изменения ограничены новым `submission_bold_shell_first/`.
