# Шаблон ДЗ «Real-time NLP/CV инференс»

📄 **Начни с [ЗАДАНИЕ.md](ЗАДАНИЕ.md)** — там постановка, критерии и порядок сдачи.
Контракт ручек — [API_CONTRACT.md](API_CONTRACT.md) (человекочитаемо) и
[openapi.yaml](openapi.yaml) (OpenAPI 3.0 — откроется в Swagger UI / editor.swagger.io),
схема стенда — [architecture.svg](architecture.svg).

Это **скелет**: инфраструктура (`Makefile`, скрипты, `.env.example`, `Dockerfile`,
`imagenet_classes.txt`, `prometheus/`) дана, а gateway, `docker-compose.yml` и
`config.pbtxt` вы пишете сами по контракту. CV-модель у каждого своя — см. `make variant`.

## Быстрый старт

```bash
cp .env.example .env    # задайте JWT_SECRET и данные для /info (настоящий email!)
make variant            # ваша CV-модель → впишите CV_MODEL=... в .env
make init               # venv + экспорт модели → model_repository/<CV_MODEL>/1/model.onnx
make up                 # поднять стек
make token              # Bearer-токен для грейдера
make smoke              # прогнать свои ручки (для /vision положите sample.jpg)
```

Дальше — [ЗАДАНИЕ.md](ЗАДАНИЕ.md).

## Моё решение

Стек: gateway (FastAPI, JWT HS256, асинхронные клиенты) → Triton (ResNet50, ONNX Runtime,
dynamic batching) и vLLM (Qwen2.5-1.5B-Instruct), метрики в Prometheus. Обе модели
делят один GPU (NVIDIA A10, 24 ГБ).

- `gateway/main.py` — все ручки контракта. Препроцессинг ResNet50 (resize 256, center crop 224,
  нормализация ImageNet) выносится в поток через `asyncio.to_thread`, чтобы не блокировать event loop.
  `/chat` сначала проверяет статус ответа vLLM и только потом начинает стрим, поэтому ошибки
  отдаются честными HTTP-кодами.
- `model_repository/resnet50/config.pbtxt` — dynamic batching, окно `max_queue_delay_microseconds: 15000`.
  Подобрано по метрикам Triton под нагрузкой 20 параллельных запросов: при 5 мс средний батч 1.69,
  при 15 мс — 2.19; одиночный `/vision` при этом остаётся около 70 мс.
- `docker-compose.yml` — Triton грузит только модель своего варианта (`--model-control-mode=explicit`),
  vLLM ограничен по памяти (`--gpu-memory-utilization 0.5`, `--max-model-len 4096`).

Результат грейдера (25.11.2026): **100 / 100**, все 10 критериев пройдены полностью. В частности:
p95 `/vision` = 30 мс, TTFT `/chat` = 73 мс, нагрузка 4023/4023 запросов без 5xx,
2.46 изображения на один запуск модели.

Запуск: см. «Быстрый старт». Файл `model.onnx` в репозиторий не входит, его создаёт `make init`.
