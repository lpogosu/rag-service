"""Generator for the self-contained evaluation corpus.

The corpus is a fictional documentation set for a fictional event-streaming platform
called Kestrel. Nothing here is copied from a third-party dataset, so the whole
repository stays redistributable.

The important property is not that it is synthetic — it is that questions and gold
spans are derived from the *same* facts the documents are assembled from. A fact is
authored once as a sentence plus the questions it answers; the document builder places
that sentence into a section, and the gold span is the sentence's character range in
the finished document. So the ground truth cannot drift from the text, and it is
expressed in document offsets rather than chunk ids — which is what lets every chunking
strategy be evaluated against the same labels.

Run ``python -m eval.datasets.generate`` to regenerate; the output is byte-identical for
a given seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SEED = 20260214
DATA_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class Fact:
    """One retrievable statement and the questions it answers."""

    key: str
    statement: str
    questions: tuple[str, ...]
    answer: str


@dataclass(frozen=True, slots=True)
class Section:
    heading: str
    intro: str
    facts: tuple[Fact, ...] = ()
    extras: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DocumentSpec:
    doc_id: str
    title: str
    lang: str
    kind: str
    lead: str
    sections: tuple[Section, ...]


@dataclass(frozen=True, slots=True)
class BuiltDocument:
    doc_id: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class GoldQuestion:
    qid: str
    question: str
    answer: str
    doc_id: str
    start: int
    end: int
    kind: str


@dataclass(frozen=True, slots=True)
class UnanswerableQuestion:
    qid: str
    question: str


@dataclass(slots=True)
class Corpus:
    documents: list[BuiltDocument] = field(default_factory=list)
    questions: list[GoldQuestion] = field(default_factory=list)


DOCUMENTS: tuple[DocumentSpec, ...] = (
    DocumentSpec(
        doc_id="overview",
        title="Kestrel — обзор платформы",
        lang="ru",
        kind="guide",
        lead=(
            "Kestrel — платформа доставки событий для внутренних сервисов. "
            "Документ описывает состав системы и основные понятия."
        ),
        sections=(
            Section(
                heading="Компоненты",
                intro="Установка Kestrel состоит из трёх исполняемых компонентов.",
                facts=(
                    Fact(
                        key="components",
                        statement=(
                            "Платформа состоит из брокера kestrel-broker, агента "
                            "kestrel-agent и утилиты командной строки kctl."
                        ),
                        questions=(
                            "Из каких компонентов состоит Kestrel?",
                            "Какие исполняемые файлы входят в поставку Kestrel?",
                        ),
                        answer="kestrel-broker, kestrel-agent и kctl",
                    ),
                    Fact(
                        key="broker-port",
                        statement="Брокер принимает клиентские подключения на порту 7420.",
                        questions=(
                            "На каком порту брокер принимает клиентов?",
                            "Какой порт использует kestrel-broker по умолчанию?",
                        ),
                        answer="7420",
                    ),
                ),
                extras=(
                    "Агент разворачивается рядом с сервисом-источником и буферизует события на "
                    "диске.",
                    "kctl не хранит состояние и обращается к брокеру по тому же протоколу, что и "
                    "агент.",
                ),
            ),
            Section(
                heading="Модель данных",
                intro="Единица публикации — событие, единица хранения — сегмент.",
                facts=(
                    Fact(
                        key="topics-partitions",
                        statement=(
                            "События публикуются в топики; каждый топик делится на партиции, "
                            "и порядок гарантируется только внутри одной партиции."
                        ),
                        questions=(
                            "Где Kestrel гарантирует порядок событий?",
                            "Как связаны топики и партиции в Kestrel?",
                        ),
                        answer="только внутри одной партиции",
                    ),
                    Fact(
                        key="transport",
                        statement=(
                            "Обмен между агентом и брокером идёт по gRPC поверх TLS, "
                            "простой HTTP для публикации событий не поддерживается."
                        ),
                        questions=(
                            "По какому протоколу агент общается с брокером?",
                            "Можно ли публиковать события в Kestrel по обычному HTTP?",
                        ),
                        answer="gRPC поверх TLS; обычный HTTP не поддерживается",
                    ),
                ),
                extras=(
                    "Смещение (offset) — монотонно растущий номер события внутри партиции.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="install",
        title="Установка Kestrel",
        lang="ru",
        kind="guide",
        lead="Порядок установки брокера на выделенный узел и требования к окружению.",
        sections=(
            Section(
                heading="Требования",
                intro="Перед установкой проверьте окружение узла.",
                facts=(
                    Fact(
                        key="kernel",
                        statement=(
                            "Брокеру требуется ядро Linux версии 5.10 или новее: "
                            "на более старых ядрах недоступен io_uring, и запись сегментов "
                            "деградирует примерно вдвое."
                        ),
                        questions=(
                            "Какая минимальная версия ядра Linux нужна брокеру?",
                            "Почему Kestrel требует ядро 5.10?",
                        ),
                        answer="Linux 5.10 или новее, из-за io_uring",
                    ),
                    Fact(
                        key="memory",
                        statement=(
                            "На один брокер нужно не менее 4 ГБ оперативной памяти, "
                            "из них 2 ГБ отводится под страничный кеш."
                        ),
                        questions=(
                            "Сколько памяти нужно брокеру?",
                            "Какой объём RAM требуется на один узел Kestrel?",
                        ),
                        answer="не менее 4 ГБ",
                    ),
                ),
                extras=(
                    "Файловая система на каталоге данных должна поддерживать O_DIRECT.",
                ),
            ),
            Section(
                heading="Пакеты и каталоги",
                intro="Поставка собрана для двух семейств дистрибутивов.",
                facts=(
                    Fact(
                        key="packages",
                        statement=(
                            "Готовые пакеты собираются для Debian 12 и RHEL 9; "
                            "для остальных дистрибутивов поставляется tar-архив."
                        ),
                        questions=(
                            "Для каких дистрибутивов есть пакеты Kestrel?",
                            "Что делать, если моего дистрибутива нет в списке пакетов?",
                        ),
                        answer="Debian 12 и RHEL 9, остальные — tar-архив",
                    ),
                    Fact(
                        key="data-dir",
                        statement=(
                            "Каталог данных по умолчанию — /var/lib/kestrel, "
                            "он должен принадлежать пользователю kestrel."
                        ),
                        questions=(
                            "Где Kestrel хранит данные по умолчанию?",
                            "Какой каталог данных используется брокером?",
                        ),
                        answer="/var/lib/kestrel",
                    ),
                    Fact(
                        key="systemd",
                        statement=(
                            "Служба регистрируется как kestrel-broker.service "
                            "и запускается через systemctl enable --now kestrel-broker."
                        ),
                        questions=(
                            "Как называется systemd-юнит брокера?",
                            "Какой командой включить автозапуск брокера?",
                        ),
                        answer="kestrel-broker.service",
                    ),
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="broker-config",
        title="Конфигурация брокера",
        lang="ru",
        kind="reference",
        lead="Справочник параметров broker.yaml с их значениями по умолчанию.",
        sections=(
            Section(
                heading="Расположение и формат",
                intro="Конфигурация читается один раз при старте процесса.",
                facts=(
                    Fact(
                        key="config-path",
                        statement=(
                            "Основной файл конфигурации — /etc/kestrel/broker.yaml; "
                            "путь переопределяется флагом --config."
                        ),
                        questions=(
                            "Где лежит конфигурация брокера?",
                            "Как указать брокеру другой файл конфигурации?",
                        ),
                        answer="/etc/kestrel/broker.yaml, флаг --config",
                    ),
                ),
                extras=(
                    "Изменения конфигурации не подхватываются на лету, требуется перезапуск.",
                ),
            ),
            Section(
                heading="Хранение",
                intro="Параметры этой секции определяют, сколько данных остаётся на диске.",
                facts=(
                    Fact(
                        key="retention",
                        statement=(
                            "Параметр retention.hours задаёт срок хранения событий "
                            "и по умолчанию равен 168 часам, то есть одной неделе."
                        ),
                        questions=(
                            "Сколько Kestrel хранит события по умолчанию?",
                            "Чему равен retention.hours по умолчанию?",
                        ),
                        answer="168 часов (одна неделя)",
                    ),
                    Fact(
                        key="segment-size",
                        statement=(
                            "Параметр segment.size.mb определяет размер сегмента на диске "
                            "и по умолчанию равен 512 мегабайтам."
                        ),
                        questions=(
                            "Какой размер сегмента используется по умолчанию?",
                            "За что отвечает segment.size.mb?",
                        ),
                        answer="512 МБ",
                    ),
                    Fact(
                        key="flush",
                        statement=(
                            "Параметр flush.interval.ms задаёт интервал принудительного сброса "
                            "на диск и по умолчанию равен 200 миллисекундам."
                        ),
                        questions=(
                            "Как часто брокер сбрасывает данные на диск?",
                            "Чему равен flush.interval.ms по умолчанию?",
                        ),
                        answer="200 мс",
                    ),
                    Fact(
                        key="max-message",
                        statement=(
                            "Максимальный размер одного события ограничен параметром "
                            "max.message.bytes, значение по умолчанию — 1048576 байт."
                        ),
                        questions=(
                            "Какой максимальный размер события в Kestrel?",
                            "Чему равен max.message.bytes по умолчанию?",
                        ),
                        answer="1048576 байт (1 МиБ)",
                    ),
                ),
                extras=(
                    "Уменьшение segment.size.mb ускоряет удаление старых данных, но увеличивает "
                    "число открытых файлов.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="auth",
        title="Аутентификация и токены",
        lang="ru",
        kind="reference",
        lead="Как клиенты подтверждают свою личность и какие права выдаются.",
        sections=(
            Section(
                heading="Токены",
                intro="Все обращения к API требуют токена.",
                facts=(
                    Fact(
                        key="token-prefix",
                        statement=(
                            "Токены доступа имеют префикс ktk_ и передаются "
                            "в заголовке Authorization: Bearer."
                        ),
                        questions=(
                            "Как выглядит токен доступа Kestrel?",
                            "В каком заголовке передаётся токен?",
                        ),
                        answer="префикс ktk_, заголовок Authorization: Bearer",
                    ),
                    Fact(
                        key="token-ttl",
                        statement=(
                            "Срок жизни выданного токена по умолчанию составляет 24 часа; "
                            "после истечения запрос возвращает ошибку KA-2003."
                        ),
                        questions=(
                            "Сколько живёт токен Kestrel?",
                            "Какая ошибка возвращается на просроченный токен?",
                        ),
                        answer="24 часа, затем KA-2003",
                    ),
                    Fact(
                        key="key-rotation",
                        statement=(
                            "Ключи подписи токенов ротируются каждые 30 дней, "
                            "предыдущий ключ остаётся действительным ещё 48 часов."
                        ),
                        questions=(
                            "Как часто ротируются ключи подписи токенов?",
                            "Сколько действует предыдущий ключ подписи после ротации?",
                        ),
                        answer="раз в 30 дней, старый ключ живёт ещё 48 часов",
                    ),
                ),
            ),
            Section(
                heading="Роли",
                intro="Права выдаются ролями, роль привязывается к namespace.",
                facts=(
                    Fact(
                        key="roles",
                        statement=(
                            "Доступны три роли: reader — только чтение, writer — чтение и запись, "
                            "admin — управление топиками и квотами."
                        ),
                        questions=(
                            "Какие роли есть в Kestrel?",
                            "Какая роль нужна, чтобы менять квоты?",
                        ),
                        answer="reader, writer, admin; квоты меняет admin",
                    ),
                ),
                extras=(
                    "Роль admin не даёт доступа к содержимому событий в чужих namespace.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="quotas",
        title="Квоты и ограничения",
        lang="ru",
        kind="reference",
        lead="Ограничения на скорость публикации и поведение при их превышении.",
        sections=(
            Section(
                heading="Значения по умолчанию",
                intro="Квоты применяются на уровне namespace, а не отдельного клиента.",
                facts=(
                    Fact(
                        key="bandwidth-quota",
                        statement=(
                            "Квота полосы по умолчанию — 50 мегабайт в секунду на namespace."
                        ),
                        questions=(
                            "Какая квота по полосе действует по умолчанию?",
                            "Сколько мегабайт в секунду можно писать в namespace?",
                        ),
                        answer="50 МБ/с",
                    ),
                    Fact(
                        key="rate-quota",
                        statement=(
                            "Дополнительно действует ограничение в 10000 событий в секунду, "
                            "оно проверяется раньше квоты по полосе."
                        ),
                        questions=(
                            "Сколько событий в секунду разрешено по умолчанию?",
                            "Какое ограничение проверяется раньше — по полосе или по числу "
                            "событий?",
                        ),
                        answer="10000 событий/с, проверяется раньше квоты по полосе",
                    ),
                ),
            ),
            Section(
                heading="Поведение при превышении",
                intro="Превышение не приводит к разрыву соединения.",
                facts=(
                    Fact(
                        key="quota-exceeded",
                        statement=(
                            "При превышении квоты брокер возвращает HTTP 429 с кодом KQ-0429 "
                            "и заголовком Retry-After в секундах."
                        ),
                        questions=(
                            "Что возвращает брокер при превышении квоты?",
                            "Какой код ошибки соответствует превышению квоты?",
                        ),
                        answer="HTTP 429, код KQ-0429, заголовок Retry-After",
                    ),
                ),
                extras=(
                    "Агент при получении 429 переходит в экспоненциальный бэкофф и продолжает "
                    "копить события на диске.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="replication",
        title="Репликация и отказоустойчивость",
        lang="ru",
        kind="guide",
        lead="Как партиции переживают потерю узла.",
        sections=(
            Section(
                heading="Фактор репликации",
                intro="Реплики распределяются по узлам кластера автоматически.",
                facts=(
                    Fact(
                        key="replication-factor",
                        statement=(
                            "Фактор репликации по умолчанию равен 3, "
                            "минимально допустимое значение для продуктивного контура — 2."
                        ),
                        questions=(
                            "Какой фактор репликации используется по умолчанию?",
                            "Какой минимальный фактор репликации допустим в проде?",
                        ),
                        answer="по умолчанию 3, минимум 2",
                    ),
                    Fact(
                        key="min-insync",
                        statement=(
                            "Параметр min.insync.replicas по умолчанию равен 2: "
                            "запись подтверждается только после сохранения на двух репликах."
                        ),
                        questions=(
                            "Сколько реплик должны подтвердить запись?",
                            "Чему равен min.insync.replicas по умолчанию?",
                        ),
                        answer="2",
                    ),
                ),
            ),
            Section(
                heading="Выборы лидера",
                intro="Лидер партиции обслуживает и запись, и чтение.",
                facts=(
                    Fact(
                        key="leader-timeout",
                        statement=(
                            "Если лидер партиции не отвечает 15 секунд, "
                            "запускаются выборы нового лидера среди синхронных реплик."
                        ),
                        questions=(
                            "Через сколько секунд начинаются выборы нового лидера?",
                            "Кто может стать новым лидером партиции?",
                        ),
                        answer="через 15 секунд, из числа синхронных реплик",
                    ),
                    Fact(
                        key="rack-awareness",
                        statement=(
                            "При включённом rack.awareness реплики одной партиции "
                            "не размещаются в одной стойке."
                        ),
                        questions=(
                            "Что делает параметр rack.awareness?",
                            "Как запретить размещение реплик в одной стойке?",
                        ),
                        answer="включить rack.awareness",
                    ),
                ),
                extras=(
                    "Во время выборов запись в партицию возвращает ошибку KR-3007.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="monitoring",
        title="Мониторинг",
        lang="ru",
        kind="guide",
        lead="Метрики, которые стоит снимать с брокера, и пороги для алертов.",
        sections=(
            Section(
                heading="Экспорт метрик",
                intro="Брокер отдаёт метрики в формате Prometheus.",
                facts=(
                    Fact(
                        key="metrics-endpoint",
                        statement=(
                            "Метрики публикуются на порту 9420 по пути /metrics "
                            "и не требуют аутентификации."
                        ),
                        questions=(
                            "На каком порту брокер отдаёт метрики?",
                            "Нужен ли токен, чтобы забрать метрики?",
                        ),
                        answer="порт 9420, путь /metrics, токен не нужен",
                    ),
                ),
            ),
            Section(
                heading="Что смотреть",
                intro="Три метрики закрывают большинство инцидентов.",
                facts=(
                    Fact(
                        key="lag-metric",
                        statement=(
                            "Отставание потребителя показывает метрика "
                            "kestrel_consumer_lag_seconds; алерт имеет смысл ставить на 60 секунд."
                        ),
                        questions=(
                            "Какая метрика показывает отставание потребителя?",
                            "На каком пороге отставания ставить алерт?",
                        ),
                        answer="kestrel_consumer_lag_seconds, порог 60 секунд",
                    ),
                    Fact(
                        key="disk-metric",
                        statement=(
                            "Метрика kestrel_broker_disk_free_ratio опускается ниже 0.15 "
                            "примерно за час до того, как брокер начнёт отклонять запись."
                        ),
                        questions=(
                            "Какая метрика предупреждает о нехватке места?",
                            "При каком значении disk_free_ratio пора реагировать?",
                        ),
                        answer="kestrel_broker_disk_free_ratio ниже 0.15",
                    ),
                ),
                extras=(
                    "Метрика kestrel_broker_uptime_seconds полезна только для проверки "
                    "перезапусков.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="backup",
        title="Резервное копирование",
        lang="ru",
        kind="guide",
        lead="Снапшоты сегментов и восстановление кластера.",
        sections=(
            Section(
                heading="Снапшоты",
                intro="Снапшот сохраняет сегменты и метаданные топиков.",
                facts=(
                    Fact(
                        key="snapshot-schedule",
                        statement=(
                            "Снапшоты создаются автоматически каждые 6 часов "
                            "и хранятся 14 дней."
                        ),
                        questions=(
                            "Как часто создаются снапшоты?",
                            "Сколько хранятся снапшоты Kestrel?",
                        ),
                        answer="каждые 6 часов, хранение 14 дней",
                    ),
                    Fact(
                        key="snapshot-create",
                        statement=(
                            "Внеочередной снапшот создаётся командой kctl backup create "
                            "--namespace <ns>."
                        ),
                        questions=(
                            "Как создать снапшот вручную?",
                            "Какая команда создаёт внеочередной бэкап?",
                        ),
                        answer="kctl backup create --namespace <ns>",
                    ),
                ),
            ),
            Section(
                heading="Восстановление",
                intro="Восстановление всегда выполняется на остановленный брокер.",
                facts=(
                    Fact(
                        key="restore",
                        statement=(
                            "Восстановление запускается командой kctl backup restore "
                            "--snapshot <id> и требует остановленного брокера."
                        ),
                        questions=(
                            "Как восстановить кластер из снапшота?",
                            "Можно ли восстанавливаться на работающем брокере?",
                        ),
                        answer="kctl backup restore --snapshot <id>, брокер должен быть остановлен",
                    ),
                ),
                extras=(
                    "После восстановления смещения потребителей сбрасываются на момент снапшота.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="errors",
        title="Коды ошибок",
        lang="ru",
        kind="reference",
        lead="Полный список кодов, которые возвращают брокер и агент.",
        sections=(
            Section(
                heading="Ошибки брокера",
                intro="Коды с префиксом KB относятся к самому брокеру.",
                facts=(
                    Fact(
                        key="kb-1001",
                        statement=(
                            "KB-1001 — брокер недоступен: агент не смог установить соединение "
                            "в течение таймаута подключения."
                        ),
                        questions=(
                            "Что означает ошибка KB-1001?",
                            "Какой код возвращается, если брокер недоступен?",
                        ),
                        answer="брокер недоступен",
                    ),
                    Fact(
                        key="kb-1002",
                        statement=(
                            "KB-1002 — партиция не найдена: топик существует, "
                            "но указанный номер партиции выходит за границы."
                        ),
                        questions=(
                            "Что означает ошибка KB-1002?",
                            "Какой код придёт при обращении к несуществующей партиции?",
                        ),
                        answer="партиция не найдена",
                    ),
                ),
            ),
            Section(
                heading="Ошибки квот и доступа",
                intro="Эти коды означают, что запрос корректен, но отклонён политикой.",
                facts=(
                    Fact(
                        key="kr-3007",
                        statement=(
                            "KR-3007 — недостаточно синхронных реплик: "
                            "запись отклонена, потому что число in-sync реплик меньше "
                            "min.insync.replicas."
                        ),
                        questions=(
                            "Что означает ошибка KR-3007?",
                            "Почему запись отклоняется с KR-3007?",
                        ),
                        answer="меньше in-sync реплик, чем min.insync.replicas",
                    ),
                ),
                extras=(
                    "Коды KA относятся к аутентификации, коды KQ — к квотам.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="cli",
        title="Справочник kctl",
        lang="ru",
        kind="reference",
        lead="Команды утилиты kctl, сгруппированные по объекту.",
        sections=(
            Section(
                heading="Топики",
                intro="Управление топиками требует роли admin.",
                facts=(
                    Fact(
                        key="topic-create",
                        statement=(
                            "Топик создаётся командой kctl topic create <name> --partitions N; "
                            "число партиций после создания уменьшить нельзя."
                        ),
                        questions=(
                            "Как создать топик в Kestrel?",
                            "Можно ли уменьшить число партиций после создания топика?",
                        ),
                        answer="kctl topic create <name> --partitions N; уменьшить нельзя",
                    ),
                ),
            ),
            Section(
                heading="Чтение",
                intro="Чтение из терминала удобно для отладки, но не для нагрузки.",
                facts=(
                    Fact(
                        key="consume",
                        statement=(
                            "Команда kctl consume <topic> --from-beginning читает партиции "
                            "с самого раннего доступного смещения."
                        ),
                        questions=(
                            "Как прочитать топик с самого начала?",
                            "Что делает флаг --from-beginning?",
                        ),
                        answer="kctl consume <topic> --from-beginning",
                    ),
                    Fact(
                        key="context",
                        statement=(
                            "Текущий кластер выбирается командой kctl config set-context <name>, "
                            "список контекстов хранится в ~/.kestrel/config."
                        ),
                        questions=(
                            "Как переключиться на другой кластер в kctl?",
                            "Где kctl хранит список контекстов?",
                        ),
                        answer="kctl config set-context <name>, файл ~/.kestrel/config",
                    ),
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="http-api",
        title="Kestrel HTTP API",
        lang="en",
        kind="reference",
        lead="Administrative HTTP API. Event publishing uses gRPC and is not covered here.",
        sections=(
            Section(
                heading="Versioning",
                intro="Every administrative endpoint is versioned in the path.",
                facts=(
                    Fact(
                        key="api-prefix",
                        statement=(
                            "All administrative endpoints live under the /v1 prefix; "
                            "an unversioned path returns 404 rather than redirecting."
                        ),
                        questions=(
                            "What path prefix does the Kestrel admin API use?",
                            "What happens if a client calls an unversioned endpoint?",
                        ),
                        answer="/v1; unversioned paths return 404",
                    ),
                ),
            ),
            Section(
                heading="Endpoints",
                intro="Three endpoints cover topic lifecycle and offset inspection.",
                facts=(
                    Fact(
                        key="create-topic-endpoint",
                        statement=(
                            "POST /v1/topics creates a topic and accepts a JSON body with "
                            "name and partitions fields."
                        ),
                        questions=(
                            "Which endpoint creates a topic over HTTP?",
                            "What fields does the topic creation request body take?",
                        ),
                        answer="POST /v1/topics with name and partitions",
                    ),
                    Fact(
                        key="offsets-endpoint",
                        statement=(
                            "GET /v1/topics/{name}/offsets returns the earliest and latest "
                            "offset of every partition in the topic."
                        ),
                        questions=(
                            "How do I read partition offsets over HTTP?",
                            "Which endpoint reports the latest offset of a topic?",
                        ),
                        answer="GET /v1/topics/{name}/offsets",
                    ),
                    Fact(
                        key="quota-header",
                        statement=(
                            "Every response carries the X-Kestrel-Quota-Remaining header "
                            "with the number of bytes left in the current second."
                        ),
                        questions=(
                            "Which header reports the remaining quota?",
                            "How can a client see how much quota is left?",
                        ),
                        answer="X-Kestrel-Quota-Remaining",
                    ),
                ),
                extras=(
                    "Responses are JSON only; the API does not perform content negotiation.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="performance",
        title="Kestrel performance tuning",
        lang="en",
        kind="guide",
        lead="Settings that matter under load, and the ones that do not.",
        sections=(
            Section(
                heading="Producer side",
                intro="Most throughput problems are batching problems.",
                facts=(
                    Fact(
                        key="linger",
                        statement=(
                            "The agent waits linger.ms before sending a partial batch; "
                            "the default of 5 milliseconds trades a little latency for a much "
                            "larger batch."
                        ),
                        questions=(
                            "What is the default value of linger.ms?",
                            "Why does the agent wait before sending a batch?",
                        ),
                        answer="5 ms, to build larger batches",
                    ),
                    Fact(
                        key="compression",
                        statement=(
                            "Compression is zstd at level 3 by default; raising the level above 6 "
                            "costs more CPU than the bandwidth it saves."
                        ),
                        questions=(
                            "Which compression algorithm does Kestrel use by default?",
                            "Is it worth raising the zstd level above 6?",
                        ),
                        answer="zstd level 3; above level 6 is not worth the CPU",
                    ),
                ),
            ),
            Section(
                heading="Broker side",
                intro="The broker is almost always page-cache bound, not CPU bound.",
                facts=(
                    Fact(
                        key="page-cache",
                        statement=(
                            "Leave at least half of the broker's RAM to the page cache; "
                            "a larger JVM-style heap does not exist here and tuning it is not "
                            "possible."
                        ),
                        questions=(
                            "How much RAM should be left to the page cache?",
                            "Can the broker heap size be tuned?",
                        ),
                        answer="at least half the RAM; there is no tunable heap",
                    ),
                ),
                extras=(
                    "Throughput scales close to linearly with partition count until the disk "
                    "saturates.",
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="migration",
        title="Миграция с версии 1.x",
        lang="ru",
        kind="guide",
        lead="Порядок перехода на 2.x и ограничения обновления.",
        sections=(
            Section(
                heading="Путь обновления",
                intro="Прямое обновление с 1.x на 2.6 не поддерживается.",
                facts=(
                    Fact(
                        key="upgrade-path",
                        statement=(
                            "Обновление с 1.x выполняется через промежуточную версию 2.4, "
                            "которая умеет читать оба формата сегментов."
                        ),
                        questions=(
                            "Как обновиться с Kestrel 1.x до 2.6?",
                            "Зачем нужна промежуточная версия 2.4?",
                        ),
                        answer="через версию 2.4, читающую оба формата сегментов",
                    ),
                    Fact(
                        key="rollback",
                        statement=(
                            "После записи первого сегмента в новом формате откат на 1.x "
                            "невозможен без восстановления из снапшота."
                        ),
                        questions=(
                            "Можно ли откатиться с 2.x обратно на 1.x?",
                            "Что нужно для отката после миграции?",
                        ),
                        answer="только восстановление из снапшота",
                    ),
                ),
            ),
            Section(
                heading="Проверка перед миграцией",
                intro="Проверка не меняет данные и запускается на работающем кластере.",
                facts=(
                    Fact(
                        key="migrate-check",
                        statement=(
                            "Команда kctl migrate check выводит список топиков, "
                            "несовместимых с новым форматом, и завершается с кодом 1, если такие "
                            "есть."
                        ),
                        questions=(
                            "Как проверить готовность кластера к миграции?",
                            "С каким кодом завершается kctl migrate check при несовместимости?",
                        ),
                        answer="kctl migrate check, код возврата 1",
                    ),
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="security",
        title="Безопасность",
        lang="ru",
        kind="reference",
        lead="Требования к транспорту, секретам и аудиту.",
        sections=(
            Section(
                heading="Транспорт",
                intro="Незашифрованные соединения не принимаются ни в одном режиме.",
                facts=(
                    Fact(
                        key="tls-version",
                        statement=(
                            "Брокер принимает только TLS 1.3; поддержка TLS 1.2 удалена в версии "
                            "2.5."
                        ),
                        questions=(
                            "Какие версии TLS поддерживает брокер?",
                            "В какой версии убрали поддержку TLS 1.2?",
                        ),
                        answer="только TLS 1.3, TLS 1.2 удалён в 2.5",
                    ),
                    Fact(
                        key="mtls",
                        statement=(
                            "Агенты аутентифицируются взаимным TLS, "
                            "клиентский сертификат должен содержать namespace в поле OU."
                        ),
                        questions=(
                            "Как аутентифицируются агенты?",
                            "Что должно быть в поле OU клиентского сертификата?",
                        ),
                        answer="mTLS, в OU указывается namespace",
                    ),
                ),
            ),
            Section(
                heading="Секреты и аудит",
                intro="Секреты передаются файлами, чтобы не попадать в список процессов.",
                facts=(
                    Fact(
                        key="secrets",
                        statement=(
                            "Пароли и ключи передаются только через файлы, "
                            "переменные окружения для секретов не поддерживаются."
                        ),
                        questions=(
                            "Можно ли передать секрет Kestrel через переменную окружения?",
                            "Как правильно передавать ключи брокеру?",
                        ),
                        answer="только через файлы",
                    ),
                    Fact(
                        key="audit-log",
                        statement=(
                            "Журнал аудита пишется в /var/log/kestrel/audit.log "
                            "и содержит все административные операции."
                        ),
                        questions=(
                            "Где находится журнал аудита?",
                            "Какие операции попадают в аудит?",
                        ),
                        answer="/var/log/kestrel/audit.log, административные операции",
                    ),
                ),
            ),
        ),
    ),
    DocumentSpec(
        doc_id="changelog",
        title="История изменений",
        lang="ru",
        kind="changelog",
        lead="Значимые изменения последних версий.",
        sections=(
            Section(
                heading="2.6.0",
                intro="Релиз сосредоточен на наблюдаемости.",
                facts=(
                    Fact(
                        key="v260",
                        statement=(
                            "В версии 2.6.0 добавлена метрика kestrel_broker_disk_free_ratio "
                            "и флаг --dry-run у kctl migrate."
                        ),
                        questions=(
                            "Что нового в версии 2.6.0?",
                            "В какой версии появился флаг --dry-run у kctl migrate?",
                        ),
                        answer="2.6.0: метрика disk_free_ratio и --dry-run",
                    ),
                ),
            ),
            Section(
                heading="2.5.0",
                intro="Релиз с несовместимыми изменениями.",
                facts=(
                    Fact(
                        key="v250",
                        statement=(
                            "В версии 2.5.0 удалена поддержка TLS 1.2 "
                            "и изменено значение flush.interval.ms по умолчанию со 100 на 200 "
                            "миллисекунд."
                        ),
                        questions=(
                            "Что изменилось в версии 2.5.0?",
                            "Каким был flush.interval.ms до версии 2.5.0?",
                        ),
                        answer="удалён TLS 1.2, flush.interval.ms изменён со 100 на 200 мс",
                    ),
                ),
                extras=(
                    "Обновление на 2.5.0 требует перегенерации клиентских сертификатов.",
                ),
            ),
        ),
    ),
)

# Distractor documents. Fifteen well-written pages make every question easy: with so few
# candidates almost any retriever reaches recall@5 = 1.0 and the benchmark table stops
# telling you anything. These neighbouring fictional services deliberately reuse the same
# vocabulary — порт, квота, retention, реплики, журнал — so a query has to be resolved by
# meaning and by exact identifiers, not by topic. No distractor carries a gold answer;
# ``build`` asserts that none of them repeats a gold statement.
DISTRACTOR_SERVICES: tuple[tuple[str, str, str], ...] = (
    ("Harrier", "harrier", "объектное хранилище"),
    ("Merlin", "merlin", "планировщик пакетных задач"),
    ("Osprey", "osprey", "шлюз API"),
    ("Falcon", "falcon", "хранилище конфигураций"),
    ("Kite", "kite", "сборщик логов"),
    ("Buzzard", "buzzard", "сервис уведомлений"),
    ("Goshawk", "goshawk", "реестр артефактов"),
    ("Saker", "saker", "кеш метаданных"),
    ("Hobby", "hobby", "служба фоновой очистки"),
    ("Kestrel Console", "kestrel-console", "веб-интерфейс администратора"),
    ("Peregrine", "peregrine", "шина конфигурационных событий"),
    ("Gyrfalcon", "gyrfalcon", "координатор распределённых блокировок"),
)

_DISTRACTOR_PARAMS: tuple[tuple[str, str, str], ...] = (
    ("request.timeout.ms", "таймаут одного запроса", "мс"),
    ("worker.threads", "число рабочих потоков", ""),
    ("queue.capacity", "глубину внутренней очереди", ""),
    ("cache.ttl.seconds", "время жизни записи в кеше", "с"),
    ("batch.max.records", "максимальный размер пачки", ""),
    ("shutdown.grace.seconds", "время ожидания при остановке", "с"),
    ("history.retention.days", "срок хранения истории операций", "дней"),
)

_DISTRACTOR_LIMITS: tuple[tuple[str, str], ...] = (
    ("числа одновременных соединений", "HX-4010"),
    ("размера загружаемого объекта", "HX-4021"),
    ("глубины очереди задач", "MX-5104"),
    ("частоты обращений к API", "OX-6002"),
)


def build_distractors(rng: random.Random) -> list[BuiltDocument]:
    documents: list[BuiltDocument] = []
    for index, (name, slug, purpose) in enumerate(DISTRACTOR_SERVICES):
        port = 8100 + index * 7
        metrics_port = 9100 + index * 3
        overview = [
            f"{name} — {purpose}, разворачивается рядом с Kestrel в том же контуре.",
            f"Служба слушает порт {port} и отдаёт метрики Prometheus на порту {metrics_port}.",
            f"Конфигурация читается из /etc/{slug}/{slug}.yaml при старте процесса.",
            f"Журнал пишется в /var/log/{slug}/{slug}.log и ротируется раз в "
            f"{rng.choice([7, 10, 21, 28])} дней.",
        ]
        params = rng.sample(_DISTRACTOR_PARAMS, k=4)
        settings = [
            f"Параметр {param} задаёт {description} и по умолчанию составляет "
            f"{rng.choice([25, 40, 75, 120, 320, 640, 900])} {unit}.".replace(" .", ".")
            for param, description, unit in params
        ]
        limit, code = rng.choice(_DISTRACTOR_LIMITS)
        limits = [
            f"Ограничение {limit} проверяется на стороне {name} и при превышении "
            f"возвращает код {code}.",
            f"Клиент повторяет отклонённый запрос {rng.choice([2, 3, 4])} раза "
            "с экспоненциальной задержкой.",
            f"Резервная копия состояния {name} создаётся раз в "
            f"{rng.choice([4, 8, 12])} часов и хранится {rng.choice([7, 21, 30])} дней.",
        ]
        diagnostics = [
            f"Состояние службы проверяется запросом GET /healthz на порту {port}.",
            f"Для {name} требуется {rng.choice([1, 2, 8])} ГБ памяти "
            f"и {rng.choice([1, 2, 4])} ядра.",
            f"Обновление {name} выполняется без остановки, если запущено не менее "
            f"{rng.choice([2, 3])} экземпляров.",
        ]
        text = "\n".join(
            [
                f"# {name}",
                "",
                f"Служебная документация: {name}, {purpose}.",
                "",
                "## Назначение",
                "",
                " ".join(overview[:2]),
                "",
                " ".join(overview[2:]),
                "",
                "## Конфигурация",
                "",
                " ".join(settings[:2]),
                "",
                " ".join(settings[2:]),
                "",
                "## Ограничения",
                "",
                " ".join(limits),
                "",
                "## Диагностика",
                "",
                " ".join(diagnostics),
                "",
            ]
        ).rstrip() + "\n"
        documents.append(
            BuiltDocument(
                doc_id=f"neighbour-{slug}",
                text=text,
                metadata={"title": name, "lang": "ru", "kind": "neighbour"},
            )
        )
    return documents


# Questions whose answer is genuinely absent from the corpus. They exist to measure the
# refusal path: a system that answers these is hallucinating, and a retrieval metric
# alone would never notice.
UNANSWERABLE: tuple[str, ...] = (
    "Сколько стоит коммерческая лицензия Kestrel?",
    "Поддерживает ли Kestrel хранение событий в S3?",
    "Как настроить географическую репликацию между дата-центрами?",
    "Какой SLA даёт команда сопровождения Kestrel?",
    "Есть ли у Kestrel клиентская библиотека для Rust?",
    "How do I enable exactly-once delivery in Kestrel?",
    "Какая версия Kestrel поддерживает Windows Server?",
    "Как импортировать данные из Kafka в Kestrel одной командой?",
)


def build_document(spec: DocumentSpec, rng: random.Random) -> BuiltDocument:
    """Assemble one markdown document from its sections."""
    parts = [f"# {spec.title}", "", spec.lead, ""]
    for section in spec.sections:
        parts.extend([f"## {section.heading}", "", section.intro, ""])
        body: list[Fact | str] = [*section.facts, *section.extras]
        # Shuffling fact and filler sentences within a section keeps the corpus from
        # having a fixed "the answer is always the first sentence" shape, which would
        # flatter position-sensitive chunkers.
        rng.shuffle(body)
        buffer: list[str] = []
        for item in body:
            buffer.append(item.statement if isinstance(item, Fact) else item)
            if len(buffer) == 2:
                parts.extend([" ".join(buffer), ""])
                buffer = []
        if buffer:
            parts.extend([" ".join(buffer), ""])
    text = "\n".join(parts).rstrip() + "\n"
    metadata = {"title": spec.title, "lang": spec.lang, "kind": spec.kind}
    return BuiltDocument(doc_id=spec.doc_id, text=text, metadata=metadata)


def build(seed: int = DEFAULT_SEED) -> tuple[Corpus, list[UnanswerableQuestion]]:
    rng = random.Random(seed)
    corpus = Corpus()
    for spec in DOCUMENTS:
        document = build_document(spec, rng)
        corpus.documents.append(document)
        for section in spec.sections:
            for fact in section.facts:
                occurrences = document.text.count(fact.statement)
                if occurrences != 1:
                    raise ValueError(
                        f"{spec.doc_id}/{fact.key}: statement occurs {occurrences} times, "
                        "gold spans must be unambiguous"
                    )
                start = document.text.index(fact.statement)
                for index, question in enumerate(fact.questions, start=1):
                    corpus.questions.append(
                        GoldQuestion(
                            qid=f"{spec.doc_id}.{fact.key}.{index}",
                            question=question,
                            answer=fact.answer,
                            doc_id=document.doc_id,
                            start=start,
                            end=start + len(fact.statement),
                            kind="answerable",
                        )
                    )
    by_id = {document.doc_id: document for document in corpus.documents}
    gold_statements = {
        by_id[question.doc_id].text[question.start : question.end]
        for question in corpus.questions
    }
    for distractor in build_distractors(rng):
        if any(statement in distractor.text for statement in gold_statements):
            raise ValueError(f"{distractor.doc_id} repeats a gold statement")
        corpus.documents.append(distractor)
    unanswerable = [
        UnanswerableQuestion(qid=f"unanswerable.{index:02d}", question=question)
        for index, question in enumerate(UNANSWERABLE, start=1)
    ]
    return corpus, unanswerable


def write(directory: Path = DATA_DIR, seed: int = DEFAULT_SEED) -> tuple[int, int, int]:
    corpus, unanswerable = build(seed)
    directory.mkdir(parents=True, exist_ok=True)
    _dump(
        directory / "corpus.jsonl",
        [
            {"doc_id": doc.doc_id, "text": doc.text, "metadata": doc.metadata}
            for doc in corpus.documents
        ],
    )
    _dump(
        directory / "questions.jsonl",
        [
            {
                "qid": question.qid,
                "question": question.question,
                "answer": question.answer,
                "doc_id": question.doc_id,
                "gold_start": question.start,
                "gold_end": question.end,
                "kind": question.kind,
            }
            for question in corpus.questions
        ],
    )
    _dump(
        directory / "unanswerable.jsonl",
        [{"qid": item.qid, "question": item.question} for item in unanswerable],
    )
    return len(corpus.documents), len(corpus.questions), len(unanswerable)


def _dump(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    corpus, unanswerable = build()
    documents, questions, unanswerable_count = write()
    characters = sum(len(document.text) for document in corpus.documents)
    del unanswerable
    print(
        f"wrote {documents} documents ({characters} characters), "
        f"{questions} answerable questions, {unanswerable_count} unanswerable"
    )


if __name__ == "__main__":
    main()
