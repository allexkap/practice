import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Table:
    name: str  # имя таблицы
    data: list[list[Any]]  # сами данные
    meta: list[str]  # имена полей

    def get_sample(self, n: int = 10):
        rows_count = len(self.data)
        sample = [
            self.data[i]
            for i in random.sample(range(0, rows_count), min(rows_count, n))
        ]
        return Table(self.name, sample, self.meta)

    def filtred(self, indexes: list[int]):
        for row in self.data:
            yield [row[i] for i in indexes]

    def __add__(self, rhs: "Table"):
        if (
            self.name != rhs.name
            or len(self.data) != len(rhs.data)
            or self.meta != rhs.meta
        ):
            raise ValueError("nelzya")
        return Table(self.name, self.data + rhs.data, self.meta)


def ai_mock(table: Table, needed_columns: tuple) -> list[int]:
    return [
        (table.meta.index(col) if col in table.meta else -1) for col in needed_columns
    ]


@dataclass
class AI:
    def request(self, table: Table, needed_columns: tuple) -> list[int]:
        return ai_mock(table, needed_columns)


@dataclass
class Params:
    ai: AI  # интерфейс к ИИ, если парсеру требуется для внутренней работы
    # и другое если кому то что то понадобится, расширяемость как никак


def parse_csv(path: Path, params: Params) -> list[Table]:
    with open(path) as file:
        first_line = file.readline()
        delim = max((first_line.count(d), d) for d in ",;\t")[1]
        file.seek(0)
        reader = csv.reader(file, delimiter=delim)
        meta = list(next(reader))
        content = [[*map(str.strip, lines)] for lines in reader]
    return [Table(path.name, content, meta)]


params = Params(AI())
table = parse_csv(Path("./res/partselect.ru.csv"), params)[0]
indexes = params.ai.request(
    table.get_sample(), needed_columns=("LastName", "Email", "Phones")
)
for row in table.filtred(indexes):
    print(row)
