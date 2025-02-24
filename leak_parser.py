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
            raise ValueError("Different tables structure")
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


def parse_string(string: str):
    string = string.strip()
    try:
        if string[0] in "'\"`":
            string = string[1:-1]
    except IndexError:
        return None
    if string == "NULL":
        return None
    return string


def get_table_info(table: list[str]) -> Table:
    first_line = table[0].strip()
    db_name = first_line[len("INSERT INTO ") : first_line.find("(")]
    db_name = parse_string(db_name)
    meta = first_line[first_line.find("(") + 1 : first_line.find(")")]
    meta = [parse_string(elem) for elem in meta.split(",")]
    data = table[1:]
    content = [[*map(parse_string, lines[1:-1].split(","))] for lines in data]
    return Table(content, meta, db_name)


def parse_sql(path: Path, file_encoding: str = "utf8") -> list[Table]:
    with open(path, encoding=file_encoding) as file:
        data = file.readlines()
        table_start_indexes = [
            i for i, line in enumerate(data) if line.strip().startswith("INSERT INTO")
        ] + [len(data)]
        tables = [
            get_table_info(data[i:j])
            for i, j in zip(table_start_indexes, table_start_indexes[1:])
        ]
    return tables


params = Params(AI())
table = parse_csv(Path("./res/partselect.ru.csv"), params)[0]
indexes = params.ai.request(
    table.get_sample(), needed_columns=("LastName", "Email", "Phones")
)
for row in table.filtred(indexes):
    print(row)
