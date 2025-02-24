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


@dataclass
class Parser:
    path: Path  # путь к файлу
    params: Params  # параметры парсера
    encoding: str = "utf-8"  # кодировка файла

    def parse_string(self, string: str):
        string = string.strip()
        try:
            if string[0] in "'\"`":
                string = string[1:-1]
        except IndexError:
            return None
        if string == "NULL":
            return None
        return string

    def parse_csv(self) -> list[Table]:
        with open(self.path, encoding=self.encoding) as file:
            first_line = file.readline()
            delim = max((first_line.count(d), d) for d in ",;\t")[1]
            file.seek(0)
            reader = csv.reader(file, delimiter=delim)
            meta = list(next(reader))
            content = [[*map(self.parse_string, lines)] for lines in reader]
        return [Table(self.path.name, content, meta)]

    def get_table_info(self, table: list[str]) -> Table:
        first_line = table[0].strip()
        db_name = first_line[len("INSERT INTO") : first_line.find("(")]
        db_name = self.parse_string(db_name)
        meta = first_line[first_line.find("(") + 1 : first_line.find(")")]
        meta = [self.parse_string(elem) for elem in meta.split(",")]
        data = table[1:]
        data = [[*map(self.parse_string, lines[1:-1].split(","))] for lines in data]
        return Table(db_name, data, meta)

    def parse_sql(self) -> list[Table]:
        with open(self.path, encoding=self.encoding) as file:
            data = file.readlines()
            table_start_indexes = [
                i
                for i, line in enumerate(data)
                if line.strip().startswith("INSERT INTO")
            ] + [len(data)]
            tables = [
                self.get_table_info(data[i:j])
                for i, j in zip(table_start_indexes, table_start_indexes[1:])
            ]
        return tables


def parse_data(path: Path, params: Params):
    parser = Parser(path, params)
    if path.suffix == ".csv":
        return parser.parse_csv()
    elif path.suffix == ".sql":
        return parser.parse_sql()
    else:
        raise ValueError("Unknown file type")


def main(path: str, need_params: bool = False):
    params = Params(AI()) if need_params else None
    tables = parse_data(Path(path), params)
    for table in tables:
        indexes = params.ai.request(
            table.get_sample(), needed_columns=("LastName", "Email", "Phones")
        )
        for row in table.filtred(indexes):
            print(row)


if __name__ == "__main__":
    main("partselect.ru.csv", need_params=True)
