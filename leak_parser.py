import csv
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Table:
    content: list
    meta: tuple


def parse_csv(path: Path) -> list[Table]:
    with open(path) as file:
        first_line = file.readline()
        delim = max((first_line.count(d), d) for d in ',;\t')[1]
        file.seek(0)
        reader = csv.reader(file, delimiter=delim)
        meta = tuple(next(reader))
        content = [[*map(str.strip, lines)] for lines in reader]

    return [Table(content, meta)]


def get_sample(table: Table, n: int = 10):
    rows_count = len(table.content)
    content = [
        table.content[i]
        for i in random.sample(range(0, rows_count), min(rows_count, n))
    ]
    return Table(content, table.meta)


def reorder_columns(table: Table, indexes):
    content = [[row[i] if i != -1 else '' for i in indexes] for row in table.content]
    return Table(content, table.meta)


def save_to_db(path: Path, table: Table):
    print(*table.content, sep='\n')


def ai_mock(table: Table, needed_columns: tuple) -> list[int]:
    return [
        (table.meta.index(col) if col in table.meta else -1) for col in needed_columns
    ]


def ai(table: Table, needed_columns: tuple) -> list[int]:
    return ai_mock(table, needed_columns)


table = parse_csv(Path('./res/partselect.ru.csv'))[0]
part = get_sample(table)
indexe = ai(part, needed_columns=('LastName', 'Email', 'Phones'))
out_table = reorder_columns(table, indexe)
save_to_db(Path('out.db'), out_table)
