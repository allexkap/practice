import argparse
import csv
import logging
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import sqlparse
from openai import APIError, OpenAI
from sqlparse.sql import Identifier, Parenthesis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

    def filtered(self, indexes: list[int]):
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


def ai_mock(table: Table, needed_columns: tuple):
    return {
        random.randint(0, len(table.meta)): col if random.randint(0, 3) == 0 else None
        for col in needed_columns
    }


datatypes = """
Here are most common semantical types of columns in databases:
**Name** -  common human or pet or geographical names
**Surname** -  common human surnames
**Age** - digit in range from 0 to 140
**ID** - digit in wide not negative range.
**Telegram_tag* - starts with `@`
**Mail_address** - ends with `@mail.ru` or other domains
**Geo_longitude\\Geo_latitude**
**Phone** - starts with +
**Password** - senseless sequence of symbols up to 20 characters
**Order**
**Message** - long series of characters with some semantics
"""


@dataclass
class AI:
    api_key: str

    @staticmethod
    def task_request_columns_order(headers, csv_string) -> str:
        return datatypes + (
            f"Here is a CSV file:\n{csv_string}\n"
            f"Find the IDs of columns whose content matches the following headers, and return a single line of column IDs separated by commas (e.g., '0,2,4,'), if no column matches header, return -1 for this header: {headers}. Also use trailing coma\n"
            f"NEVER REPEAT THE ANSWER TWICE or add extra information. For example, this response is error: '-1,0,-1-1,0,-1'"
        )

    @staticmethod
    def task_request_columns_names(csv_string):
        # Write proper prompt to define column names and return tuple of their names
        return datatypes + (
            f"Given the following CSV data: {csv_string}, "
            "identify and return a tuple of column names that represent the data. If there is no header row - identify column names on your own. Possible headers: `Phone`, `Name`, `Surname`, `Email`, `Age`, `ID`, `Telegram`, `Address`, `City`, `Message`."
            "If there are any missing or broken headers, fill them with appropriate names."
            "return only header row as coma-separated list without spaces or quotes, use trailing coma"
        )

    @staticmethod
    def task_find_header(csv_string: str) -> str:
        return datatypes + (
            f"Analyze the first row of the following CSV data: {csv_string}. "
            "Determine whether this row contains column headers. "
            "If headers are not entirely clear, determine more suitable header name, using context of csv file."
            "return only header row as coma-separated list without spaces or quotes"
        )

    @staticmethod
    def toCSV(table: Table):
        if len(table.data) > 100: # ._.
            table = table.get_sample(10)
        csv_data = ""
        for row in table.data:
            csv_data += ",".join(map(str, row)) + "\n"
        return csv_data

    # принимает таблицу и список заголовков. Выдает словарь формата
    # { <column_id> | -2 : <header_string> }
    # -2 если подходящей колонки нет

    def request_columns_order(
            self, table: Table, needs: tuple
    ) -> dict[int | None, Any]:
        csv_str = self.toCSV(table)
        # print("Sample:\n", csv_str)
        prompt = self.task_request_columns_order(needs, csv_str)
        # print('\nPrompt:\n', prompt, '\n')
        response = self.request_cloud_model(prompt)
        # print('Model returned:\n', response)

        ids = response.split(",")  # type: ignore
        ids = self.bullshit_crutch_for_duble_response_bug(ids, len(table.data[0]))

        response = {header: int(index) for index, header in zip(ids, needs)}
        # print(response);
        return response

    # принимает таблицу. Возвращает список заголовков, подобранных для ее столбцов
    def request_columns_names(self, table: Table):
        csv_str = self.toCSV(table)
        prompt = self.task_request_columns_names(self.toCSV(table))
        response = self.request_cloud_model(prompt)
        headers = response.split(",")  # type: ignore
        headers = self.bullshit_crutch_for_duble_response_bug(
            headers, len(table.data[0])
        )
        # print(tuple(headers))
        return headers

    # баг - ИИ почему-то иногда возвращает ответ дважды вопреки инструкциям
    # я не смог исправить, так что пусть будет так
    @staticmethod
    def bullshit_crutch_for_duble_response_bug(items: list, required_headers_len):
        # print(f"req: {required_headers_len}", f"act: {len(items)}")
        if len(items) > required_headers_len:
            return items[0:required_headers_len]
        return items

    # запрос к openrouter.ai на модель qwen 2.5 .
    # Здесь мой токен, юзайте на здоровье пока тестируете.
    # Пожалуйста, замените его на что-то другое, когда будете выкладывать (я напомню)
    def request_cloud_model(self, prompt_final) -> str:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        while True:  # Бесконечный цикл для повторных попыток
            try:
                # print("request sent")
                # Отправляем запрос к API
                completion = client.chat.completions.create(
                    extra_headers={},
                    extra_body={},
                    model="qwen/qwen2.5-vl-72b-instruct:free",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_final},
                            ],
                        }
                    ],
                )

                # Проверяем, что ответ содержит данные
                if (
                        completion
                        and hasattr(completion, "choices")
                        and len(completion.choices) > 0
                ):
                    message = completion.choices[0].message
                    # print(message.content)
                    # print(message.content.strip())
                    return message.content or ""
                    if message and hasattr(message, "content"):
                        # print(message.content.strip())
                        return message.content.strip()
                    else:
                        raise ValueError(
                            "API response does not contain a valid message."
                        )
                else:
                    raise ValueError("API response does not contain any choices.")

            except APIError as e:
                print(f"API error occurred: {e}. Retrying in 30 seconds...")
                time.sleep(30)
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Retrying in 30 seconds...")
                time.sleep(30)
            return ""


class AIMock(AI):
    def __init__(self) -> None:
        pass

    def request_columns_order(
            self, table: Table, needs: tuple
    ) -> dict[int | None, Any]:
        return {
            random.randint(0, len(table.meta)): (
                col if random.randint(0, 3) == 0 else None
            )
            for col in needs
        }

    def request_columns_names(self, table: Table) -> bool:
        return table is None


class DB:
    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path)

    def insert(self, obj):
        raise NotImplementedError


class DBMock(DB):
    def __init__(self, path: Path = ':memory:') -> None: # type: ignore
        super().__init__(path)

    def insert(self, obj):
        print(f"params: {obj['params']}")
        print(f"meta: {obj['table'].meta}")
        for v in [obj['table'].data[i] for i in range(0,5)]:
            print(v)


@dataclass
class Params:
    ai: AI  # интерфейс к ИИ, если парсеру требуется для внутренней работы
    db: DB
    needs: tuple[str]
    experiments: list[str]
    encoding: str = "utf-8"


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


def get_table_info(table: List[str]) -> Table:
    """Extracts table information from a list of SQL INSERT statements.

    Args:
        table (list[str]): List of SQL INSERT statements.

    Returns:
        Table: A Table instance with name, data, and metadata.
    """
    first_line = table[0].strip()
    db_name = first_line[len("INSERT INTO"): first_line.find("(")]
    db_name = parse_string(db_name)
    meta = first_line[first_line.find("(") + 1: first_line.find(")")]
    meta = [parse_string(elem) for elem in meta.split(",")]
    data = table[1:]
    data = [[*map(parse_string, lines[1:-1].split(","))] for lines in data]
    return Table(db_name, data, meta) # type: ignore


@dataclass
class Parser:
    """Base class for file parsers.

    Attributes:
        path (Path): Path to the file to parse.
        params (Params): Parser parameters
    """
    def __init__(self, path: Path, params: Params):
        self.path = path
        self.params = params

    def parse_csv(self) -> list[Table]:
        with open(self.path, encoding=self.params.encoding) as file:
            first_line = file.readline()
            delim = max((first_line.count(d), d) for d in ",;\t")[1]
            file.seek(0)
            reader = csv.reader(file, delimiter=delim)
            meta = list(next(reader))
            content = [[*map(parse_string, lines)] for lines in reader]
            result = Table(self.path.name, content, meta)
            if not self.params.ai.request_columns_names(result):
                result.meta = [str(i) for i in range(len(result.data[0]))]
        return [result]

    def parse_csv_experimental(self) -> List[Table]:
        """Method to parse CSV content.

        Returns:
            list[Table]: A list containing one Table instance.

        Raises:
            ValueError: If file cannot be read or parsed.
        """
        try:
            with open(self.path, encoding=self.params.encoding) as file:
                dialect = csv.Sniffer().sniff(file.readline())
                file.seek(0)
                reader = csv.reader(file, dialect)
                meta = list(next(reader))
                content = [[*map(parse_string, lines)] for lines in reader]
                result = Table(self.path.name, content, meta)
                if not self.params.ai.request_columns_names(result):
                    result.meta = [str(i) for i in range(len(result.data[0]))]
            return [result]
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.path}")
        except csv.Error as e:
            raise ValueError(f"CSV parsing error: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")

    def parse_sql(self) -> list[Table]:
        with open(self.path, encoding=self.params.encoding) as file:
            data = file.readlines()
            table_start_indexes = [
                                      i
                                      for i, line in enumerate(data)
                                      if line.strip().startswith("INSERT INTO")
                                  ] + [len(data)]
            tables = [
                get_table_info(data[i:j])
                for i, j in zip(table_start_indexes, table_start_indexes[1:])
            ]
        return tables

    def parse_sql_experimental(self) -> List[Table]:
        """Parse SQL INSERT statements using sqlparse."""
        try:
            with open(self.path, encoding=self.params.encoding) as file:
                data = file.read()

            statements = sqlparse.split(data)
            tables = []

            for stmt in statements:
                parsed = sqlparse.parse(stmt)
                if not parsed:
                    continue

                parsed_stmt = parsed[0]
                # Check if statement is an INSERT
                if parsed_stmt.get_type() != "INSERT":
                    continue

                # Find the table name
                table_name = None
                for i, token in enumerate(parsed_stmt.tokens):
                    if token.ttype is None and token.value.upper() == "INTO":
                        # Table name should follow the INTO keyword
                        for j in range(i + 1, len(parsed_stmt.tokens)):
                            if isinstance(parsed_stmt.tokens[j], Identifier):
                                table_name = parsed_stmt.tokens[j].get_real_name()
                                break
                        if table_name:
                            break

                if not table_name:
                    continue

                # Find column names
                columns = []
                for token in parsed_stmt.tokens:
                    if isinstance(token, Parenthesis):
                        # First parenthesis after table name should contain column names
                        col_content = token.value.strip("()")
                        # Handle quoted identifiers
                        columns = []
                        in_quotes = False
                        quote_char = None
                        current_col = ""

                        for char in col_content:
                            if char in ['"', "'"]:
                                if not in_quotes:
                                    in_quotes = True
                                    quote_char = char
                                elif char == quote_char:
                                    in_quotes = False
                                    quote_char = None
                                current_col += char
                            elif char == "," and not in_quotes:
                                columns.append(current_col.strip().strip('"\''))
                                current_col = ""
                            else:
                                current_col += char

                        if current_col:
                            columns.append(current_col.strip().strip('"\''))
                        break

                # Find VALUES token and parse value rows
                values_section = False
                values_rows = []
                current_row = []
                in_row_parenthesis = False

                for token in parsed_stmt.tokens:
                    if token.value.upper() == "VALUES":
                        values_section = True
                        continue

                    if not values_section:
                        continue

                    if isinstance(token, Parenthesis) and token.value.startswith("("):
                        # Parse individual values in this row
                        row_content = token.value.strip("()")
                        row_values = []
                        current_value = ""
                        in_quotes = False
                        quote_char = None

                        for char in row_content:
                            if char in ['"', "'"]:
                                if not in_quotes:
                                    in_quotes = True
                                    quote_char = char
                                elif char == quote_char:
                                    in_quotes = False
                                    quote_char = None
                                current_value += char
                            elif char == "," and not in_quotes:
                                row_values.append(self._parse_value(current_value.strip()))
                                current_value = ""
                            else:
                                current_value += char

                        if current_value:
                            row_values.append(self._parse_value(current_value.strip()))

                        values_rows.append(row_values)

                if table_name and values_rows:
                    tables.append(Table(table_name, values_rows, columns))
                return tables
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.path}")
        except Exception as e:
            raise ValueError(f"SQL parsing error: {e}")

    @staticmethod
    def _parse_value(value_str):
        """Parse a SQL value into the appropriate Python type."""
        # Handle NULL values
        if value_str.upper() == "NULL":
            return None

        # Handle string literals
        if (value_str.startswith("'") and value_str.endswith("'")) or \
                (value_str.startswith('"') and value_str.endswith('"')):
            return value_str[1:-1]

        # Handle boolean values
        if value_str.upper() in ["TRUE", "T"]:
            return True
        if value_str.upper() in ["FALSE", "F"]:
            return False

        # Handle numeric values
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # Return as is for other cases
        return value_str


def parse_data(path: Path, params: Params):
    parser = Parser(path, params)
    if path.suffix == ".csv":
        logging.info(f"Parsing CSV file: {path}")
        if params.experiments.__contains__('new_csv'):
            return parser.parse_csv_experimental()
        else:
            return parser.parse_csv()
    if path.suffix == ".sql":
        logging.info(f"Parsing SQL file: {path}")
        if params.experiments.__contains__('new_sql'):
            return parser.parse_sql_experimental()
        else:
            return parser.parse_sql()
    raise ValueError("Unknown file type")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse CSV or SQL files",
        epilog="Examples:\n"
               "  python parser.py -i data.csv   # Parse a single CSV file\n"
               "  python parser.py -i data/      # Parse all CSV/SQL files in a directory\n"
               "  python parser.py -k API_KEY -i data.csv # Use AI model to detect data type\n"
               "  python parser.py -k API_KEY -i data.csv -o db  # Collect in DB",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        required=True,
        help="path to the folder or file containing the data to be parsed.",
    )
    parser.add_argument(
        "-o",
        "--output_db",
        type=Path,
        help="name of the output database where the parsed data will be stored.",
    )
    parser.add_argument(
        "-k",
        "--api_key",
        type=str,
        help="API token for authenticating with the AI service.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output for detailed logging.",
    )
    parser.add_argument("--encoding", type=str, default="utf-8", help="File encoding (default: utf-8)")
    parser.add_argument(
        "--needs",
        type=str,
        nargs="+",
        default=["LastName", "FirstName", "SecondName", "Email", "PhoneNumber", "HomeAddress", "WorkAddress", "Login",
                 "Password"],
        help="List of column names to process"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=[],
        help="List of experiments [new_sql, new_cvs]"
    )
    parser.add_argument("--version", action="version", version="%(prog)s v0.1")

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def process_file(path: Path, params: Params):
    tables = parse_data(Path(path), params)

    indexes = [
        *map(
            lambda table: {
                "table": table,
                "params": params.ai.request_columns_order(
                    table.get_sample(10),
                    needs=params.needs,
                ),
            },
            tables,
        )
    ]
    for obj in indexes:
        params.db.insert(obj)


def main():
    args = parse_args()
    db = DB(args.output_db) if args.output_db else DBMock()
    ai = AI(args.api_key) if args.api_key else AIMock()
    needs = tuple(args.needs)
    experiments = args.experiments
    params = Params(ai, db, needs, experiments)
    if args.input_path.is_dir():
        logging.info(f"Recursively parsing directory: {args.input_path}")
        for path in args.input_path.glob("**/*"):
            process_file(path, params)
    else:
        process_file(args.input_path, params)


if __name__ == "__main__":
    main()
