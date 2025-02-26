import sys
import csv
import sqlparse
import random
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class Table:
    """Represents a table with a name, data rows, and column metadata.

    Attributes:
        name (str): The name of the table.
        data (list[list[Any]]): The table data as a list of rows, where each row is a list of values.
        meta (list[str]): The column names (metadata) of the table.
    """
    name: str
    data: List[List[Any]]
    meta: List[str]

    def get_sample(self, n: int = 10) -> "Table":
        """Returns a new Table with a random sample of n rows.

        Args:
            n (int): Number of rows to sample. Defaults to 10.

        Returns:
            Table: A new Table instance with sampled rows.
        """
        rows_count = len(self.data)
        sample = [self.data[i] for i in random.sample(range(0, rows_count), min(rows_count, n))]
        return Table(self.name, sample, self.meta)

    def filtered(self, indexes: List[int]) -> Iterator[List[Any]]:
        """Yields rows filtered by the specified column indexes.

        Args:
            indexes (list[int]): List of column indexes to include in each row.

        Yields:
            list[Any]: A filtered row containing only the specified columns.
        """
        for row in self.data:
            yield [row[i] for i in indexes]

    def __add__(self, rhs: "Table") -> "Table":
        """Concatenates two tables if they have compatible structures.

        Args:
            rhs (Table): The table to concatenate with.

        Returns:
            Table: A new Table with combined data.

        Raises:
            ValueError: If tables have different names, row counts, or metadata.
        """
        if self.name != rhs.name or len(self.data) != len(rhs.data) or self.meta != rhs.meta:
            raise ValueError("Different tables structure")
        return Table(self.name, self.data + rhs.data, self.meta)


@dataclass
class AI:
    """AI module that providing column ordering and validation methods."""

    @staticmethod
    def request_columns_order(table: Table, needed_columns: Tuple[str, ...]) -> Dict[int, str | None]:
        """Requests column order mapping from the AI (LLM).

        Args:
            table (Table): The table to process.
            needed_columns (tuple[str, ...]): Tuple of needed column names.

        Returns:
            dict[int, str | None]: Mapping of column indices to names or None.
        """
        return ai_mock(table, needed_columns)

    @staticmethod
    def request_columns_names(table: Table) -> bool:
        """Validates if the table is None.

        Args:
            table (Table): The table to check.

        Returns:
            bool: True if table is None, False otherwise.
        """
        return table is None


def ai_mock(table: Table, needed_columns: Tuple[str, ...]) -> Dict[int, str | None]:
    """Mock AI function that randomly maps column indices to needed columns or None.

    Args:
        table (Table): The table to process (unused in this mock implementation).
        needed_columns (tuple[str, ...]): Tuple of column names to map.

    Returns:
        dict[int, str | None]: A dictionary mapping indices to column names or None.
    """
    random.seed(42)  # For reproducibility
    return {i: col if random.randint(0, 3) == 0 else None for i, col in enumerate(needed_columns)}


@dataclass
class Params:
    """Parameters for parsing, including AI interface.

    Attributes:
        ai (AI): The AI interface used by the parser.
    """
    ai: AI


def parse_string(string: str) -> Any:
    """Parses a string into an appropriate data type or None.

    Args:
        string (str): The string to parse.

    Returns:
        Any: Parsed value (str, int, float, or None).
    """
    string = string.strip()
    if not string or string == "NULL":
        return None
    try:
        if string[0] in "'\"`":
            return string[1:-1]
        if string.isdigit():
            return int(string)
        if string.replace(".", "").isdigit():
            return float(string)
    except IndexError:
        return None
    except ValueError:
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
    return Table(db_name, data, meta)


class Parser:
    """Base class for file parsers.

    Attributes:
        path (Path): Path to the file to parse.
        params (Params): Parser parameters, including AI interface.
        encoding (str): File encoding (default: 'utf-8').
    """

    def __init__(self, path: Path, params: Params, encoding: str = "utf-8"):
        self.path = path
        self.params = params
        self.encoding = encoding

    def parse_csv(self) -> List[Table]:
        """Method to parse CSV content.

        Returns:
            list[Table]: A list containing one Table instance.

        Raises:
            ValueError: If file cannot be read or parsed.
        """
        try:
            with open(self.path, encoding=self.encoding) as file:
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

    def parse_sql(self) -> List[Table]:
        """Method to parse SQL INSERT statements.

        Returns:
            list[Table]: A list of Table instances.

        Raises:
            ValueError: If file cannot be read or parsed.
        """
        try:
            with open(self.path, encoding=self.encoding) as file:
                data = file.read()
            statements = sqlparse.split(data)
            tables = []
            for stmt in statements:
                parsed = sqlparse.parse(stmt)[0]
                if parsed.get_type() == "INSERT":
                    db_name = parsed[2].value.strip()
                    meta = [t.value for t in parsed[3].tokens if isinstance(t, sqlparse.sql.Identifier)]
                    values = [v.value.split(",") for v in parsed.tokens[-1].tokens if v.value.startswith("(")]
                    data = [[parse_string(v.strip()) for v in row] for row in values]
                    tables.append(Table(db_name, data, meta))
            return tables
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.path}")
        except Exception as e:
            raise ValueError(f"SQL parsing error: {e}")


def parse_data(path: Path, params: Params) -> List[Table]:
    """Parses data from files in a directory or a single file based on type (CSV or SQL).

    Args:
        path (Path): Path to a file or directory.
        params (Params): Parser parameters.

    Returns:
        list[Table]: List of parsed Table instances from all matching files.

    Raises:
        ValueError: If no supported file types are found or parsing fails.
    """
    tables: List[Table] = []
    parser = Parser(path, params)
    if path.is_dir():
        logging.info(f"Recursively parsing directory: {path}")
        # Recursively find all .csv and .sql files
        for file_path in path.rglob("*"):
            if file_path.is_file():
                if file_path.suffix == ".csv":
                    logging.info(f"Parsing CSV file: {file_path}")
                    tables.extend(parser.parse_csv())
                elif file_path.suffix == ".sql":
                    logging.info(f"Parsing SQL file: {file_path}")
                    tables.extend(parser.parse_sql())
    elif path.is_file():
        logging.info(f"Parsing single file: {path}")
        if path.suffix == ".csv":
            tables.extend(parser.parse_csv())
        elif path.suffix == ".sql":
            tables.extend(parser.parse_sql())
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    else:
        raise ValueError(f"Path does not exist or is invalid: {path}")

    if not tables:
        raise ValueError("No valid files found to parse")
    return tables


def insert_into_db(obj: Dict[str, Any]) -> None:
    """Simulates database insertion by logging table metadata and parameters."""
    logging.info(f"Inserting into DB: {obj['params']}, {obj['table'].meta}")
    print(obj['table'])


def main() -> None:
    """Main function to parse files."""
    parser = argparse.ArgumentParser(
        description="Parse CSV or SQL files recursively.",
        epilog="Examples:\n"
               "  python leak_parser.py data.csv                # Parse a single CSV file\n"
               "  python leak_parser.py data/                   # Parse all CSV/SQL files in a directory\n"
               "  python leak_parser.py data/ --needed-columns Name Age Email  # Parse with custom columns\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=str, help="Path to a file or directory to parse")
    parser.add_argument("--encoding", type=str, default="utf-8", help="File encoding (default: utf-8)")
    parser.add_argument(
        "--needed-columns",
        type=str,
        nargs="+",
        default=["LastName", "FirstName", "SecondName", "Email", "PhoneNumber", "HomeAddress", "WorkAddress", "Login", "Password"],
        help="List of column names to process"
    )

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    params = Params(AI())
    tables = parse_data(Path(args.path), params)

    needed_columns = tuple(args.needed_columns)
    indexes = [
        {
            "table": table,
            "params": params.ai.request_columns_order(table, needed_columns=needed_columns),
        }
        for table in tables
    ]
    for obj in indexes:
        insert_into_db(obj)


if __name__ == "__main__":
    main()
