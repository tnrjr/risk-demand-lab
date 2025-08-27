"""Validação básica com Great Expectations (schema por YAML).

Uso:
    python -m src.validation.ge_validate --dataset precificacao --path data/Precificacao/sample_submission.csv

Edite `schemas/precificacao.yaml` para ajustar as colunas esperadas.
"""

import argparse
import json
import pandas as pd
import great_expectations as ge
from pathlib import Path
import sys


def load_expected_columns(schema_path: Path) -> list[str]:
    import yaml  # pyyaml não é estritamente necessário, mas GE o instala como dependência

    with open(schema_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(map(str, data.get("expected_columns", [])))


def validate_csv(csv_path: Path, expected_columns: list[str]) -> dict:
    df = pd.read_csv(csv_path)
    ge_df = ge.from_pandas(df)

    # Checa que as colunas são as esperadas (ignorando ordem)
    ge_df.expect_table_columns_to_match_set(expected_columns)

    # Checa não-nulos em cada coluna esperada
    for col in expected_columns:
        ge_df.expect_column_values_to_not_be_null(col)

    results = ge_df.validate()
    return results.to_json_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["precificacao"], help="qual conjunto validar"
    )
    parser.add_argument("--path", required=True, help="caminho do CSV a validar")
    args = parser.parse_args()

    dataset = args.dataset
    csv_path = Path(args.path)

    if dataset == "precificacao":
        schema_path = Path("schemas/precificacao.yaml")
    else:
        print("Dataset desconhecido.", file=sys.stderr)
        sys.exit(2)

    expected = load_expected_columns(schema_path)
    result = validate_csv(csv_path, expected)

    out_dir = Path("reports/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Validação concluída. Relatório em {out_path}")


if __name__ == "__main__":
    main()
