# src/train/train_credit.py
import os, re, json, pathlib
import pandas as pd
import numpy as np
import mlflow, mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from mlflow.models.signature import infer_signature

CANDIDATE_TARGETS = ["risk", "default", "target", "y", "label", "bad"]
DEFAULT_STATES = {"bad", "default", "charged off", "chargeoff", "write-off"}

def norm_cols(cols):
    return [re.sub(r"\s+", "_", c.strip()).lower() for c in cols]

def drop_index_like(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c.strip() == "" or c.strip().lower() in ("unnamed: 0", "index")]
    return df.drop(columns=drop) if drop else df

def coerce_target_to_binary(df, target, positive_values=None):
    s = df[target]
    if positive_values is None:
        positive_values = {"1", "true", "sim", "yes", "y", "bad"}
    positive_values = {str(v).strip().lower() for v in positive_values}
    # numérico
    if s.dtype.kind in "biu":
        return s.astype(int)
    # texto -> binário
    return s.astype(str).str.strip().str.lower().isin(positive_values).astype(int)

def pick_or_build_target(df, target_env, positive_values_env):
    cols = df.columns.tolist()
    # 1) variável de ambiente manda
    if target_env:
        t = target_env.strip().lower()
        if t not in cols:
            raise ValueError(f"TARGET_COL='{t}' não existe no CSV. Colunas: {cols}")
        pos = None
        if positive_values_env:
            pos = [v for v in positive_values_env.split(",") if v != ""]
        y = coerce_target_to_binary(df, t, pos)
        return y, t
    # 2) tentar candidatos padrão
    for c in CANDIDATE_TARGETS:
        if c in cols:
            return coerce_target_to_binary(df, c, None), c
    raise ValueError(
        "Não achei coluna-alvo. Informe TARGET_COL=Risk (ou nome equivalente) "
        "ou inclua uma coluna alvo no CSV."
    )

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit_scoring")

    csv_path = os.getenv("CREDIT_CSV", "data/Credit/credit.csv")
    csv_sep = os.getenv("CSV_SEP", None)
    # ler CSV: tratar 'NA' como nulo
    if csv_sep:
        df = pd.read_csv(csv_path, sep=csv_sep, na_values=["NA"])
    else:
        df = pd.read_csv(csv_path, na_values=["NA"])

    # normaliza nomes e remove coluna de índice ("", "Unnamed: 0", "index")
    orig_cols = df.columns.tolist()
    df.columns = norm_cols(df.columns)
    df = drop_index_like(df)

    target_env = os.getenv("TARGET_COL")
    pos_env = os.getenv("POSITIVE_VALUES")  # ex.: "bad,1,TRUE"
    y, target_used = pick_or_build_target(df, target_env, pos_env)

    # separa numéricas e categóricas (exclui alvo)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_used in num_cols: num_cols.remove(target_used)
    if target_used in cat_cols: cat_cols.remove(target_used)
    if not num_cols and not cat_cols:
        raise ValueError("Não há colunas para treinar (após excluir o alvo).")

    X = df[num_cols + cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=0.25, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop"
    )

    pipe = Pipeline([
        ("prep", pre),
        ("model", LogisticRegression(max_iter=1000))
    ])

    with mlflow.start_run(run_name="logreg_mixed_baseline"):
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y_test, proba)),
            "f1": float(f1_score(y_test, preds)),
            "acc": float(accuracy_score(y_test, preds)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "model": "LogisticRegression",
            "n_num": len(num_cols),
            "n_cat": len(cat_cols),
            "target": target_used
        })

        sig = infer_signature(X_train, pipe.predict_proba(X_train)[:, 1])
        mlflow.sklearn.log_model(pipe, artifact_path="model",
                                 signature=sig, input_example=X_train.iloc[:5])

        out_dir = pathlib.Path("models/credit/latest")
        out_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(pipe, out_dir / "model.pkl")
        meta = {"features": num_cols + cat_cols, "target": target_used, "csv_path": csv_path}
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("=== Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("Model saved to:", out_dir / "model.pkl")
        print("Metadata saved to:", out_dir / "metadata.json")
        print("Tracking URI:", tracking_uri)

if __name__ == "__main__":
    main()
