from typing import Iterable
import joblib
import numpy as np
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


LIMITE_MINIMO_PROPORCAO = 0.05
LIMITE_MAXIMO_PROPORCAO = 0.95


def gerar_dataset(
    n_samples: int = 1000,
    seed: int = 42,
    proporcao_positivos: float = 0.3,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Gera um dataset sintetico de deteccao de fraude.

    Parametros
    ----------
    n_samples : int
        Numero de amostras a gerar.
    seed : int
        Seed para reprodutibilidade.
    proporcao_positivos : float
        Proporcao da classe positiva. Deve estar entre 0.05 e 0.95.

    Retorna
    -------
    df : pd.DataFrame
        Dataset completo com features e target.
    X : np.ndarray
        Matriz de features pronta para uso com scikit-learn.
    y : np.ndarray
        Vetor de targets pronto para uso com scikit-learn.
    """
    if not (LIMITE_MINIMO_PROPORCAO <= proporcao_positivos <= LIMITE_MAXIMO_PROPORCAO):
        raise ValueError(
            "proporcao_positivos deve estar entre "
            f"{LIMITE_MINIMO_PROPORCAO:.2f} e {LIMITE_MAXIMO_PROPORCAO:.2f}, "
            f"recebido: {proporcao_positivos}"
        )

    rng = np.random.default_rng(seed)
    fraude = rng.choice(
        [0, 1],
        size=n_samples,
        p=[1 - proporcao_positivos, proporcao_positivos],
    )

    valor_transacao = np.where(
        fraude,
        rng.uniform(500, 10000, n_samples),
        rng.uniform(10, 800, n_samples),
    ).round(2)

    hora_transacao = np.where(
        fraude,
        rng.integers(0, 6, n_samples),
        rng.integers(7, 23, n_samples),
    )

    distancia_ultima_compra = np.where(
        fraude,
        rng.uniform(100, 5000, n_samples),
        rng.uniform(0, 50, n_samples),
    ).round(1)

    tentativas_senha = np.where(
        fraude,
        rng.integers(2, 10, n_samples),
        rng.integers(1, 2, n_samples),
    )

    pais_diferente = (rng.random(n_samples) < np.where(fraude, 0.4, 0.05)).astype(int)

    df = pd.DataFrame(
        {
            "valor_transacao": valor_transacao,
            "hora_transacao": hora_transacao,
            "distancia_ultima_compra": distancia_ultima_compra,
            "tentativas_senha": tentativas_senha,
            "pais_diferente": pais_diferente,
            "target": fraude,
        }
    )

    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()
    return df, X, y


def avaliar_proporcao_positivos(
    proporcao_positivos: float,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Gera dados e treina um modelo simples usando X e y diretamente com scikit-learn.
    """
    _, X, y = gerar_dataset(
        n_samples=n_samples,
        seed=seed,
        proporcao_positivos=proporcao_positivos,
    )

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_treino, y_treino)
    y_prev = modelo.predict(X_teste)

    return {
        "proporcao_solicitada": proporcao_positivos,
        "proporcao_real": float(y.mean()),
        "acuracia": float(accuracy_score(y_teste, y_prev)),
    }


def demonstrar_proporcoes(proporcoes: Iterable[float]) -> None:
    for proporcao in proporcoes:
        resultado = avaliar_proporcao_positivos(proporcao)
        print(
            "proporcao_positivos="
            f"{resultado['proporcao_solicitada']:.2f} | "
            f"proporcao_real={resultado['proporcao_real']:.3f} | "
            f"acuracia={resultado['acuracia']:.3f}"
        )


def demonstrar_erro_parametro() -> None:
    for proporcao_invalida in (0.01, 1.2):
        try:
            gerar_dataset(proporcao_positivos=proporcao_invalida)
        except ValueError as exc:
            print(f"erro para proporcao_positivos={proporcao_invalida}: {exc}")


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    df, X, y = gerar_dataset(n_samples=2000, seed=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["legítimo", "fraude"]))
    joblib.dump(model, "model.pkl")
    tamanho_kb = os.path.getsize("model.pkl") / 1024
    print(f"Modelo salvo: model.pkl ({tamanho_kb:.1f} KB)")

    model_carregado = joblib.load("model.pkl")
    amostra = X_test[:5]

    pred_original  = model.predict(amostra)
    pred_carregado = model_carregado.predict(amostra)

    assert np.array_equal(pred_original, pred_carregado), "Predições divergem!"
    print("✅ Artefato validado")
    print(f"Predições: {pred_carregado}")
    print(f"Probabilidades: {model_carregado.predict_proba(amostra).round(3)}")


