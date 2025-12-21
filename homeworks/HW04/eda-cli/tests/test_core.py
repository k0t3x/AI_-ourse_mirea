from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    """Базовый тестовый датафрейм"""
    return pd.DataFrame({
        "age": [10, 20, 30, None],  # числовая колонка с пропуском
        "height": [140, 150, 160, 170],  # числовая без пропусков
        "city": ["A", "B", "A", None],  # категориальная с пропуском
    })


def _test_df_for_quality_heuristics() -> pd.DataFrame:
    """Датафрейм с разными проблемами качества данных"""
    return pd.DataFrame({
        "user_id": [1001, 1002, 1003, 1003, 1005],  # дублирующиеся ID
        "country": ["RU", "RU", "RU", "RU", "RU"],  # одно значение
        "category": ["A", "A", "A", "A", "A"],  # константная колонка
        "revenue": [0, 0, 0, 0, 0],  # все нули
        "name": [f"User_{i}" for i in range(5)],  # уникальные значения
        "value": [1.5, 2.3, 3.1, None, 4.2],  # обычная числовая колонка
    })


def test_summarize_dataset_basic():
    """Тест базовой сводки по данным"""
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    """Тест анализа пропусков и флагов качества"""
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    """Тест корреляции и топ-категорий"""
    df = _sample_df()
    corr = correlation_matrix(df)

    # Проверяем, что корреляция посчитана
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_new_heuristics():
    """Тест новых эвристик обнаружения проблем"""
    df = _test_df_for_quality_heuristics()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Константные колонки
    assert flags["has_constant_columns"] is True
    assert "country" in flags["constant_columns"]
    assert "category" in flags["constant_columns"]

    # Дубликаты в ID колонках
    assert flags["has_suspicious_id_duplicates"] is True
    assert len(flags["id_duplicates_info"]) > 0

    found_user_id = False
    for info in flags["id_duplicates_info"]:
        if info["name"] == "user_id":
            assert info["duplicate_rate"] > 0
            found_user_id = True
            break
    assert found_user_id

    # Высокая кардинальность
    assert flags["has_high_cardinality_categoricals"] is True
    assert len(flags["high_cardinality_columns"]) > 0

    found_name_column = False
    for col in flags["high_cardinality_columns"]:
        if col["name"] == "name":
            assert col["unique"] == 5
            found_name_column = True
            break
    assert found_name_column

    # Колонки с нулями
    assert flags["has_many_zero_values"] is True
    assert len(flags["many_zero_columns"]) > 0

    found_revenue_column = False
    for col in flags["many_zero_columns"]:
        if col["name"] == "revenue":
            assert col["zero_share"] == 1.0
            found_revenue_column = True
            break
    assert found_revenue_column

    # Общая оценка качества
    assert 0.0 <= flags["quality_score"] <= 1.0
    assert flags["quality_score"] < 1.0  # должна снижаться при проблемах


def test_quality_flags_no_issues():
    """Тест с чистыми данными без проблем"""
    # Много строк чтобы избежать ложных срабатываний
    df = pd.DataFrame({
        "id": list(range(1, 101)),  # 100 уникальных ID
        "value": list(range(10, 1010, 10)),  # 100 уникальных значений
        "category": ["A", "B", "C", "D", "E"] * 20,  # низкая кардинальность
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Не должно быть флагов проблем
    assert flags["has_constant_columns"] is False
    assert flags["has_suspicious_id_duplicates"] is False
    assert flags["has_high_cardinality_categoricals"] is False
    assert flags["has_many_zero_values"] is False

    # Высокий скор качества
    assert flags["quality_score"] > 0.7


def test_quality_flags_mixed_scenario():
    """Тест с частичными проблемами"""
    # Балансируем параметры чтобы не было ложных срабатываний
    df = pd.DataFrame({
        "user_id": list(range(1, 11)),  # все ID уникальны
        "status": ["active"] * 10,  # константное значение
        "score": list(range(85, 95)),  # нормальная числовая колонка
        "category": ["A", "A", "B", "B", "C", "C", "D", "D", "A", "B"],  # 40% уникальности
        "zero_col": [0, 0, 1, 2, 3, 0, 1, 2, 0, 3],  # не все нули
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Только константные колонки
    assert flags["has_constant_columns"] is True
    assert flags["has_suspicious_id_duplicates"] is False
    assert flags["has_high_cardinality_categoricals"] is False
    assert flags["has_many_zero_values"] is False

    # Проверяем конкретную проблему
    assert "status" in flags["constant_columns"]
    assert len(flags["constant_columns"]) == 1