from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset


DOMAIN_ENUM = [
    "general",
    "reasoning",
    "code",
    "math",
    "medical",
    "finance",
    "science",
    "cybersecurity",
    "translation",
    "multimodal",
    "vision-language",
    "biology",
    "summarization",
    "legal",
    "unknown",
    "insurance",
    "politics",
    "agent",
    "travel",
    "devops",
    "fashion",
    "table",
    "art",
]

# dataset-level fixed mapping for NON-mmlu_pro datasets
BASE_DATASET_DOMAIN = {
    "gsm8k": ("math", "gsm8k"),
    "math_lvl5": ("math", "math"),
    "arc": ("science", "arc"),
    "gpqa": ("science", "gpqa"),
    "musr": ("reasoning", "musr"),
    "hellaswag": ("reasoning", "commonsense_reasoning"),
    "ifeval": ("general", "instruction_following"),
    "truthfulqa": ("reasoning", "truthfulness"),
    "winogrande": ("reasoning", "commonsense_reasoning"),
}

BBH_TASK_DOMAIN = {
    "boolean_expressions": ("reasoning", "logical_reasoning"),
    "causal_judgement": ("reasoning", "causal_reasoning"),
    "date_understanding": ("reasoning", "temporal_reasoning"),
    "disambiguation_qa": ("reasoning", "disambiguation"),
    "dyck_languages": ("reasoning", "formal_reasoning"),
    "formal_fallacies": ("reasoning", "logical_reasoning"),
    "geometric_shapes": ("math", "geometry"),
    "hyperbaton": ("reasoning", "language_reasoning"),
    "logical_deduction_five_objects": ("reasoning", "logical_deduction"),
    "logical_deduction_seven_objects": ("reasoning", "logical_deduction"),
    "logical_deduction_three_objects": ("reasoning", "logical_deduction"),
    "movie_recommendation": ("general", "recommendation"),
    "multistep_arithmetic_two": ("math", "arithmetic"),
    "navigate": ("travel", "navigation"),
    "object_counting": ("math", "counting"),
    "penguins_in_a_table": ("table", "table_reasoning"),
    "reasoning_about_colored_objects": ("reasoning", "object_reasoning"),
    "ruin_names": ("art", "wordplay"),
    "salient_translation_error_detection": ("translation", "translation_quality"),
    "snarks": ("reasoning", "pragmatics"),
    "sports_understanding": ("general", "sports"),
    "temporal_sequences": ("reasoning", "temporal_reasoning"),
    "tracking_shuffled_objects_five_objects": ("reasoning", "tracking"),
    "tracking_shuffled_objects_seven_objects": ("reasoning", "tracking"),
    "tracking_shuffled_objects_three_objects": ("reasoning", "tracking"),
    "web_of_lies": ("reasoning", "truth_tracking"),
    "word_sorting": ("reasoning", "symbolic_reasoning"),
}

MMLU_SUBJECT_DOMAIN = {
    "abstract_algebra": ("math", "algebra"),
    "anatomy": ("medical", "anatomy"),
    "astronomy": ("science", "astronomy"),
    "business_ethics": ("finance", "business_ethics"),
    "clinical_knowledge": ("medical", "clinical_knowledge"),
    "college_biology": ("biology", "biology"),
    "college_chemistry": ("science", "chemistry"),
    "college_computer_science": ("code", "computer_science"),
    "college_mathematics": ("math", "mathematics"),
    "college_medicine": ("medical", "medicine"),
    "college_physics": ("science", "physics"),
    "computer_security": ("cybersecurity", "computer_security"),
    "conceptual_physics": ("science", "physics"),
    "econometrics": ("finance", "econometrics"),
    "electrical_engineering": ("science", "engineering"),
    "elementary_mathematics": ("math", "elementary_math"),
    "formal_logic": ("reasoning", "formal_logic"),
    "global_facts": ("general", "global_facts"),
    "high_school_biology": ("biology", "biology"),
    "high_school_chemistry": ("science", "chemistry"),
    "high_school_computer_science": ("code", "computer_science"),
    "high_school_european_history": ("general", "history"),
    "high_school_geography": ("general", "geography"),
    "high_school_government_and_politics": ("politics", "government_and_politics"),
    "high_school_macroeconomics": ("finance", "macroeconomics"),
    "high_school_mathematics": ("math", "mathematics"),
    "high_school_microeconomics": ("finance", "microeconomics"),
    "high_school_physics": ("science", "physics"),
    "high_school_psychology": ("general", "psychology"),
    "high_school_statistics": ("math", "statistics"),
    "high_school_us_history": ("general", "history"),
    "high_school_world_history": ("general", "history"),
    "human_aging": ("medical", "human_aging"),
    "human_sexuality": ("medical", "human_sexuality"),
    "international_law": ("legal", "international_law"),
    "jurisprudence": ("legal", "jurisprudence"),
    "logical_fallacies": ("reasoning", "logical_fallacies"),
    "machine_learning": ("code", "machine_learning"),
    "management": ("finance", "management"),
    "marketing": ("finance", "marketing"),
    "medical_genetics": ("medical", "medical_genetics"),
    "miscellaneous": ("general", "miscellaneous"),
    "moral_disputes": ("general", "moral_reasoning"),
    "moral_scenarios": ("general", "moral_reasoning"),
    "nutrition": ("medical", "nutrition"),
    "philosophy": ("general", "philosophy"),
    "prehistory": ("general", "history"),
    "professional_accounting": ("finance", "accounting"),
    "professional_law": ("legal", "professional_law"),
    "professional_medicine": ("medical", "professional_medicine"),
    "professional_psychology": ("general", "psychology"),
    "public_relations": ("finance", "public_relations"),
    "security_studies": ("politics", "security_studies"),
    "sociology": ("general", "sociology"),
    "us_foreign_policy": ("politics", "foreign_policy"),
    "virology": ("biology", "virology"),
    "world_religions": ("general", "religion"),
}

MMLU_PRO_CATEGORY_TO_DOMAIN = {
    "biology": ("biology", "biology"),
    "business": ("finance", "business"),
    "chemistry": ("science", "chemistry"),
    "computer science": ("code", "computer_science"),
    "economics": ("finance", "economics"),
    "engineering": ("science", "engineering"),
    "health": ("medical", "health"),
    "history": ("general", "history"),
    "law": ("legal", "law"),
    "math": ("math", "math"),
    "philosophy": ("general", "philosophy"),
    "physics": ("science", "physics"),
    "psychology": ("general", "psychology"),
    "other": ("general", "other"),
}


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def normalize_text(s: Any) -> str:
    s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_lower_text(s: Any) -> str:
    return normalize_text(s).lower()


def extract_mmlu_subject(dataset_name: str) -> str:
    s = normalize_text(dataset_name)
    m = re.match(r"harness_hendrycksTest_(.*)_5$", s)
    return m.group(1) if m else s


def extract_bbh_task(dataset_name: str) -> str:
    s = normalize_text(dataset_name)
    m = re.match(r"bbh_(.*)$", s)
    return m.group(1) if m else s


def extract_mathlvl5_subdomain(dataset_name: str) -> str:
    s = normalize_text(dataset_name)
    m = re.match(r"math_(.*)_hard$", s)
    return m.group(1) if m else "math"


def deterministic_label_non_mmlu_pro(row: pd.Series) -> Tuple[str, str, str]:
    ds = normalize_text(row["dataset"])
    dataset_name = normalize_text(row.get("dataset_name", ""))

    if ds == "mmlu_pro":
        return "ignored", "ignored", "ignored_mmlu_pro"

    if ds in BASE_DATASET_DOMAIN:
        domain, sub = BASE_DATASET_DOMAIN[ds]
        if ds == "math_lvl5":
            return domain, extract_mathlvl5_subdomain(dataset_name), "dataset_fixed"
        return domain, sub, "dataset_fixed"

    if ds == "bbh":
        task = extract_bbh_task(dataset_name)
        if task in BBH_TASK_DOMAIN:
            domain, sub = BBH_TASK_DOMAIN[task]
            return domain, sub, "bbh_task_rule"
        return "reasoning", task, "bbh_default"

    if ds == "mmlu":
        subject = extract_mmlu_subject(dataset_name)
        if subject in MMLU_SUBJECT_DOMAIN:
            domain, sub = MMLU_SUBJECT_DOMAIN[subject]
            return domain, sub, "mmlu_subject_rule"
        return "general", subject, "mmlu_default"

    return "unknown", "unmapped", "fallback_unknown"


def classify_non_mmlu_pro(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["dataset"].astype(str).str.strip().ne("mmlu_pro")].copy()

    out["domain"] = ""
    out["sub_domain"] = ""
    out["label_source"] = ""

    for idx, row in out.iterrows():
        domain, sub_domain, source = deterministic_label_non_mmlu_pro(row)
        out.at[idx, "domain"] = domain
        out.at[idx, "sub_domain"] = sub_domain
        out.at[idx, "label_source"] = source

    return out


def load_hf_mmlu_pro_df(split_preference: Optional[List[str]] = None) -> pd.DataFrame:
    if split_preference is None:
        split_preference = ["test", "validation", "dev", "train"]

    ds = None
    used_split = None
    last_err = None

    for sp in split_preference:
        try:
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split=sp)
            used_split = sp
            break
        except Exception as e:
            last_err = e

    if ds is None:
        raise RuntimeError(f"Failed to load TIGER-Lab/MMLU-Pro. Last error: {last_err}")

    hf = ds.to_pandas()
    print(f"[INFO] Loaded TIGER-Lab/MMLU-Pro split={used_split}, rows={len(hf)}")
    print(f"[INFO] HF columns: {list(hf.columns)}")

    category_col = next((c for c in ["category", "subject", "domain"] if c in hf.columns), None)
    if category_col is None:
        raise ValueError("Cannot find category-like column in MMLU-Pro dataset.")

    out = pd.DataFrame({
        "dataset": "mmlu_pro",
        "mmlu_pro_category": hf[category_col].astype(str).map(normalize_lower_text),
    })

    out["domain"] = ""
    out["sub_domain"] = ""
    out["label_source"] = "hf_mmlu_pro_category"

    for idx, row in out.iterrows():
        cat = row["mmlu_pro_category"]
        if cat in MMLU_PRO_CATEGORY_TO_DOMAIN:
            dom, sub = MMLU_PRO_CATEGORY_TO_DOMAIN[cat]
        else:
            dom = "general"
            sub = re.sub(r"\s+", "_", cat) if cat else "other"
        out.at[idx, "domain"] = dom
        out.at[idx, "sub_domain"] = sub

    return out


def build_paper_mapping_table() -> pd.DataFrame:
    rows = []

    for ds, (domain, sub_domain) in sorted(BASE_DATASET_DOMAIN.items()):
        rows.append({
            "scope": "dataset",
            "match_key": ds,
            "domain": domain,
            "sub_domain": sub_domain,
            "method": "fixed_rule",
        })

    for task, (domain, sub) in sorted(BBH_TASK_DOMAIN.items()):
        rows.append({
            "scope": "bbh_task",
            "match_key": task,
            "domain": domain,
            "sub_domain": sub,
            "method": "fixed_rule",
        })

    for subject, (domain, sub) in sorted(MMLU_SUBJECT_DOMAIN.items()):
        rows.append({
            "scope": "mmlu_subject",
            "match_key": subject,
            "domain": domain,
            "sub_domain": sub,
            "method": "fixed_rule",
        })

    rows.append({
        "scope": "dataset",
        "match_key": "mmlu_pro",
        "domain": "category_mapped_from_hf",
        "sub_domain": "category_mapped_from_hf",
        "method": "hf_source_direct",
    })

    for category, (domain, sub) in sorted(MMLU_PRO_CATEGORY_TO_DOMAIN.items()):
        rows.append({
            "scope": "mmlu_pro_category",
            "match_key": category,
            "domain": domain,
            "sub_domain": sub,
            "method": "hf_category_mapping",
        })

    return pd.DataFrame(rows)


def estimate_costs(df_csv: pd.DataFrame, df_hf_mmlu_pro: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    total_items = len(df_csv[df_csv["dataset"].astype(str).str.strip().ne("mmlu_pro")]) + len(df_hf_mmlu_pro)
    cost_rows = [{
        "scenario": "no_api_non_mmlu_pro_csv_plus_hf_mmlu_pro",
        "items": total_items,
        "input_tokens_est": 0,
        "output_tokens_est": 0,
        "input_cost_usd_est": 0.0,
        "output_cost_usd_est": 0.0,
        "total_cost_usd_est": 0.0,
    }]
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(outdir / "gpt4o_mini_cost_estimates.csv", index=False)

    dataset_breakdown = pd.DataFrame([
        {
            "dataset": "csv_non_mmlu_pro",
            "items": len(df_csv[df_csv["dataset"].astype(str).str.strip().ne("mmlu_pro")]),
            "content_tokens_est_sum": 0,
            "content_tokens_est_mean": 0,
        },
        {
            "dataset": "hf_mmlu_pro",
            "items": len(df_hf_mmlu_pro),
            "content_tokens_est_sum": 0,
            "content_tokens_est_mean": 0,
        },
    ])
    dataset_breakdown.to_csv(outdir / "gpt4o_mini_cost_estimates_by_dataset.csv", index=False)
    return cost_df


def write_summaries(df_all: pd.DataFrame, outdir: Path) -> None:
    df_valid = df_all[df_all["domain"] != "unknown"].copy()

    domain_summary = (
        df_valid.groupby("domain")
        .agg(
            item_count=("dataset", "count"),
            datasets=("dataset", lambda s: ", ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(["item_count", "domain"], ascending=[False, True])
    )
    domain_summary.to_csv(outdir / "domain_item_summary.csv", index=False)

    dataset_domain_summary = (
        df_valid.groupby(["dataset", "domain"])
        .size()
        .reset_index(name="item_count")
        .sort_values(["dataset", "item_count", "domain"], ascending=[True, False, True])
    )
    dataset_domain_summary.to_csv(outdir / "dataset_domain_item_summary.csv", index=False)

    subdomain_summary = (
        df_valid.groupby(["domain", "sub_domain"])
        .size()
        .reset_index(name="item_count")
        .sort_values(["domain", "item_count", "sub_domain"], ascending=[True, False, True])
    )
    subdomain_summary.to_csv(outdir / "domain_subdomain_item_summary.csv", index=False)

    mmlu_pro_summary = (
        df_valid[df_valid["dataset"].astype(str).eq("mmlu_pro")]
        .groupby(["mmlu_pro_category", "domain", "sub_domain"])
        .size()
        .reset_index(name="item_count")
        .sort_values(["item_count", "mmlu_pro_category"], ascending=[False, True])
    )
    mmlu_pro_summary.to_csv(outdir / "mmlu_pro_category_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df_csv = pd.read_csv(args.input_csv)
    if "dataset" not in df_csv.columns:
        raise ValueError("Input CSV must contain a 'dataset' column.")
    if "dataset_name" not in df_csv.columns:
        df_csv["dataset_name"] = ""

    # 1) CSV 中除 mmlu_pro 之外的数据
    df_non_mmlu_pro = classify_non_mmlu_pro(df_csv)

    # 2) HF 原始 mmlu_pro
    df_hf_mmlu_pro = load_hf_mmlu_pro_df()

    # 3) 合并用于统计
    common_cols = sorted(set(df_non_mmlu_pro.columns).union(set(df_hf_mmlu_pro.columns)))
    for col in common_cols:
        if col not in df_non_mmlu_pro.columns:
            df_non_mmlu_pro[col] = ""
        if col not in df_hf_mmlu_pro.columns:
            df_hf_mmlu_pro[col] = ""

    df_all = pd.concat(
        [df_non_mmlu_pro[common_cols], df_hf_mmlu_pro[common_cols]],
        ignore_index=True,
    )

    # 4) 只写汇总，不写 items_with_final_domains.csv
    write_summaries(df_all, outdir)
    build_paper_mapping_table().to_csv(outdir / "paper_domain_mapping_table.csv", index=False)
    estimate_costs(df_csv, df_hf_mmlu_pro, outdir)

    print(f"Saved outputs to: {outdir}")
    print(f"CSV non-mmlu_pro rows used: {len(df_non_mmlu_pro)}")
    print(f"HF mmlu_pro rows used: {len(df_hf_mmlu_pro)}")
    print("Generated:")
    print("- domain_item_summary.csv")
    print("- dataset_domain_item_summary.csv")
    print("- domain_subdomain_item_summary.csv")
    print("- mmlu_pro_category_summary.csv")
    print("- paper_domain_mapping_table.csv")
    print("- gpt4o_mini_cost_estimates.csv")
    print("- gpt4o_mini_cost_estimates_by_dataset.csv")
    print("Not generated:")
    print("- items_with_final_domains.csv")


if __name__ == "__main__":
    main()