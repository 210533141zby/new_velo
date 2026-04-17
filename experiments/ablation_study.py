from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import nbformat as nbf
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from mainstream_rag_benchmark import (
    BOOTSTRAP_SAMPLES,
    CACHE_ROOT,
    DATASETS,
    EMBEDDING_FALLBACK_MODEL,
    FINAL_TOP_K,
    OUR_RERANK_TOP_K,
    RANDOM_SEED,
    REPO_ROOT,
    RETRIEVAL_TOP_K,
    RRF_K,
    bootstrap_mean_ci,
    bm25_rank,
    build_cache_signature,
    build_relevance_lookup,
    compute_dense_rankings,
    compute_query_profile,
    coverage_ratio,
    ensure_dir,
    ensure_doc_embeddings,
    ensure_query_embeddings,
    extract_identifiers,
    load_public_dataset,
    load_reranker,
    metric_hit_at_1,
    metric_ndcg,
    normalize_scores,
    reciprocal_rank_fusion,
    sample_query_frame,
    slugify,
    tokenize_text,
)

# 这次消融专门围绕正式实验的完整方案展开，目标是回答：
# “如果把某个增强模块拿掉，排序质量会掉多少？”
ABLATION_ROOT = Path('/root/Velo/experiments/消融实验_模块贡献分析')
DATASET_SPEC = DATASETS['du']
SAMPLED_QUERY_COUNT = 1200
FULL_CANDIDATE_K = OUR_RERANK_TOP_K
MAINSTREAM_B_CANDIDATE_K = 20
FIXED_LEXICAL_WEIGHT = 0.36
CJK_FONT_PATH = REPO_ROOT / 'experiments' / '.assets' / 'fonts' / 'SourceHanSansSC-Regular.otf'

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 导出 SVG 时把文字直接转成路径，避免在不同查看器里因为缺字库而显示成方框。
plt.rcParams['svg.fonttype'] = 'path'


def load_cjk_font() -> font_manager.FontProperties | None:
    if not CJK_FONT_PATH.exists():
        return None
    font_manager.fontManager.addfont(str(CJK_FONT_PATH))
    return font_manager.FontProperties(fname=str(CJK_FONT_PATH))


@dataclass(frozen=True)
class AblationConfig:
    key: str
    label: str
    score_mode: str
    use_coverage: bool
    use_identifier_reward: bool
    use_identifier_penalty: bool
    candidate_strategy: str
    candidate_k: int


ABLATION_CONFIGS = (
    AblationConfig(
        key='mainstream_b',
        label='主流方案 B（RRF + Rerank Top20）',
        score_mode='rrf_only',
        use_coverage=False,
        use_identifier_reward=False,
        use_identifier_penalty=False,
        candidate_strategy='mainstream_b',
        candidate_k=MAINSTREAM_B_CANDIDATE_K,
    ),
    AblationConfig(
        key='w_o_adaptive',
        label='去掉自适应权重（固定 lexical/dense 权重）',
        score_mode='fixed',
        use_coverage=True,
        use_identifier_reward=True,
        use_identifier_penalty=True,
        candidate_strategy='mixed',
        candidate_k=FULL_CANDIDATE_K,
    ),
    AblationConfig(
        key='w_o_coverage',
        label='去掉覆盖率奖励',
        score_mode='adaptive',
        use_coverage=False,
        use_identifier_reward=True,
        use_identifier_penalty=True,
        candidate_strategy='mixed',
        candidate_k=FULL_CANDIDATE_K,
    ),
    AblationConfig(
        key='w_o_identifier',
        label='去掉标识符约束',
        score_mode='adaptive',
        use_coverage=True,
        use_identifier_reward=False,
        use_identifier_penalty=False,
        candidate_strategy='mixed',
        candidate_k=FULL_CANDIDATE_K,
    ),
    AblationConfig(
        key='w_o_mixed_pool',
        label='去掉混合候选池',
        score_mode='adaptive',
        use_coverage=True,
        use_identifier_reward=True,
        use_identifier_penalty=True,
        candidate_strategy='topk_scored',
        candidate_k=FULL_CANDIDATE_K,
    ),
    AblationConfig(
        key='full',
        label='完整方案',
        score_mode='adaptive',
        use_coverage=True,
        use_identifier_reward=True,
        use_identifier_penalty=True,
        candidate_strategy='mixed',
        candidate_k=FULL_CANDIDATE_K,
    ),
)


def build_fixed_profile() -> dict[str, float]:
    # 固定权重消融用来回答：如果不做查询自适应，只用一套固定配方会怎样。
    return {
        'lexical_weight': FIXED_LEXICAL_WEIGHT,
        'dense_weight': 1.0 - FIXED_LEXICAL_WEIGHT,
        'token_count': 0.0,
        'has_identifier': 0.0,
    }


def score_candidate(
    config: AblationConfig,
    candidate_index: int,
    adaptive_profile: dict[str, float],
    fixed_profile: dict[str, float],
    normalized_dense: dict[int, float],
    normalized_bm25: dict[int, float],
    normalized_rrf: dict[int, float],
    coverage_map: dict[int, float],
    identifier_map: dict[int, float],
    query_has_identifier: bool,
) -> float:
    if config.score_mode == 'rrf_only':
        return normalized_rrf.get(candidate_index, 0.0)

    profile = adaptive_profile if config.score_mode == 'adaptive' else fixed_profile
    score = (
        profile['dense_weight'] * normalized_dense.get(candidate_index, 0.0)
        + profile['lexical_weight'] * normalized_bm25.get(candidate_index, 0.0)
        + normalized_rrf.get(candidate_index, 0.0) * 0.26
    )
    if config.use_coverage:
        score += coverage_map.get(candidate_index, 0.0) * 0.08
    if config.use_identifier_reward:
        score += identifier_map.get(candidate_index, 0.0) * 0.06
    if config.use_identifier_penalty and query_has_identifier and identifier_map.get(candidate_index, 0.0) == 0.0:
        score -= 0.10
    return score


def build_candidate_pool(
    config: AblationConfig,
    scored_ranked: list[int],
    rrf_ranked: list[int],
) -> list[int]:
    if config.candidate_strategy == 'mainstream_b':
        return rrf_ranked[: config.candidate_k]
    if config.candidate_strategy == 'topk_scored':
        return scored_ranked[: config.candidate_k]

    # 完整方案使用“自适应排序 + RRF 排序”的混合候选池。
    candidate_indices: list[int] = []
    half_k = config.candidate_k // 2 + 4
    for source in (scored_ranked[:half_k], rrf_ranked[:half_k], scored_ranked):
        for index in source:
            if index not in candidate_indices:
                candidate_indices.append(index)
            if len(candidate_indices) >= config.candidate_k:
                return candidate_indices
    return candidate_indices


def rerank_score_map(
    reranker,
    query: str,
    candidate_indices: list[int],
    corpus_texts: list[str],
) -> dict[int, float]:
    if not candidate_indices:
        return {}
    pairs = [(query, corpus_texts[index][:900]) for index in candidate_indices]
    raw_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    return {int(index): float(score) for index, score in zip(candidate_indices, raw_scores)}


def paired_delta_ci(
    query_df: pd.DataFrame,
    config_key: str,
    metric: str,
    full_key: str = 'full',
) -> tuple[float, float, float]:
    pivot = query_df.pivot(index='query_id', columns='config_key', values=metric)
    deltas = pivot[full_key] - pivot[config_key]
    low, high = bootstrap_mean_ci(deltas, samples=BOOTSTRAP_SAMPLES, seed=RANDOM_SEED)
    return float(deltas.mean()), float(low), float(high)


def to_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    headers = '| ' + ' | '.join(columns) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    rows: list[str] = []
    for _, row in df[columns].iterrows():
        values: list[str] = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f'{value:.4f}')
            else:
                values.append(str(value))
        rows.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join([headers, divider] + rows)


def plot_ablation_bars(metrics_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = metrics_df['label'].tolist()
    x = np.arange(len(labels))
    colors = ['#3a6ea5', '#557c55', '#7aa95c', '#9bc53d', '#db9d47', '#2f855a']
    cjk_font = load_cjk_font()
    metric_specs = (
        ('ndcg@10', 'nDCG@10', 'ndcg@10_ci_low', 'ndcg@10_ci_high'),
        ('hit@1', 'Hit@1', 'hit@1_ci_low', 'hit@1_ci_high'),
    )

    for axis, (metric, title, ci_low_key, ci_high_key) in zip(axes, metric_specs):
        values = metrics_df[metric].to_numpy(dtype=np.float32)
        ci_low = metrics_df[ci_low_key].to_numpy(dtype=np.float32)
        ci_high = metrics_df[ci_high_key].to_numpy(dtype=np.float32)
        lower = values - ci_low
        upper = ci_high - values
        axis.bar(x, values, color=colors[: len(labels)], width=0.72)
        axis.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt='none', ecolor='#1f2937', capsize=4, lw=1.1)
        axis.set_xticks(x)
        if cjk_font is not None:
            axis.set_xticklabels(labels, rotation=18, ha='right', fontproperties=cjk_font)
            axis.set_ylabel(title, fontproperties=cjk_font)
            axis.set_title(f'DuRetrieval 去模块消融：{title}', fontproperties=cjk_font)
        else:
            axis.set_xticklabels(labels, rotation=18, ha='right')
            axis.set_ylabel(title)
            axis.set_title(f'DuRetrieval Ablation: {title}')
        axis.set_ylim(0, min(1.0, float(np.max(ci_high) * 1.15)))
        axis.grid(axis='y', alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, format='svg')
    plt.close(figure)


def build_ablation_notebook(path: Path, config: dict[str, Any], metrics_df: pd.DataFrame) -> None:
    metrics_table = to_markdown_table(
        metrics_df,
        ['label', 'candidate_k', 'hit@1', 'ndcg@10', 'hit@1_ci95', 'ndcg@10_ci95'],
    )
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            '# DuRetrieval 去模块消融实验\n\n'
            '这份 Notebook 用来回答一个更细的问题：完整方案的收益究竟来自哪些模块，去掉某个模块后会掉多少。'
        ),
        nbf.v4.new_code_cell(
            'config = ' + json.dumps(config, ensure_ascii=False, indent=2),
            outputs=[nbf.v4.new_output('execute_result', data={'text/plain': json.dumps(config, ensure_ascii=False, indent=2)}, execution_count=1)],
            execution_count=1,
        ),
        nbf.v4.new_markdown_cell(metrics_table),
        nbf.v4.new_markdown_cell('![消融实验核心图](outputs/ablation_core_metrics.svg)'),
    ]
    notebook.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.12'},
    }
    path.write_text(nbf.writes(notebook), encoding='utf-8')


def build_ablation_summary(path: Path, config: dict[str, Any], metrics_df: pd.DataFrame) -> None:
    full = metrics_df[metrics_df['key'] == 'full'].iloc[0]
    no_adaptive = metrics_df[metrics_df['key'] == 'w_o_adaptive'].iloc[0]
    no_coverage = metrics_df[metrics_df['key'] == 'w_o_coverage'].iloc[0]
    no_identifier = metrics_df[metrics_df['key'] == 'w_o_identifier'].iloc[0]
    no_mixed_pool = metrics_df[metrics_df['key'] == 'w_o_mixed_pool'].iloc[0]
    mainstream_b = metrics_df[metrics_df['key'] == 'mainstream_b'].iloc[0]

    metrics_table = to_markdown_table(
        metrics_df,
        ['label', 'candidate_k', 'hit@1', 'ndcg@10', 'hit@1_ci95', 'ndcg@10_ci95'],
    )
    content = f"""# 消融实验说明

## 1. 实验目的

这次消融实验回答的是一个更细的问题：

1. 如果拿掉查询自适应权重，性能会掉多少
2. 如果拿掉覆盖率奖励，性能会掉多少
3. 如果拿掉标识符约束，性能会掉多少
4. 如果拿掉混合候选池，性能会掉多少

## 2. 为什么重做消融协议

前一版消融把 `RRF + 候选池30` 作为累加式对照，结果发现：

1. 单独扩大候选池本身就可能带来很强收益
2. 容易把“候选池大小变化”和“模块本身作用”混在一起

所以这一版改成更稳的“去模块消融”：

1. 以完整方案为中心
2. 每次只拿掉一个模块
3. 其他设置尽量保持不变

这样更适合做模块归因。

## 3. 实验配置

```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

## 4. 核心结果

{metrics_table}

## 5. 结果解读

完整方案相对主流方案 B 的绝对提升为：

- `Hit@1`：`{full['hit@1'] - mainstream_b['hit@1']:.4f}`
- `nDCG@10`：`{full['ndcg@10'] - mainstream_b['ndcg@10']:.4f}`

相对“去掉自适应权重”版本，完整方案的提升为：

- `Hit@1`：`{full['hit@1'] - no_adaptive['hit@1']:.4f}`
- `nDCG@10`：`{full['ndcg@10'] - no_adaptive['ndcg@10']:.4f}`

相对“去掉覆盖率奖励”版本，完整方案的提升为：

- `Hit@1`：`{full['hit@1'] - no_coverage['hit@1']:.4f}`
- `nDCG@10`：`{full['ndcg@10'] - no_coverage['ndcg@10']:.4f}`

相对“去掉标识符约束”版本，完整方案的提升为：

- `Hit@1`：`{full['hit@1'] - no_identifier['hit@1']:.4f}`
- `nDCG@10`：`{full['ndcg@10'] - no_identifier['ndcg@10']:.4f}`

相对“去掉混合候选池”版本，完整方案的提升为：

- `Hit@1`：`{full['hit@1'] - no_mixed_pool['hit@1']:.4f}`
- `nDCG@10`：`{full['ndcg@10'] - no_mixed_pool['ndcg@10']:.4f}`
"""
    path.write_text(content, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='DuRetrieval 去模块消融实验：分析完整检索增强链路中各模块的贡献。')
    parser.add_argument('--output-root', default=str(ABLATION_ROOT))
    parser.add_argument('--embedding-model', default=EMBEDDING_FALLBACK_MODEL)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    outputs_dir = output_root / 'outputs'
    ensure_dir(output_root)
    ensure_dir(outputs_dir)

    embedding_model = args.embedding_model
    reranker = load_reranker()

    print(f'Use embedding model: {embedding_model}')
    print(f'Use dataset: {DATASET_SPEC.label}')
    print('Ablation configs:')
    for item in ABLATION_CONFIGS:
        print(f'  - {item.label}')

    corpus_df, queries_df, qrels_df = load_public_dataset(DATASET_SPEC)
    sampled_queries_df = sample_query_frame(queries_df, qrels_df, sample_size=SAMPLED_QUERY_COUNT, seed=RANDOM_SEED)
    sampled_qrels_df = qrels_df[qrels_df['qid'].astype(str).isin(set(sampled_queries_df['id'].astype(str)))].copy()

    corpus_ids = corpus_df['id'].astype(str).tolist()
    corpus_texts = corpus_df['text'].astype(str).tolist()
    corpus_tokens = [tokenize_text(text) for text in corpus_texts]
    bm25 = BM25Okapi(corpus_tokens)

    doc_embeddings = ensure_doc_embeddings(DATASET_SPEC.name, embedding_model, corpus_ids, corpus_texts)
    query_ids = sampled_queries_df['id'].astype(str).tolist()
    query_texts = sampled_queries_df['text'].astype(str).tolist()
    query_embeddings = ensure_query_embeddings(DATASET_SPEC.name, embedding_model, query_ids, query_texts)
    dense_top_indices, dense_top_scores = compute_dense_rankings(doc_embeddings, query_embeddings, RETRIEVAL_TOP_K)
    relevance_lookup = build_relevance_lookup(sampled_qrels_df)

    per_query_rows: list[dict[str, Any]] = []

    for query_index, row in enumerate(sampled_queries_df.itertuples(index=False)):
        if query_index and query_index % 100 == 0:
            print(f'[{DATASET_SPEC.label}] processed {query_index}/{len(sampled_queries_df)} queries')

        query_id = str(row.id)
        query = str(row.text)
        positives = relevance_lookup[query_id]
        query_tokens = tokenize_text(query)
        query_token_set = set(query_tokens)
        query_identifiers = extract_identifiers(query)
        query_has_identifier = bool(query_identifiers)

        bm25_indices, bm25_scores = bm25_rank(query_tokens, bm25, RETRIEVAL_TOP_K)
        dense_indices = dense_top_indices[query_index].tolist()
        dense_scores = dense_top_scores[query_index].tolist()

        dense_score_map = {int(index): float(score) for index, score in zip(dense_indices, dense_scores)}
        bm25_score_map = {int(index): float(score) for index, score in zip(bm25_indices, bm25_scores)}
        rrf_scores = reciprocal_rank_fusion([dense_indices, bm25_indices], top_k=RETRIEVAL_TOP_K, k=RRF_K)
        fused_candidates = sorted(set(rrf_scores) | set(dense_indices) | set(bm25_indices))

        normalized_dense = normalize_scores({index: dense_score_map.get(index, 0.0) for index in fused_candidates})
        normalized_bm25 = normalize_scores({index: bm25_score_map.get(index, 0.0) for index in fused_candidates})
        normalized_rrf = normalize_scores({index: rrf_scores.get(index, 0.0) for index in fused_candidates})

        adaptive_profile = compute_query_profile(query, bm25)
        fixed_profile = build_fixed_profile()

        coverage_map: dict[int, float] = {}
        identifier_map: dict[int, float] = {}
        for candidate_index in fused_candidates:
            candidate_tokens = set(corpus_tokens[candidate_index])
            coverage_map[candidate_index] = coverage_ratio(query_token_set, candidate_tokens)
            identifier_map[candidate_index] = coverage_ratio(query_identifiers, extract_identifiers(corpus_texts[candidate_index]))

        all_rerank_scores = rerank_score_map(reranker, query, fused_candidates, corpus_texts)
        rrf_ranked = [index for index, _ in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)]

        for config in ABLATION_CONFIGS:
            scored_candidates = {
                candidate_index: score_candidate(
                    config,
                    candidate_index,
                    adaptive_profile,
                    fixed_profile,
                    normalized_dense,
                    normalized_bm25,
                    normalized_rrf,
                    coverage_map,
                    identifier_map,
                    query_has_identifier,
                )
                for candidate_index in fused_candidates
            }
            scored_ranked = [index for index, _ in sorted(scored_candidates.items(), key=lambda item: item[1], reverse=True)]
            candidate_pool = build_candidate_pool(config, scored_ranked, rrf_ranked)
            reranked = sorted(candidate_pool, key=lambda index: all_rerank_scores.get(index, float('-inf')), reverse=True)
            ranked_ids = [corpus_ids[index] for index in reranked[:FINAL_TOP_K]]

            per_query_rows.append(
                {
                    'dataset': DATASET_SPEC.label,
                    'query_id': query_id,
                    'query': query,
                    'config_key': config.key,
                    'label': config.label,
                    'hit@1': metric_hit_at_1(ranked_ids, positives),
                    'ndcg@10': metric_ndcg(ranked_ids, positives, FINAL_TOP_K),
                    'top1_doc_id': ranked_ids[0] if ranked_ids else '',
                    'candidate_pool_size': len(candidate_pool),
                }
            )

    query_df = pd.DataFrame(per_query_rows)
    metric_rows: list[dict[str, Any]] = []

    for config in ABLATION_CONFIGS:
        group = query_df[query_df['config_key'] == config.key]
        hit_low, hit_high = bootstrap_mean_ci(group['hit@1'], samples=BOOTSTRAP_SAMPLES, seed=RANDOM_SEED)
        ndcg_low, ndcg_high = bootstrap_mean_ci(group['ndcg@10'], samples=BOOTSTRAP_SAMPLES, seed=RANDOM_SEED)
        hit_delta_vs_full, hit_delta_low, hit_delta_high = paired_delta_ci(query_df, config.key, 'hit@1')
        ndcg_delta_vs_full, ndcg_delta_low, ndcg_delta_high = paired_delta_ci(query_df, config.key, 'ndcg@10')
        metric_rows.append(
            {
                'key': config.key,
                'label': config.label,
                'hit@1': group['hit@1'].mean(),
                'ndcg@10': group['ndcg@10'].mean(),
                'query_count': int(len(group)),
                'candidate_k': int(group['candidate_pool_size'].mean()),
                'hit@1_ci_low': hit_low,
                'hit@1_ci_high': hit_high,
                'hit@1_ci95': f'[{hit_low:.4f}, {hit_high:.4f}]',
                'ndcg@10_ci_low': ndcg_low,
                'ndcg@10_ci_high': ndcg_high,
                'ndcg@10_ci95': f'[{ndcg_low:.4f}, {ndcg_high:.4f}]',
                'full_minus_config_hit@1': hit_delta_vs_full,
                'full_minus_config_hit@1_ci_low': hit_delta_low,
                'full_minus_config_hit@1_ci_high': hit_delta_high,
                'full_minus_config_ndcg@10': ndcg_delta_vs_full,
                'full_minus_config_ndcg@10_ci_low': ndcg_delta_low,
                'full_minus_config_ndcg@10_ci_high': ndcg_delta_high,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df[
        [
            'key',
            'label',
            'candidate_k',
            'hit@1',
            'ndcg@10',
            'query_count',
            'hit@1_ci_low',
            'hit@1_ci_high',
            'hit@1_ci95',
            'ndcg@10_ci_low',
            'ndcg@10_ci_high',
            'ndcg@10_ci95',
            'full_minus_config_hit@1',
            'full_minus_config_hit@1_ci_low',
            'full_minus_config_hit@1_ci_high',
            'full_minus_config_ndcg@10',
            'full_minus_config_ndcg@10_ci_low',
            'full_minus_config_ndcg@10_ci_high',
        ]
    ]

    config = {
        'random_seed': RANDOM_SEED,
        'dataset': asdict(DATASET_SPEC),
        'sample_queries': SAMPLED_QUERY_COUNT,
        'embedding_model': embedding_model,
        'cache_root': str(CACHE_ROOT),
        'doc_embedding_cache': f"{DATASET_SPEC.name}_{slugify(embedding_model)}_{build_cache_signature(corpus_ids)}_doc_embeddings.npy",
        'query_embedding_cache': f"{DATASET_SPEC.name}_{slugify(embedding_model)}_{build_cache_signature(query_ids)}_query_embeddings.npy",
        'retrieval_top_k': RETRIEVAL_TOP_K,
        'mainstream_b_candidate_k': MAINSTREAM_B_CANDIDATE_K,
        'full_candidate_k': FULL_CANDIDATE_K,
        'final_top_k': FINAL_TOP_K,
        'fixed_lexical_weight': FIXED_LEXICAL_WEIGHT,
        'bootstrap_samples': BOOTSTRAP_SAMPLES,
        'ablation_protocol': 'leave-one-out around the full pipeline, with mainstream B as external reference',
        'ablation_configs': [asdict(item) for item in ABLATION_CONFIGS],
    }

    metrics_df.to_csv(outputs_dir / 'ablation_metrics.csv', index=False)
    query_df.to_json(outputs_dir / 'ablation_per_query.json', orient='records', force_ascii=False, indent=2)
    (outputs_dir / 'ablation_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_ablation_bars(metrics_df, outputs_dir / 'ablation_core_metrics.svg')
    build_ablation_notebook(output_root / '消融实验.ipynb', config, metrics_df)
    build_ablation_summary(output_root / '消融实验说明.md', config, metrics_df)

    print('Ablation complete.')
    print(metrics_df[['label', 'hit@1', 'ndcg@10', 'hit@1_ci95', 'ndcg@10_ci95']].to_string(index=False))


if __name__ == '__main__':
    main()
