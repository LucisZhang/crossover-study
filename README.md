# crossover-study
Crossover Study — when does personalization beat popularity?
> 研究个性化推荐何时真正胜过热门推荐。

Crossover Study is a traceable recommender-systems experiment over 43.9M Amazon Electronics reviews, built with Spark, Iceberg, full-catalog ranking, and bootstrap confidence intervals.
> 这是一个可追溯的推荐系统实验。结论对应固定数据快照、配置与 append-only 运行记录。

**Research question.** How much user history is needed before personalized ranking beats trailing-12-month popularity? The Amazon Electronics result is a null: no history-depth crossover appears, and pure personalization does not beat the popularity baseline in any segment.
> 用户需要积累多少历史，个性化排序才会优于过去 12 个月的热门推荐？Amazon Electronics 的答案是未观察到 crossover，纯个性化在各深度段都没有胜出。

## Quickstart · 快速开始

```bash
uv sync
make demo-offline-check   # verify that the static exhibit makes no external requests
make reproduce-headline   # rerun the headline evaluation on the pinned Iceberg snapshot
make eval                 # run configured evaluations and append to results/runs.jsonl
uv run pytest             # metric math, contracts, and temporal-split tests
```

The headline evaluation reproduced `byte_exact` on both the original and a deliberately churned warehouse (records `20260807T153823Z-9a9fb4c` and `20260807T164622Z-3e2c665`).
> 主结果在原始数据仓库与人为变更后的数据仓库中均达到 `byte_exact`。

## Findings · 核心发现

### Amazon null · 零结果

The original ALS, item-kNN, and pure-content arms lose to trailing-12-month popularity in every history segment, with every 95% confidence interval excluding zero. Phase 8's recency-matched rematch still finds no crossover: ALS-decay's `20+` NDCG@10 interval straddles zero while its Recall@20 guard is significantly negative, and item-kNN-t12m remains negative at depth level (paired-delta records `20260818T064002Z-56d871c` and `20260818T064207Z-56d871c`).
> 原始 ALS、item-kNN 与纯 content retrieval 在各段都落后，95% CI 均不含 0。Phase 8 对齐 recency 后仍无 crossover。ALS-decay 在 `20+` 的 NDCG@10 CI 跨 0，但 Recall@20 guard 显著为负。

### Catalog turnover mechanism · 换血机制

The measured catalog turnover mechanism explains the null: 41.11% of 2023 TEST ground-truth purchase mass lands on items with TRAIN support of 0–4. Within that mass, 34.54% sits on zero-support items a TRAIN-frozen factor model cannot represent, setting a 65.5% Recall@K ceiling; requiring core support of at least 5 lowers the ceiling to 58.9% (run `20260817T095926Z-633d454`).
> catalog churn（目录换血）使 41.11% 的 TEST 购买落在 TRAIN support 为 0–4 的商品上。其中 34.54% 完全没有 support，TRAIN-frozen factor model 的 Recall@K 上限因此是 65.5%；若要求 support ≥5，上限为 58.9%。

### Small effective arm · 小幅收益

The only effective arm is a blend: trailing-12-month popularity re-ranked by a MiniLM content score at α=0.3. It gains +0.000322 NDCG@10 over popularity alone, about +6% relative, with a paired confidence interval excluding zero (runs `20260807T055333Z-c320c79` / `20260818T181443Z-6744efc` versus `20260805T172047Z-035042b`).
> 唯一胜出的方案是在热门推荐上加入 α=0.3 的 MiniLM 重排。NDCG@10 增加 0.000322，相对增幅约 6%，收益小但置信区间不含 0。

### Routing result · 路由结论

The fitted Amazon routing policy is `n*=∞`: every finite history-depth threshold scores worse than sending every user to the blend. The optimal routing is therefore no routing (policy grid record `20260808T030659Z-43c90c8`).
> Amazon 的拟合结果是 `n*=∞`。所有有限阈值都更差，因此最佳策略是不按历史深度分流，全部使用 blend。

## Phase 9 · 对照实验

Phase 9 repeats the test on ML-32M, where measured churn is 6.40% rather than Amazon Electronics' 41.11%. In this lower-churn regime, the pre-registered primary NDCG@10 family finds a crossover: item-kNN-t12m beats trailing popularity from `n*=20` upward, with 95% confidence intervals excluding zero and Benjamini-Hochberg significance at FDR 0.05 (record `results/confirmatory_ml32m_test.json`; source runs `20260820T221701Z-20d8ff9` / `20260820T221055Z-20d8ff9`).
> ML-32M 的 churn 只有 6.40%。同一套 pre-registered（预注册）检验在 `n*=20` 起观察到 item-kNN-t12m 胜出。三个深桶的 95% CI 均不含 0，经 Benjamini-Hochberg 校正后仍显著。

Here `n*=20` means the first confirmatory history bucket, 20–49 TRAIN interactions, where the personalized arm wins; it is not the separately fitted routing threshold. The VAL-fitted ML-32M routing threshold is `n*=100`.
> `n*=20` 指首个通过 confirmatory 检验的 20–49 TRAIN interactions 分桶，不是 routing threshold。VAL 拟合的路由阈值是 `n*=100`。

## Limitations · 局限

Five recency-matched Amazon regime cells show item-kNN edging popularity, but they come from roughly 80 uncorrected tests, where about four false positives are expected at α=0.05. They remain a mechanism footnote, not a crossover result, and the cells depend on ground-truth item properties unavailable to a serving-time router.
> Amazon 的 5 个局部胜出来自约 80 次未校正检验，按 α=0.05 预计约有 4 个假阳性。这些结果只支持机制解释，也不能直接用于线上路由。

The ML-32M contrast supports a regime difference without identifying a cause. MovieLens changes domain, density, feedback type, and timestamp semantics at once; its timestamps record rating entry on a backfilled catalog. Recall@20 does not corroborate the NDCG@10 win, cold users lose significantly, and popularity still wins globally.
> ML-32M 只能说明低 churn 场景存在对照结果，不能识别因果。Recall@20 未重复 NDCG@10 的胜出，冷启动用户的指标显著下降，全局结果仍由热门推荐领先。

All reported metrics are offline. Nothing here is an A/B result, and the measurements do not establish CTR, conversion, or revenue lift; reviews are treated as positive interactions even though feedback is missing-not-at-random and already popularity-biased. No counterfactual correction is applied.
> 所有指标都来自离线评估，这些结果都不是 A/B 实验结论，也不代表 CTR、转化率或收入变化。评论被当作正反馈，数据本身带有选择偏差与热门偏差，也没有做反事实校正。

The 5-core filter inflates absolute NDCG@10 by 1.20× for the measured popularity baseline. Results cover one Amazon category and one snapshot ending in 2023-09, so absolute metrics and thresholds do not transfer without re-measurement.
> 5-core 过滤让热门基线的 NDCG@10 绝对值放大到 1.20 倍。结论只覆盖一个 Amazon 类别和一个截至 2023-09 的快照，跨数据集必须重新测量。

No serving system is included: there is no service, latency SLA, or throughput target. Spark runs on one node; distributed behavior, concurrent Iceberg writers, catalog services, and object-store operation were not measured.
> 仓库不包含线上推荐服务，也没有 latency SLA 或吞吐目标。Spark 只在单机测过，分布式行为与并发 Iceberg 写入也没有实测。

## Where the details live

Engineering decisions and phase logs live in [docs/engineering-log/](docs/engineering-log/).
> 工程决策与阶段日志在 docs/engineering-log/。

- `docs/case_study.md` — the full case study, every claim labeled with its evidence class and its `results/runs.jsonl` run ID.
- `demo/` — the static offline exhibit: crossover explorer, regime map, data-quality dashboard, and receipts drawer.
- `docs/engineering-log/EXPERIMENT_LOG.md` — append-only hypothesis → result → verdict entries, including failures and supersessions.
- `results/runs.jsonl` — append-only run records with config hashes, git SHAs, dataset manifest hashes, Iceberg snapshot IDs, and seeds.
- `docs/engineering-log/UPGRADE_PLAN.md` — scope and phase order.

## Provenance and license

This lab grew out of a five-person CISC3018 course project at the University of Macau in Fall 2025 that built a recommendation bot on a 1M-interaction sample. The owner's primary individual contribution there was presentation material.
> 本实验源于澳门大学 CISC3018 的五人课程项目。该项目基于 100 万条交互构建推荐机器人，作者的主要个人贡献是演示材料。

Nothing from the course project ships here, and none of its code, models, or results is presented as individual work. The current pipeline, contracts, models, evaluation harness, and demo were designed and built solo from the raw public dataset with a separate verification chain.
> 课程项目的代码、模型和结果均未作为个人成果使用。当前 pipeline、contracts、models、evaluation harness 与 demo 由作者基于公开原始数据独立完成。

Dataset: Amazon Reviews 2023 (McAuley Lab, UCSD), research-use terms, never redistributed; cite Hou et al. 2024, *Bridging Language and Items for Retrieval and Recommendation*.
