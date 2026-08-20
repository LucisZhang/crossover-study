"""ML-32M content-recipe identity + text-template tests (Phase 9, T9-3b).

JVM-free. These are the preregistration §9 "unit tests land WITH that code and
BEFORE any VAL run" gates for the recipe half of the content arm:

* the Amazon recipe hash is byte-identical after ``recipe_hash`` grew its
  optional ``extra`` mapping (recorded prefix ``1f7878ff82bf``);
* the ML-32M ``extra`` block is exactly the preregistered §3(h) dict — any drift
  is a different recipe, and the test says so before an embedding exists;
* the §3(g) text template: ``" ".join`` of title + genres + tags, empties
  skipped, MovieLens year suffix kept.
"""

from __future__ import annotations

import json

import pytest

from batch_recsys_lab.models.minilm_embed import (
    AMAZON_RECIPE,
    ML32M_RECIPE,
    ML32M_RECIPE_EXTRA,
    MODEL_ID,
    RECIPE_ID,
    RECIPE_ID_ML32M,
    build_recipe_text,
    build_recipe_text_ml32m,
    recipe_hash,
)

# The recipe hash recorded in results/runs.jsonl, the demo receipts, the case
# study and the pinned headline. It may never move.
AMAZON_RECIPE_HASH_SHORT = "1f7878ff82bf"
AMAZON_FIELDS = ["title", "brand_norm", "main_category", "features"]

# T9-3b preregistration §3(h), transcribed here INDEPENDENTLY of the source
# module so a silent edit to the module's dict fails this test.
PREREGISTERED_ML32M_EXTRA = {
    "tag_source": "local.silver_ml32m.tags",
    "tag_cutoff": "2022-06-30T23:59:59.999Z",
    "tag_norm": "silver_sanitized|lower|trim",
    "tag_weight": "count_distinct_user_id",
    "tag_order": "weight_desc,tag_asc",
    "tag_top_k": 10,
    "genres_source": "local.gold_ml32m.item_features.genres",
    "genres_order": "as_stored",
    "empty_policy": "skip",
}


# --------------------------------------------------------------------------- #
# Recipe-hash back-compat (§3h: "Amazon's recorded recipe hash 1f7878ff82bf is
# provably unchanged").
# --------------------------------------------------------------------------- #


def test_amazon_recipe_hash_is_the_recorded_one():
    assert recipe_hash(RECIPE_ID, AMAZON_FIELDS, " ", MODEL_ID).startswith(
        AMAZON_RECIPE_HASH_SHORT
    )


def test_extra_none_is_identical_to_the_pre_extra_signature():
    # Explicit extra=None must hash exactly like the 4-argument call: the key
    # enters the canonical JSON ONLY when non-None.
    assert recipe_hash(RECIPE_ID, AMAZON_FIELDS, " ", MODEL_ID, extra=None) == recipe_hash(
        RECIPE_ID, AMAZON_FIELDS, " ", MODEL_ID
    )


def test_amazon_registry_entry_carries_no_extra_and_hashes_to_the_record():
    assert AMAZON_RECIPE.extra is None
    assert AMAZON_RECIPE.hash().startswith(AMAZON_RECIPE_HASH_SHORT)
    assert AMAZON_RECIPE.build_text is build_recipe_text


def test_empty_extra_mapping_is_not_the_same_as_no_extra():
    # {} is a declaration ("this recipe has an extra block, and it is empty"),
    # None is the absence of one. Conflating them would let a future recipe
    # collide with an Amazon-shaped hash.
    assert recipe_hash(RECIPE_ID, AMAZON_FIELDS, " ", MODEL_ID, extra={}) != recipe_hash(
        RECIPE_ID, AMAZON_FIELDS, " ", MODEL_ID
    )


# --------------------------------------------------------------------------- #
# ML-32M recipe identity.
# --------------------------------------------------------------------------- #


def test_ml32m_extra_matches_the_preregistration_verbatim():
    assert ML32M_RECIPE_EXTRA == PREREGISTERED_ML32M_EXTRA
    assert ML32M_RECIPE.extra == PREREGISTERED_ML32M_EXTRA
    assert ML32M_RECIPE.recipe_id == RECIPE_ID_ML32M == "v1_ml32m_title_genres_tags"
    assert ML32M_RECIPE.fields == ("title", "genres", "tags_top10")
    assert ML32M_RECIPE.joiner == " "
    # The SAME model as the Amazon recipe (§3h: same locally cached artifact).
    assert ML32M_RECIPE.model_id == MODEL_ID


def test_ml32m_recipe_hash_is_deterministic_and_distinct():
    expected = recipe_hash(
        RECIPE_ID_ML32M,
        ["title", "genres", "tags_top10"],
        " ",
        MODEL_ID,
        extra=PREREGISTERED_ML32M_EXTRA,
    )
    assert ML32M_RECIPE.hash() == expected
    assert ML32M_RECIPE.hash() != AMAZON_RECIPE.hash()
    # Key order in the mapping must not matter (canonical JSON sorts keys).
    shuffled = dict(reversed(list(PREREGISTERED_ML32M_EXTRA.items())))
    assert (
        recipe_hash(
            RECIPE_ID_ML32M, ["title", "genres", "tags_top10"], " ", MODEL_ID, extra=shuffled
        )
        == expected
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("tag_top_k", 20),  # the barred K sweep
        ("tag_cutoff", "2022-12-31T23:59:59.999Z"),  # a leaked (VAL-end) cutoff
        ("tag_order", "weight_desc,tag_desc"),
        ("tag_weight", "count_rows"),
    ],
)
def test_a_changed_aggregation_rule_changes_the_recipe_hash(key, value):
    """§3(h): "The recipe hash MUST bind the aggregation rule, otherwise this
    specification is not enforced by the artifact identity."""
    mutated = dict(PREREGISTERED_ML32M_EXTRA)
    mutated[key] = value
    assert (
        recipe_hash(
            RECIPE_ID_ML32M, ["title", "genres", "tags_top10"], " ", MODEL_ID, extra=mutated
        )
        != ML32M_RECIPE.hash()
    )


def test_extra_is_json_serializable_for_the_artifact_manifest():
    # The manifest records recipe_extra verbatim; a non-serializable value would
    # only fail at the end of an embedding run.
    assert json.loads(json.dumps(ML32M_RECIPE_EXTRA)) == PREREGISTERED_ML32M_EXTRA


# --------------------------------------------------------------------------- #
# §3(g) text template.
# --------------------------------------------------------------------------- #


def test_ml32m_text_template_is_title_genres_tags_space_joined():
    text = build_recipe_text_ml32m(
        {
            "title": "Toy Story (1995)",
            "genres": ["Adventure", "Animation", "Children"],
            "tags_top10": ["pixar", "funny"],
        }
    )
    # Year suffix KEPT (§3g), single-space joins, no labels, no separators.
    assert text == "Toy Story (1995) Adventure Animation Children pixar funny"


def test_ml32m_text_template_skips_empty_parts():
    assert (
        build_recipe_text_ml32m(
            {"title": "Mystery Film (2023)", "genres": [], "tags_top10": []}
        )
        == "Mystery Film (2023)"
    )
    assert (
        build_recipe_text_ml32m(
            {"title": None, "genres": ["Drama"], "tags_top10": ["noir", ""]}
        )
        == "Drama noir"
    )
    # NULL arrays (item absent from item_features) must not blow up.
    assert (
        build_recipe_text_ml32m({"title": "A", "genres": None, "tags_top10": None}) == "A"
    )


def test_ml32m_text_template_on_a_wholly_empty_row_is_the_empty_string():
    # §3(j)'s coverage threshold counts exactly these rows; the recipe must
    # produce "" rather than a placeholder token.
    assert build_recipe_text_ml32m({"title": None, "genres": [], "tags_top10": []}) == ""


def test_amazon_text_template_is_untouched():
    assert (
        build_recipe_text(
            {
                "title": "Widget",
                "brand_norm": "acme",
                "main_category": "Electronics",
                "features": ["fast", ""],
            }
        )
        == "Widget acme Electronics fast"
    )
