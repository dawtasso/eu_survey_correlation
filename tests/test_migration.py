"""Tests for match_id_utils — deterministic IDs, vote mapping, migration."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "scripts"))

from match_id_utils import (
    build_main_vote_map,
    build_migration_mapping,
    build_vote_to_procedure_map,
    decode_match_key,
    make_match_id,
    make_match_key,
)


# ── Fixtures ──────────────────────────────────────────────────────────


VOTES_HEADER = [
    "id", "timestamp", "display_title", "reference", "description",
    "amendment_subject", "amendment_number", "is_main", "procedure_reference",
    "procedure_title", "procedure_type", "procedure_stage",
    "count_for", "count_against", "count_abstention", "count_did_not_vote",
    "result", "texts_adopted_reference",
]


def _write_votes_csv(rows: list[list], tmp_path: Path) -> Path:
    path = tmp_path / "votes.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(VOTES_HEADER)
        w.writerows(rows)
    return path


@pytest.fixture
def sample_votes_csv(tmp_path: Path) -> Path:
    """A minimal votes.csv with known data."""
    rows = [
        # Amendment vote (is_main=FALSE)
        [108530, "18/07/2019 11:30", "Venezuela", "RC-B9-0006/2019",
         "§ 2/1", "", "", "FALSE", "2019/2730(RSP)", "Venezuela", "RSP", "",
         506, 58, 70, 117, "", "P9_TA(2019)0007"],
        # Main vote for same procedure
        [108532, "18/07/2019 11:30", "Venezuela", "RC-B9-0006/2019",
         "Résolution", "", "", "TRUE", "2019/2730(RSP)", "Venezuela", "RSP", "",
         455, 85, 105, 106, "", "P9_TA(2019)0007"],
        # Main vote for another procedure
        [153304, "14/03/2023 12:15", "Emissions reduction", "A9-0163/2022",
         "Résolution", "", "", "TRUE", "2021/0200(COD)", "Emissions reduction", "COD", "",
         400, 100, 50, 200, "", ""],
        # Amendment for same procedure
        [153300, "14/03/2023 12:10", "Emissions reduction", "A9-0163/2022",
         "Am 1", "", "", "FALSE", "2021/0200(COD)", "Emissions reduction", "COD", "",
         350, 150, 40, 210, "", ""],
        # Vote with no procedure_reference
        [999999, "01/01/2020 10:00", "Procedural", "", "", "", "", "FALSE",
         "", "Procedural", "", "", 100, 50, 10, 590, "", ""],
    ]
    return _write_votes_csv(rows, tmp_path)


# ── match_id generation ──────────────────────────────────────────────


class TestMakeMatchId:
    def test_deterministic(self):
        id1 = make_match_id("T13", "ebs_509_volume_B.xlsx", "2021/0200(COD)")
        id2 = make_match_id("T13", "ebs_509_volume_B.xlsx", "2021/0200(COD)")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = make_match_id("T13", "ebs_509_volume_B.xlsx", "2021/0200(COD)")
        id2 = make_match_id("T14", "ebs_509_volume_B.xlsx", "2021/0200(COD)")
        id3 = make_match_id("T13", "other_file.xlsx", "2021/0200(COD)")
        id4 = make_match_id("T13", "ebs_509_volume_B.xlsx", "2022/0100(COD)")
        assert len({id1, id2, id3, id4}) == 4

    def test_length_is_16(self):
        mid = make_match_id("T13", "ebs_509_volume_B.xlsx", "2021/0200(COD)")
        assert len(mid) == 16

    def test_hex_chars_only(self):
        mid = make_match_id("T13", "ebs_509_volume_B.xlsx", "2021/0200(COD)")
        assert all(c in "0123456789abcdef" for c in mid)

    def test_special_characters_in_question_id(self):
        mid = make_match_id("QB4.5", "file.xlsx", "2021/0200(COD)")
        assert len(mid) == 16
        assert mid == make_match_id("QB4.5", "file.xlsx", "2021/0200(COD)")


class TestMatchKey:
    def test_make_and_decode_roundtrip(self):
        key = make_match_key("T13", "ebs_509_volume_B.xlsx", "2021/0200(COD)")
        assert key == "T13::ebs_509_volume_B.xlsx::2021/0200(COD)"
        q, f, p = decode_match_key(key)
        assert q == "T13"
        assert f == "ebs_509_volume_B.xlsx"
        assert p == "2021/0200(COD)"

    def test_decode_invalid_key(self):
        with pytest.raises(ValueError, match="Invalid match_key"):
            decode_match_key("only_two::parts")

    def test_decode_too_many_parts(self):
        with pytest.raises(ValueError, match="expected 3 parts"):
            decode_match_key("a::b::c::d")


# ── Vote mapping ──────────────────────────────────────────────────────


class TestBuildVoteToProcedureMap:
    def test_maps_all_votes_with_procedure(self, sample_votes_csv):
        mapping = build_vote_to_procedure_map(sample_votes_csv)
        assert mapping[108530] == "2019/2730(RSP)"
        assert mapping[108532] == "2019/2730(RSP)"
        assert mapping[153304] == "2021/0200(COD)"
        assert mapping[153300] == "2021/0200(COD)"
        # Vote 999999 has no procedure_reference → excluded
        assert 999999 not in mapping

    def test_count(self, sample_votes_csv):
        mapping = build_vote_to_procedure_map(sample_votes_csv)
        assert len(mapping) == 4


class TestBuildMainVoteMap:
    def test_picks_main_vote_only(self, sample_votes_csv):
        mapping = build_main_vote_map(sample_votes_csv)
        assert mapping["2019/2730(RSP)"] == 108532  # not 108530
        assert mapping["2021/0200(COD)"] == 153304  # not 153300

    def test_count(self, sample_votes_csv):
        mapping = build_main_vote_map(sample_votes_csv)
        assert len(mapping) == 2

    def test_picks_highest_id_on_duplicate_main(self, tmp_path):
        rows = [
            [1, "01/01/2020", "A", "R1", "", "", "", "TRUE", "2020/0001(COD)",
             "Title", "COD", "", 100, 50, 10, 50, "", ""],
            [2, "01/01/2020", "A", "R2", "", "", "", "TRUE", "2020/0001(COD)",
             "Title", "COD", "", 100, 50, 10, 50, "", ""],
        ]
        csv_path = _write_votes_csv(rows, tmp_path)
        mapping = build_main_vote_map(csv_path)
        # Should pick highest id (2) when multiple is_main=True
        assert mapping["2020/0001(COD)"] == 2


# ── Migration mapping ────────────────────────────────────────────────


class TestBuildMigrationMapping:
    def test_basic_mapping(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        old_rows = [
            {
                "match_id": "T13_153304_0",
                "question_id": "T13",
                "survey_file": "ebs_509_volume_B.xlsx",
                "vote_id": 153304,
            },
            {
                "match_id": "QA23_108532_1",
                "question_id": "QA23",
                "survey_file": "ebs_520_volume_B.xlsx",
                "vote_id": 108532,
            },
        ]

        id_map, key_map, missing, dupes = build_migration_mapping(old_rows, vote_to_proc)

        assert len(id_map) == 2
        assert len(missing) == 0
        assert len(dupes) == 0

        # Verify new IDs are deterministic
        expected_0 = make_match_id("T13", "ebs_509_volume_B.xlsx", "2021/0200(COD)")
        assert id_map["T13_153304_0"] == expected_0

        expected_1 = make_match_id("QA23", "ebs_520_volume_B.xlsx", "2019/2730(RSP)")
        assert id_map["QA23_108532_1"] == expected_1

    def test_missing_vote_id(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        old_rows = [
            {
                "match_id": "T13_999999_0",
                "question_id": "T13",
                "survey_file": "file.xlsx",
                "vote_id": 999999,  # has no procedure_reference
            },
        ]

        id_map, key_map, missing, dupes = build_migration_mapping(old_rows, vote_to_proc)
        assert len(id_map) == 0
        assert len(missing) == 1
        assert missing[0]["match_id"] == "T13_999999_0"

    def test_no_collisions(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        old_rows = [
            {"match_id": f"Q{i}_{153304}_{i}", "question_id": f"Q{i}",
             "survey_file": f"file_{i}.xlsx", "vote_id": 153304}
            for i in range(50)
        ]

        id_map, _, _, _ = build_migration_mapping(old_rows, vote_to_proc)
        new_ids = list(id_map.values())
        assert len(new_ids) == len(set(new_ids)), "Collision detected in new match_ids"

    def test_count_preservation(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        old_rows = [
            {"match_id": f"T13_{153304}_{i}", "question_id": "T13",
             "survey_file": f"file_{i}.xlsx", "vote_id": 153304}
            for i in range(10)
        ]

        id_map, key_map, missing, dupes = build_migration_mapping(old_rows, vote_to_proc)
        assert len(id_map) + len(missing) + len(dupes) == len(old_rows)


# ── Dry-run simulation with mock Supabase data ───────────────────────


class TestDryRunSimulation:
    """Simulate migration with hardcoded data resembling real Supabase rows."""

    MOCK_SUPABASE_ROWS = [
        {
            "match_id": "T13_153304_0",
            "question_id": "T13",
            "survey_file": "ebs_509_volume_B.xlsx",
            "vote_id": 153304,
            "admin_validated": True,
            "similarity_score": 0.665,
        },
        {
            "match_id": "QA23_108532_5",
            "question_id": "QA23",
            "survey_file": "ebs_520_volume_B.xlsx",
            "vote_id": 108532,
            "admin_validated": False,
            "similarity_score": 0.660,
        },
        {
            "match_id": "QB6_153304_12",
            "question_id": "QB6",
            "survey_file": "ebs_521_volume_B.xlsx",
            "vote_id": 153304,
            "admin_validated": None,
            "similarity_score": 0.679,
        },
    ]

    def test_all_rows_get_new_ids(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        id_map, key_map, missing, dupes = build_migration_mapping(
            self.MOCK_SUPABASE_ROWS, vote_to_proc
        )
        assert len(missing) == 0
        assert len(dupes) == 0
        assert len(id_map) == 3

    def test_admin_validated_survives(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        id_map, _, _, _ = build_migration_mapping(
            self.MOCK_SUPABASE_ROWS, vote_to_proc
        )
        # Simulate applying the migration: admin_validated stays the same
        migrated = []
        for row in self.MOCK_SUPABASE_ROWS:
            old_id = row["match_id"]
            if old_id in id_map:
                new_row = {**row, "match_id": id_map[old_id]}
                migrated.append(new_row)

        assert len(migrated) == 3
        assert migrated[0]["admin_validated"] is True
        assert migrated[1]["admin_validated"] is False
        assert migrated[2]["admin_validated"] is None

    def test_no_duplicate_new_ids(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        id_map, _, _, _ = build_migration_mapping(
            self.MOCK_SUPABASE_ROWS, vote_to_proc
        )
        new_ids = list(id_map.values())
        assert len(new_ids) == len(set(new_ids))

    def test_match_keys_are_readable(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        _, key_map, _, _ = build_migration_mapping(
            self.MOCK_SUPABASE_ROWS, vote_to_proc
        )
        for old_id, key in key_map.items():
            q, f, p = decode_match_key(key)
            assert q != ""
            assert f != ""
            assert p != ""


# ── Collision resolution ──────────────────────────────────────────────


class TestCollisionResolution:
    def test_keeps_validated_true_over_none(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        # Same question+file, different vote_ids for same procedure
        old_rows = [
            {"match_id": "T13_153304_0", "question_id": "T13",
             "survey_file": "file.xlsx", "vote_id": 153304,
             "admin_validated": None, "similarity_score": 0.9},
            {"match_id": "T13_153300_1", "question_id": "T13",
             "survey_file": "file.xlsx", "vote_id": 153300,
             "admin_validated": True, "similarity_score": 0.5},
        ]
        id_map, _, _, dupes = build_migration_mapping(old_rows, vote_to_proc)
        assert len(id_map) == 1
        assert len(dupes) == 1
        # Winner should be the validated=True row
        assert "T13_153300_1" in id_map
        assert dupes[0]["match_id"] == "T13_153304_0"

    def test_keeps_higher_score_as_tiebreaker(self, sample_votes_csv):
        vote_to_proc = build_vote_to_procedure_map(sample_votes_csv)
        old_rows = [
            {"match_id": "T13_153304_0", "question_id": "T13",
             "survey_file": "file.xlsx", "vote_id": 153304,
             "admin_validated": True, "similarity_score": 0.9},
            {"match_id": "T13_153300_1", "question_id": "T13",
             "survey_file": "file.xlsx", "vote_id": 153300,
             "admin_validated": True, "similarity_score": 0.5},
        ]
        id_map, _, _, dupes = build_migration_mapping(old_rows, vote_to_proc)
        assert len(id_map) == 1
        assert len(dupes) == 1
        # Winner should be higher score
        assert "T13_153304_0" in id_map


# ── Integration with real votes.csv (if available) ────────────────────


REAL_VOTES_CSV = Path(__file__).resolve().parents[1] / "data" / "votes" / "votes.csv"


@pytest.mark.skipif(not REAL_VOTES_CSV.exists(), reason="Real votes.csv not found")
class TestWithRealData:
    def test_main_vote_map_loads(self):
        mapping = build_main_vote_map(REAL_VOTES_CSV)
        assert len(mapping) > 100  # expect many procedures
        # Each procedure maps to exactly one vote_id
        assert len(set(mapping.keys())) == len(mapping)

    def test_vote_to_procedure_map_loads(self):
        mapping = build_vote_to_procedure_map(REAL_VOTES_CSV)
        assert len(mapping) > 1000

    def test_known_vote_id_maps_correctly(self):
        mapping = build_vote_to_procedure_map(REAL_VOTES_CSV)
        # From the CSV header we saw: vote 108532 → 2019/2730(RSP)
        assert mapping[108532] == "2019/2730(RSP)"

    def test_main_vote_for_known_procedure(self):
        mapping = build_main_vote_map(REAL_VOTES_CSV)
        # 2019/2730(RSP) main vote should be 108532 (Résolution, is_main=TRUE)
        assert mapping["2019/2730(RSP)"] == 108532
