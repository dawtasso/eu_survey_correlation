import json
import re


def build_survey_date_mapping(distribution_meta_json_path: str) -> dict[str, str]:
    with open(distribution_meta_json_path) as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        title_en = item.get("title", {}).get("en", "") or item.get("title", {}).get(
            "fr", ""
        )
        issued = item.get("issued", "")
        if not issued or not title_en:
            continue
        if "volume_B" not in title_en or "volume_BP" in title_en:
            continue
        name = title_en
        for prefix in ["Link to ", "Lien vers ", "Lien_vers_"]:
            name = name.replace(prefix, "")
        name = re.sub(r"\.(zip|xlsx|xls)$", "", name, flags=re.I)
        mapping[name.strip()] = issued[:10]
    return mapping


def resolve_survey_date(file_name: str, date_mapping: dict[str, str]) -> str | None:
    base = re.sub(r"\.(xlsx|xls|zip)$", "", file_name, flags=re.I)
    if base in date_mapping:
        return date_mapping[base]
    without_lien_vers_base = base.split("Lien_vers_")[-1]
    if without_lien_vers_base in date_mapping:
        return date_mapping[without_lien_vers_base]

    file_nums = re.findall(r"(?:fl|ebs|eb|sp)_?(\d{3})", file_name, re.I)
    if file_nums:
        target = file_nums[0]
        for key, date in date_mapping.items():
            key_nums = re.findall(r"(?:fl|ebs|eb|sp)_?(\d{3})", key, re.I)
            if target in key_nums:
                return date
    year_match = re.search(r"(20\d{2})", file_name)
    if year_match:
        return f"{year_match.group(1)}-06-01"
    return None
