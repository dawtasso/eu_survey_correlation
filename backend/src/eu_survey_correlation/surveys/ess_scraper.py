from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


class ESSCodebookParser:
    """Parses an ESS codebook HTML file into a flat DataFrame."""

    def __init__(self, html_path: str | Path):
        self.html_path = Path(html_path)

    def parse(self) -> pd.DataFrame:
        soup = BeautifulSoup(self.html_path.read_text(encoding="utf-8"), "html.parser")

        rows: list[dict] = []
        for h3 in soup.find_all("h3", id=True):
            div = h3.parent
            if div is None or div.name != "div":
                continue

            variable_id = h3["id"]
            child_divs = div.find_all("div", recursive=False)
            variable_name = child_divs[0].text
            # Description is the first <div> child that is not a meta-string or data-table
            description = ""
            for child_div in child_divs[1:]:
                cls = child_div.get("class") or []
                if "data-table" not in cls:  # and "variable-meta-string" not in cls:
                    description += child_div.get_text(strip=True)

            # Answer values from the data-table
            table = div.find("tbody", class_="codelist")
            if table is None:
                rows.append(
                    {
                        "variable_id": variable_id,
                        "variable_name": variable_name,
                        "variable_description": description,
                        "answer_id": None,
                        "answer_value": None,
                    }
                )
                continue

            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    rows.append(
                        {
                            "variable_id": variable_id,
                            "variable_name": variable_name,
                            "variable_description": description,
                            "answer_id": tds[0].get_text(strip=True),
                            "answer_value": tds[1].get_text(strip=True),
                        }
                    )

        return pd.DataFrame(rows)
