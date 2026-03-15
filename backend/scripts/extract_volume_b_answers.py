from eu_survey_correlation.surveys import volume_b_parser
from loguru import logger


def main():
    parser = volume_b_parser.VolumeBParser()
    df = parser.extract_all()

    if df.empty:
        logger.error("No data extracted!")
        return

    volume_b_parser.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(volume_b_parser.OUTPUT_CSV, index=False)
    logger.success(f"Saved {len(df)} rows to {volume_b_parser.OUTPUT_CSV}")

    # Quick summary
    logger.info("\n--- Summary ---")
    logger.info(f"Questions: {df['sheet_id'].nunique()}")
    logger.info(f"Files: {df['file_name'].nunique()}")
    logger.info(f"Demographic types: {df['demographic_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
