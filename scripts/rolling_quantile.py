"""
Pipeline : Compute and upload on Postgre Server the 99th quantile of 
solar regional production data, on the last 90 days.
author : Simon.sngs
date : 22/04/2026
"""

# Dependencies
from src.pipelines.quantile_update_pipeline import RollingQuantile
from src.utils.config import settings
from src.utils.logger import setup_logging

if __name__ == "__main__":
    
    # Init
    setup_logging()

    # Rolling quantile job
    rolling_quantile = RollingQuantile(
        config=settings)
    rolling_quantile.run()
# %%
