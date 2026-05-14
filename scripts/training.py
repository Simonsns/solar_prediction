#%%
# Libraries
import os
from pathlib import Path
PROJECT_ROOT = Path().resolve().parent
os.chdir(PROJECT_ROOT)
from pipelines.training_pipeline import SolarTrainingOrchestrator
from src.etl import processors
from src.utils.config import settings
from src.utils.logger import setup_logging
import joblib

if __name__ == "__main__":
    meteo_features = joblib.load(PROJECT_ROOT / "data/processed/features/meteo_features.pkl")
    setup_logging()
    
    # 1 - Download artifact
    instance_artifact = processors.LGBMDataloader(
        url=settings.supabase_url.get_secret_value(),
        key=settings.supabase_key.get_secret_value(),
        )
    X_train, X_test, y_train, y_test = instance_artifact.run(
        bucket_name=settings.training_bucket_name.get_secret_value(),
        file_path="latest_dataset.parquet")
    
    # 2 - Training pipeline
    training_orchestrator_instance = SolarTrainingOrchestrator(
        config=settings,
        selector_parameters= {
            "lgbm_params": {
                'num_leaves': 61,              
                'max_depth': -1,               
                'learning_rate': 0.01,         
                'n_estimators': 500,
                'min_child_samples': 20,
                'verbosity': -1,
                'random_state': 42,
                'importance_type': 'gain',     
                'n_jobs': -1,
            },
            "threshold": 0.95,
        },
        meteo_features=meteo_features,
        experiment_name="LGBM_Model_1.7-founder_edition",
        num_trials=25
    )

    # 3 - 24h training
    training_orchestrator_instance.run_training_pipeline(X_train=X_train, y_train=y_train)

    # 4 - Final meta-model packaging
    training_orchestrator_instance.package_and_log_meta_model(X_sample=X_train.iloc[:3])
# %%
