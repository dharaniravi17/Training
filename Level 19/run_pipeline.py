# run_pipeline.py

from pipeline import training_pipeline
from pipeline import load_data, train_model

if __name__ == "__main__":
    pipeline_instance = training_pipeline(load_data_step=load_data(), train_model_step=train_model())
    pipeline_instance.run()
