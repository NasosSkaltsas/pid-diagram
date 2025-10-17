from pathlib import Path
import papermill as pm
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_image_artifact,create_markdown_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = PROJECT_ROOT / "notebooks"
OUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data" / "dummy"

@task
def run_notebook(name: str, params: dict):
    """Run one notebook via Papermill."""
    logger = get_run_logger()
    nb_path = NB_DIR / name
    out_nb = OUT_DIR / f"{nb_path.stem}_executed.ipynb"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running notebook {name}")
    pm.execute_notebook(
        input_path=str(nb_path),
        output_path=str(out_nb),
        parameters=params,
        raise_on_error=True,
    )
    logger.info(f"✅ Finished {name}")
    return out_nb


@task
def publish_artifacts():
    """Publish outputs as artifacts."""
    # images
    detections = OUT_DIR / "detections"
    if detections.exists():
        for img in detections.glob("*_annotated.png"):
            create_image_artifact(
                image_url=str(img),
                description=f"Annotated detection: {img.name}"
            )

    # markdown (diagnosis)
    diag = OUT_DIR / "diagnosis.md"
    if diag.exists():
        create_markdown_artifact(
            key="llm-diagnosis",
            description="Final diagnosis output by Google AI",
            markdown=diag.read_text(encoding="utf-8"),
        )


@flow(name="PID Pipeline")
def pid_pipeline(
    image_path: str = r"C:\Users\Administrator\Desktop\PID_PROJECT\data\test\images\test_pid.png",
):
    """Sequentially run all 4 notebooks."""
    params = {"image_path": image_path, 
              "outputs_dir": str(OUT_DIR),
              "data_dir": str(DATA_DIR)}

    run_notebook.with_options(name="Notebook 1 - PID System Components Detection")("notebook_1_detect.ipynb", params)
    run_notebook.with_options(name="Notebook 2 - Network Mapping")("notebook_2_network.ipynb", params)
    run_notebook.with_options(name="Notebook 3 - Anomaly Detection")("notebook_3_anomaly.ipynb", params)
    run_notebook.with_options(name="Notebook 4 - LLM Explanation")("notebook_4_explain.ipynb", params)

    publish_artifacts.with_options(name="Publish Artifacts")()

if __name__ == "__main__":
    pid_pipeline()
