from pathlib import Path
import modal

modal_app = modal.App.lookup("image-to-image-translation", create_if_missing=True)
modal_image = modal.Image.debian_slim(
    python_version="3.12"
).pip_install_from_requirements(
    str(Path(__file__).parent.parent.parent / "requirements.txt")
)
modal_volume = modal.Volume.from_name("image-to-image-translation-weights", create_if_missing=True)
