import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torchvision.transforms as T
import torch.nn.functional as F
import requests
import json

from src import utils

log = utils.get_pylogger(__name__)


def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating Scripted model <{cfg.ckpt_path}>")
    # model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = torch.jit.load(cfg.ckpt_path)
    log.info(f"Loaded Model: {model}")

    log.info(f"Loaded Model: {model}")

    cifar10_labels = requests.get(
        "https://gist.githubusercontent.com/dexter11235813/78c99088035f97c2e8461fdfc17318db/raw/20ee029d9a0cc703ac87977b5d3e4104711762ad/cifar10_labels.json"
    )
    with open("cifar10_labels.json", "wb") as f:
        f.write(cifar10_labels.content)

    with open("cifar10_labels.json") as f:
        imagenet_labels = list(json.loads(f.read()).values())

    def recognize_cifar(image):
        if image is None:
            return None
        image = T.ToTensor()(image).unsqueeze(0)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        return {imagenet_labels[i]: preds[i] for i in range(10)}

    im = gr.Image(shape=(32, 32), image_mode="RGB", invert_colors=True, source="upload")

    demo = gr.Interface(
        fn=recognize_cifar,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(server_port=8080, server_name="0.0.0.0")


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="demo.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)


if __name__ == "__main__":
    main()
