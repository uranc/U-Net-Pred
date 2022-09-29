from pathlib import Path

import numpy as np

from model.model_fn import build_compile_model
from model_tools.activations.keras import KerasWrapper, load_images
from model_tools.check_submission import check_models

"""
Template module for a base model submission to brain-score
"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['gammanet']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'gammanet'
    params = {
        # https://github.com/uranc/U-Net-Pred/blob/e9170a8edca7ab3d5baf0b632c614b9bc8a27d1a/sample_nat_im_demo.ipynb
        'weight_dir': str(Path(__file__).parent / '..' / 'weights.last.h5'), 'lr': 1.56 * 1e-4,
        'p_content': 0, 'p_style': 0, 'p_hole': 0, 'p_tv': 0,
        'p_fft_s': 0.002, 'p_fft_c': 8.79,  # 'p_fft_s': 0.330, 'p_fft_c': 505.47,
        'p_fft_abs': 1, 'p_fft_log': 0.026, 'p_fft_phase': 6.45e-05,  # .0556
        'b_size': 4, 'current_epoch': 0,
        'n_epoch': 150000,
        'exp_dir': 'experiments/',
        'inst_norm': False,
        'input_size': 224
    }
    model = build_compile_model(params)
    basemodel = KerasWrapper(model=model, preprocessing=preprocessing, batch_size=1)
    return basemodel


def preprocessing(stimulus_paths):
    images = load_images(stimulus_paths, image_size=224)
    # from https://github.com/uranc/U-Net-Pred/blob/e9170a8edca7ab3d5baf0b632c614b9bc8a27d1a/sample_nat_im_demo.ipynb
    normalizer = [103.939, 116.779, 123.68]
    images = [image[:, :] - normalizer for image in images]
    images = [np.expand_dims(image, axis=0) for image in images]

    im_mask = [np.zeros(image.shape) for image in images]
    model_input = [images] + [im_mask]

    return model_input


def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    assert name == 'gammanet'
    return ['re_lu'] + [f're_lu_{num}' for num in range(1, 20)]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    assert model_identifier == 'gammanet'
    return """@article {Uran2020.08.10.242958,
        author = {Uran, Cem and Peter, Alina and Lazar, Andreea and Barnes, William and Klon-Lipok, Johanna and Shapcott, Katharine A and Roese, Rasmus and Fries, Pascal and Singer, Wolf and Vinck, Martin},
        title = {Predictive coding of natural images by V1 activity revealed by self-supervised deep neural networks},
        elocation-id = {2020.08.10.242958},
        year = {2021},
        doi = {10.1101/2020.08.10.242958},
        publisher = {Cold Spring Harbor Laboratory},
        abstract = {Predictive coding is an important candidate theory of self-supervised learning in the brain. Its central idea is that neural activity results from an integration and comparison of bottom-up inputs with contextual predictions, a process in which firing rates and synchronization may play distinct roles. Here, we quantified stimulus predictability for natural images based on self-supervised, generative neural networks. When the precise pixel structure of a stimulus falling into the V1 receptive field (RF) was predicted by the spatial context, V1 exhibited characteristic γ-synchronization (30-80Hz), despite no detectable modulation of firing rates. In contrast to γ, β-synchronization emerged exclusively for unpredictable stimuli. Natural images with high structural predictability were characterized by high compressibility and low dimensionality. Yet, perceptual similarity was mainly determined by higher-level features of natural stimuli, not by the precise pixel structure. When higher-level features of the stimulus in the receptive field were predicted by the context, neurons showed a strong reduction in firing rates and an increase in surround suppression that was dissociated from synchronization patterns. These findings reveal distinct roles of synchronization and firing rates in the predictive coding of natural images.Competing Interest StatementThe authors have declared no competing interest.},
        URL = {https://www.biorxiv.org/content/early/2021/04/22/2020.08.10.242958},
        eprint = {https://www.biorxiv.org/content/early/2021/04/22/2020.08.10.242958.full.pdf},
        journal = {bioRxiv}
    }"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
