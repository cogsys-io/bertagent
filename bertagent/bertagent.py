#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""BERTAgent main module."""

import torch
import logging
import pathlib
import pandas as pd
import numpy as np

from typing import (
    Dict,
    Union,
    Sequence,
    List,
    Optional,
)

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

EXAMPLE_SENTENCES = [
    "She is a hard working individual",
    "She is a hardly working individual",
    "This thing was made of lead",
    "This is a car, it runs on gas",
    "This is a Karen, she runs for office",
    "This is a Jane, she runs for office",
    "Striving to achieve my goals",
    "Struggling to achieve my goals",
    "Striving to make it",
    "Struggling to make it",
    "Striving to survive",
    "Struggling to survive",
    "Well planned and well executed",
    "Coordinated activity",
    "Uncoordinated activity",
    "Not coordinated activity",
    "Everything is messy and uncoordinated",
    "A bad decisionmaker",
    "A marvelous decisionmaker",
    "They are submissive",
    "They submitted to his will",
    "They submitted a paper",
    "They submitted a request",
    "We are winners",
    "We are losers",
    "motivated",
    "We are motivated",
    "We are not motivated",
    "We are unmotivated",
    "Lazy and unmotivated",
    "I want to give up",
    "lost all hope",
    "We'll lose anyway",
    "We should give up and say nothing",
    "We must win",
    "We will lead our way out of trouble",
    "We must fight for our rights",
    "We should take control and assert our position",
    "We should take control",
    "We shoud take controll",
    "Hard working individual. Hardly working individual",
]

MAX_LENGTH = 128
TOKENIZER_PARAMS = dict(
    add_special_tokens=True,
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
)


class BERTAgent:
    """Evaluates agency in a list of sentences.

    Parameters
    ----------
    model_path : Union[str, pathlib.Path]
        path to huggingface repository or a local directory
        containing fine-tuned model (e.g., BERTAgent)

    tokenizer_path : Union[str, pathlib.Path]
        path to text tokenizer

    tokenizer_params : Dict
        tokenizer parameters dictionary, see examples below
        (``TOKENIZER_PARAMS``)

    device : Union[str, torch.device] = "cuda"
        torch device to use (default = "cuda")

    factor : float
        response scaling factor (default = 1)

    bias : float = 0.0
        response shifting factor (default = 0)

    log0 : logging.Logger
        optional logger to use

    Examples
    --------

    Process a list of sentences.

    >>> # Imports
    >>> import pathlib
    >>> from bertagent import BERTAgent
    >>>
    >>> # Load BERTAgent
    >>> ba0 = BERTAgent()
    >>>
    >>> sents = [
    >>>     "stiving to achieve my goals",
    >>>     "struglling to survive",
    >>>     "hardly working individual",
    >>>     "hard working individual",
    >>> ]
    >>> vals = ba0.predict(sents)
    >>> for item in zip(sents, vals):
    >>>     print(item)
    #
    # ('stiving to achieve my goals', 0.7477692365646362)
    # ('struglling to survive', 0.043704114854335785)
    # ('hardly working individual', -0.5707859396934509)
    # ('hard working individual', 0.43518713116645813)
    #
    # NOTE: exact values may differ slightly from the above
    # depending on the BERTAgent model used and version.

    Process a texts in pandas dataframe.

    >>> # Imports.
    >>> import pathlib
    >>> import pandas as pd
    >>> from tqdm import tqdm
    >>> from bertagent import BERTAgent
    >>> from bertagent import EXAMPLE_SENTENCES as sents
    >>> tqdm.pandas()
    >>>
    >>> # Load BERTAgent.
    >>> ba0 = BERTAgent()
    >>>
    >>> # Prepare dataframe.
    >>> df0 = pd.DataFrame(dict(text=sents))
    >>>
    >>> # Extract sentences from text.
    >>> # NOTE: This is not an optimal method to get sentences from real data!
    >>> df0["sents"] = df0.text.str.split(".")
    >>>
    >>> print(df0.head(n=4))


    >>> # Evaluate agency
    >>> model_id = "ba0"
    >>> df0[model_id] = df0.sents.progress_apply(ba0.predict)
    >>>
    >>> df0["BATot"] = df0[model_id].apply(ba0.tot)
    >>> df0["BAPos"] = df0[model_id].apply(ba0.pos)
    >>> df0["BANeg"] = df0[model_id].apply(ba0.neg)
    >>> df0["BAAbs"] = df0[model_id].apply(ba0.abs)
    >>>
    >>> cols0 = [
    >>>     "sents",
    >>>     "ba0",
    >>>     "BATot",
    >>>     "BAPos",
    >>>     "BANeg",
    >>>     "BAAbs",
    >>> ]
    >>>
    >>> # Check example rows.
    >>> df0[cols0].tail(n=8)

    """

    def __init__(
        self,
        model_path: Union[str, pathlib.Path] = None,
        tokenizer_path: Union[str, pathlib.Path, None] = None,
        tokenizer_params: Dict = TOKENIZER_PARAMS,
        device: Union[str, torch.device] = "cuda",  # TODO checkup
        # device: str = "cuda",
        factor: float = 1.0,
        bias: float = 0.0,
        log0: logging.Logger = logging.getLogger("dummy"),
    ):
        if model_path is None:
            model_path = "EnchantedStardust/bertagent-best"

        if tokenizer_path is None:
            tokenizer_path = model_path

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            do_lower_case=True,
        )
        self.tokenizer_params = tokenizer_params
        self.device = device
        self.factor = factor
        self.bias = bias
        self.log0 = log0

        self.model.to(device)
        self.model.eval()

        # self.log0.debug(f"{self.model.device = }")
        self.log0.debug(f"{self.model.training = }")
        self.log0.debug(f"{self.tokenizer = }")
        self.log0.debug(f"{self.tokenizer_params = }")

    def predict(self, sentences: List[str]) -> List[float]:
        """Predict agency for a list of texts.

        Parameters
        ----------
        sentences : List[str]
            a list of texts (e.g., sentences).

        Returns
        -------
        List[float]
            List of scores.


        .. note::
            See doc for the BERTAgent class for usage examples.




        """
        batch_encodings = self.tokenizer(
            list(sentences),
            None,
            **self.tokenizer_params,
            return_tensors="pt",
        )
        self.model.eval()  # CHECKUP
        batch_encodings.to(self.model.device)
        # CONSIDER: adding here a warning if text contains too many tokens

        # with torch.inference_mode():
        with torch.no_grad():
            predictions = (
                self.model(**batch_encodings)["logits"].cpu().detach().numpy()
                * self.factor
                + self.bias
            )

        predictions = predictions.ravel().tolist()
        batch_encodings.to(self.model.device)
        torch.cuda.empty_cache()  # CONSIDER DROP
        return predictions

    @classmethod
    def tot(self, vals: List[Union[int, float]]) -> float:
        """Get the total score (mean) from a list of BERTAgent scores.

        Parameters
        ----------
        vals : List[Union[int, float]]
            a list of scores.

        Returns
        -------
        float
            Agency (total) score.


        .. note::
            See doc for the BERTAgent class for usage examples.

        """

        len0 = len(vals)
        return sum(vals) / len0 if len0 else 0

    @classmethod
    def pos(self, vals):
        """Get the agency-positive score from a list of BERTAgent scores.

        This score is commuted as mean of all scores with negative
        values replaced by 0.

        Parameters
        ----------
        vals : List[Union[int, float]]
            a list of scores.

        Returns
        -------
        float
            Agency-positive score.


        .. note::
            See doc for the BERTAgent class for usage examples.

        """

        len0 = len(vals)
        vals = [val for val in vals if val > 0]
        return sum(vals) / len0 if len0 else 0

    @classmethod
    def neg(self, vals):
        """Get the agency-negative score from a list of BERTAgent scores.

         This score is commuted as mean of all scores with positive
         values replaced by 0.

         Parameters
         ----------
         vals : List[Union[int, float]]
             a list of scores.

        Returns
        -------
        float
            Agency-negative score.


        .. note::
            See doc for the BERTAgent class for usage examples.

        """

        len0 = len(vals)
        vals = [-val for val in vals if val < 0]
        return sum(vals) / len0 if len0 else 0

    @classmethod
    def abs(self, vals):
        """Get the agency-absolute score from a list of BERTAgent scores.

         This score is commuted as mean of absolute values of all scores.

         Parameters
         ----------
         vals : List[Union[int, float]]
             a list of scores.

        Returns
        -------
        float
            Agency-absolute score.


        .. note::
            See doc for the BERTAgent class for usage examples.

        """

        len0 = len(vals)
        vals = [abs(val) for val in vals]
        return sum(vals) / len0 if len0 else 0
