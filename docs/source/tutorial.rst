.. highlight:: python

========
Tutorial
========

In this tutorial, first, we will evaluate linguistic agency in a list of
sentences (each sentence will be evaluated separately). Then we will demonstrate
how to use BERTAgent on a dataframe that contains a column of textual data (each
entry in this column will also be evaluated separately).

.. note::
   See :ref:`installation:Installation` section for installation instructions.

.. note::
   For a detailed documentation of BERTAgent's functionality
   please refer to
   :ref:`package docs <bertagent:bertagent package>` section.

Process a list of sentences
---------------------------

First, we need to import and initialize BERTAgent.


.. code-block:: python

    >>> from bertagent import BERTAgent
    >>> ba0 = BERTAgent()

Then we need to provide example sentences as a list of strings.

.. code-block:: python

    >>> sents = [
    >>>    ...:     "hardly working individual",
    >>>    ...:     "hard working individual",
    >>>    ...:     "striving to achieve my goals",
    >>>    ...:     "struggling to achieve my goals",
    >>>    ...:     "struggling to survive",
    >>>    ...:     "unable to survive",
    >>>    ...:     "this car runs on gasoline with lead",
    >>>    ...:     "this car runs on gasoline and it will lead us",
    >>>    ...:     "this politician runs for office and he will lead us",
    >>>    ...: ]


We get the linguistic agency evaluated for each utterance by running the
`predict` method.
:meth:`bertagent.BERTAgent.predict`

Check the :meth:`predict <bertagent.BERTAgent.predict>` documentation for more advanced usage.

.. code-block:: python

    >>> vals = ba0.predict(sents)

Print results

.. code-block:: python

    >>> for item in zip(sents, vals):
    >>>     print(f"  {item[0]!r} : {item[1]:.2f}")
    #
    #  'hardly working individual' : -0.57
    #  'hard working individual' : 0.44
    #  'striving to achieve my goals' : 0.73
    #  'struggling to achieve my goals' : -0.67
    #  'struggling to survive' : -0.52
    #  'unable to survive' : -0.57
    #  'this car runs on gasoline with lead' : -0.03
    #  'this car runs on gasoline and it will lead us' : 0.09
    #  'this politician runs for office and he will lead us' : 0.58
    #
    # NOTE: exact values may differ slightly from the above
    # depending on the BERTAgent model and version used.


Process a texts in pandas dataframe
-----------------------------------

.. note::
   Here we use
   ``EXAMPLE_SENTENCES`` data
   that is
   provided with ``BERTAgent``.
   The code below provides an example use of
   BERTAgent that can be uses as a template
   to analyze any other textual data provided by user.
   Importantly BERTAgent should be used to quantify agency in short
   utterances (preferably a single sentence).
   This is why we need to provide BERTAgnet
   with a list of sentences.
   If the user has longer texts they can be
   "chunkified" using brute force
   approach or (prefferabely) using
   natural language processing libraries
   such as
   `SpaCy <https://spacy.io>`_ for more information.



Imports

.. code-block:: python

    >>> import pathlib
    >>> import pandas as pd
    >>> from tqdm import tqdm
    >>> from bertagent import BERTAgent
    >>> from bertagent import EXAMPLE_SENTENCES as sents
    >>> tqdm.pandas()
    >>>

Load BERTAgent

.. code-block:: python

    >>> ba0 = BERTAgent()

Prepare dataframe.

.. code-block:: python

    >>> df0 = pd.DataFrame(dict(text=sents))

Extract sentences from text.

.. code-block:: python

    >>> # NOTE: This is not an optimal method to get sentences from real data!
    >>> df0["sents"] = df0.text.str.split(".")

Check input dataframe

.. code-block:: python

    >>> print(df0.head(n=4))


.. csv-table:: Input data (pandas dataframe containing lists of sentences)
   :file: tutorial-01-input.csv
   :widths: 10, 90
   :header-rows: 1




Evaluate agency

.. code-block:: python

    >>> model_id = "ba0"
    >>> df0[model_id] = df0.sents.progress_apply(ba0.predict)

Compute more specific indices of agency
(``tot`` = total = sum af all values for all sentences,
``pos`` = only positive,
``neg`` = only negative,
``abs`` = sum of absolute values)

.. code-block:: python

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

Check output

.. code-block:: python

    >>> df0[cols0].tail(n=8)


.. csv-table:: Output data (pandas dataframe with agency evaluation)
   :file: tutorial-02-output.csv
   :widths: 5, 70, 10, 10, 10, 10, 10
   :header-rows: 1


.. note::
   The last row demonstrates how a text that contains
   multiple sentences is handled, each sentence is assigned a
   separate agency score.
