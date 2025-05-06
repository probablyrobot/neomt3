# NeoMT3: Multi-Task Multitrack Music Transcription

NeoMT3 is a multi-instrument automatic music transcription model that uses the [T5X framework](https://github.com/google-research/t5x).

This is a fork of the original MT3 project by Google Research, now maintained by [Igor Bogicevic](https://github.com/probablyrobot).

Original work by Google Research. This is not an officially supported Google product.

## Quick Start

Use our [colab notebook](https://colab.research.google.com/github/probablyrobot/neomt3/blob/main/neomt3/colab/music_transcription_with_transformers.ipynb) to
try out the model on your own audio files.

## Installation

```bash
pip install neomt3
```

## Usage

To use the model, import the package
and use one of the tasks defined in [tasks.py](neomt3/tasks.py).

## Train a model

For now, we do not (easily) support training.  If you like, you can try to
follow the [T5X training instructions](https://github.com/google-research/t5x#training)
and use one of the tasks defined in [tasks.py](neomt3/tasks.py).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
The original work was created by Google Research and is also licensed under the Apache License 2.0.
