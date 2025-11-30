# Moonshine TFLite

[Moonshine's speech to text models](https://github.com/moonshine-ai/moonshine) converted to the TensorFlow Lite (aka LiteRT) framework.

## Running Moonshine using TensorFlow Lite

```bash
pip install -r requirements.txt
python transcribe.py
```

The `transcribe.py` script gives an example of how to run speech recognition in Python using the TFLite interpreter. This repository already includes TFLite versions of the Moonshine models in the `models` folder and it uses those by default. The script uses the class definition in `model.py`, and takes four optional arguments:

- `--audio`: The path to a WAV file containing audio that you want to convert into text. This defaults to `assets/beckett.wav` if not specified.

- `--model-name`: The name of the model architecture to use, either `tiny` or `base`. Defaults. to `base`.

- `--model-dir`: Path to a folder containing the four model files necessary for inference. If none is specified, defaults to `models` and loads the model files in this repository.

- `--benchmark`: Run a benchmark to provide latency information.

## Model Files

TensorFlow Lite versions of the English-language Moonshine models are part of this repository in the `models` folder.

## Converting from Keras

The `convert.py` script runs an export process to convert the models from Keras format to TFLite. Currently it only supports float32 models. You shouldn't need to run this yourself unless you've modified the original Keras model, since the generated files are included.

You'll need to run `pip install -r requirements-conversion.txt` to gather the right dependencies before you run the conversion script.

## Notes

The TFLite schema included here was downloaded from https://github.com/tensorflow/tensorflow/blob/395df4d064f8e4db15f5a9a14e332346d9f613bc/tensorflow/compiler/mlir/lite/schema/schema_v3c.fbs

The Python schema access files were created by running:

```bash
flatc --python --gen-object-api tflite_schema_v3c.fbs
```
