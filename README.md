# Moonshine TFLite

[Moonshine's speech to text models](https://github.com/moonshine-ai/moonshine) converted to the TensorFlow Lite (aka LiteRT) framework.

## Running Moonshine using TensorFlow Lite

```bash
python transcribe.py
```

The `transcribe.py` script gives an example of how to run speech recognition in Python using the TFLite interpreter. It uses the class definition in `model.py`, and takes three optional arguments:

 - The path to a WAV file containing audio that you want to convert into text. This defaults to `assets/beckett.wav` if not specified.

 - The name of the model architecture to use, either `tiny` or `base`. Defaults. to `base`.

 - Path to a folder containing the four model files necessary for inference. If none is specified, defaults to `models` and loads the model files in this repository.

## Model Files

TensorFlow Lite versions of the English-language Moonshine models are part of this repository in the `models` folder.

 ## Converting from Keras

 The `convert.py` script runs an export process to convert the models from Keras format to TFLite. Currently it only supports float32 models. You shouldn't need to run this yourself unless you've modified the original Keras model, since the generated files are [available on HuggingFace](https://huggingface.co/UsefulSensors/moonshine/tree/main/tflite).

## Notes

The TFLite schema included here was downloaded from https://github.com/tensorflow/tensorflow/blob/395df4d064f8e4db15f5a9a14e332346d9f613bc/tensorflow/compiler/mlir/lite/schema/schema_v3c.fbs

The Python schema access files were created by running:

```bash
flatc --python --gen-object-api tflite_schema_v3c.fbs
```