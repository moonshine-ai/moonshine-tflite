import os
import pprint
import shutil

import tensorflow as tf
from tensorflow.python.tools import saved_model_utils
from moonshine import load_model

from keras.src import tree
from keras.src.export import ExportArchive

import flatbuffers
from tflite import Model


def load_tflite_model_from_file(model_filename):
    with open(model_filename, "rb") as file:
        buffer_data = file.read()
    model_obj = Model.Model.GetRootAsModel(buffer_data, 0)
    model = Model.ModelT.InitFromObj(model_obj)
    return model


def save_tflite_model_to_file(model, model_filename):
    builder = flatbuffers.Builder(1024)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    model_data = builder.Output()
    with open(model_filename, "wb") as out_file:
        out_file.write(model_data)


def _make_tensor_spec(x):
    shape = (None,) + x.shape[1:]
    return tf.TensorSpec(shape, dtype=x.dtype, name=x.name)


def convert_keras_to_tflite(
    keras_model, path, replace_output_names=False, quantize=False
):
    saved_model_path = path.replace(".tfl", ".saved_model")

    input_signature = tree.map_structure(_make_tensor_spec, keras_model.inputs)
    if isinstance(input_signature, list) and len(input_signature) > 1:
        input_signature = [input_signature]

    decorated_fn = tf.function(
        keras_model.__call__, input_signature=input_signature, autograph=False
    )

    export_archive = ExportArchive()
    export_archive.track(keras_model)
    export_archive.add_endpoint("serve", decorated_fn, input_signature)
    export_archive.write_out(saved_model_path, verbose=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    original_tflite_model = converter.convert()

    shutil.rmtree(saved_model_path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    original_path = path.replace(".tfl", ".original.tfl")

    with open(original_path, "wb") as f:
        f.write(original_tflite_model)

    modified_tflite_model = load_tflite_model_from_file(original_path)
    signature_def = modified_tflite_model.signatureDefs[0]
    subgraph = modified_tflite_model.subgraphs[0]
    if replace_output_names:
        for index, output_tensor in enumerate(keras_model.outputs):
            original_output_key = f"output_{index}"
            found = False
            for output_map_entry in signature_def.outputs:
                entry_name_str = output_map_entry.name.decode("utf-8")
                if entry_name_str == original_output_key:
                    output_map_entry.name = output_tensor.name.encode("utf-8")
                    tensor = subgraph.tensors[output_map_entry.tensorIndex]
                    tensor.name = output_tensor.name.encode("utf-8")
                    found = True
                    break
            if not found:
                print(
                    f"Output map entry '{original_output_key}' not found in {[(output_map_entry.name.decode('utf-8'), output_map_entry.tensorIndex) for output_map_entry in signature_def.outputs]}"
                )
                exit(1)
    else:
        for output_map_entry in signature_def.outputs:
            entry_name_str = output_map_entry.name
            tensor = subgraph.tensors[output_map_entry.tensorIndex]
            tensor.name = entry_name_str

    for input_map_entry in signature_def.inputs:
        entry_name_str = input_map_entry.name.decode("utf-8")
        if entry_name_str.startswith("serving_default_"):
            entry_name_str = entry_name_str.split("serving_default_")[1]
            entry_name_str = entry_name_str.split(":")[0]
        tensor = subgraph.tensors[input_map_entry.tensorIndex]
        tensor.name = entry_name_str.encode("utf-8")

    os.remove(original_path)

    save_tflite_model_to_file(modified_tflite_model, path)


for model_name in ["tiny", "base"]:
    for quantize in [True, False]:
        model = load_model(f"moonshine/{model_name}")

        if quantize:
            precision = "quantized"
        else:
            precision = "float"

        preprocessor_keras = model.preprocessor.preprocess
        encoder_keras = model.encoder.encoder
        decoder_uncached_keras = model.decoder.uncached_call
        decoder_cached_keras = model.decoder.cached_call

        assert model_name in ["tiny", "base"]
        if model_name == "tiny":
            layer_count = 6
        elif model_name == "base":
            layer_count = 8

        model_dir = os.path.join("models", model_name, precision)

        # Never quantize the preprocessor since it needs to operate on 16-bit input data.
        convert_keras_to_tflite(
            preprocessor_keras,
            os.path.join(model_dir, "preprocessor.tfl"),
            quantize=False,
        )
        convert_keras_to_tflite(
            encoder_keras, os.path.join(model_dir, "encoder.tfl"), quantize=quantize
        )
        convert_keras_to_tflite(
            decoder_uncached_keras,
            os.path.join(model_dir, "decoder_initial.tfl"),
            replace_output_names=True,
            quantize=quantize,
        )
        convert_keras_to_tflite(
            decoder_cached_keras,
            os.path.join(model_dir, "decoder.tfl"),
            replace_output_names=True,
            quantize=quantize,
        )
