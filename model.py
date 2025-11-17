import numpy as np

import ai_edge_litert.interpreter as tflite

MOONSHINE_TOKEN_START = 1
MOONSHINE_TOKEN_END = 2


class MoonshineTFLiteModel(object):
    def __init__(self, model_dir="models", model_name="base", model_precision="float"):
        self.model_name = model_name

        assert self.model_name in ["tiny", "base"]
        if self.model_name == "tiny":
            self.layer_count = 6
        elif self.model_name == "base":
            self.layer_count = 8

        decoder_initial, decoder, encoder, preprocessor = [
            f"{model_dir}/{model_name}/{model_precision}/{x}.tfl"
            for x in ("decoder_initial", "decoder", "encoder", "preprocessor")
        ]

        self.preprocessor = tflite.Interpreter(
            preprocessor,
            # experimental_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.encoder = tflite.Interpreter(
            encoder,
            # experimental_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.decoder_initial = tflite.Interpreter(
            decoder_initial,
            # experimental_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.decoder = tflite.Interpreter(
            decoder,
            # experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )

        self.preprocessor.allocate_tensors()
        self.encoder.allocate_tensors()
        self.decoder_initial.allocate_tensors()
        self.decoder.allocate_tensors()

    def _index_for_name(self, names):
        return {name: index for index, name in enumerate(names)}

    def _compare_detail(self, detail):
        detail_name, detail_index = detail["name"].split(":")
        return (detail_name, int(detail_index))

    def _sort_by_name_and_index(self, details):
        return sorted(details, key=self._compare_detail)

    def _invoke_tflite(self, interpreter, inputs):
        input_details = interpreter.get_input_details()
        inputs_count = len(input_details)
        assert inputs_count == len(inputs), (
            f"Expected {len(inputs)} inputs, but got {inputs_count}"
        )

        for detail in input_details:
            name = detail["name"]
            tensor_index = detail["index"]
            assert name in inputs, f"Input '{name}' not found in inputs {inputs.keys()}"
            input = inputs[name]
            interpreter.resize_tensor_input(tensor_index, input.shape)

        interpreter.allocate_tensors()

        for detail in input_details:
            name = detail["name"]
            tensor_index = detail["index"]
            input = inputs[name]
            interpreter.set_tensor(tensor_index, input)

        interpreter.invoke()

        output_details = interpreter.get_output_details()
        result = {}
        for detail in output_details:
            name = detail["name"]
            tensor_index = detail["index"]
            result[name] = interpreter.get_tensor(tensor_index)

        return result

    def generate(self, audio, max_len=228):
        assert audio.shape[0] == 1, "Moonshine TFLite models support only batch size 1"

        audio_features = self._invoke_tflite(self.preprocessor, {"audio_input": audio})[
            "audio_features"
        ]

        seq_len = np.array([audio_features.shape[-2]], dtype=np.int32)
        last_hidden_state = self._invoke_tflite(
            self.encoder, {"audio_features": audio_features, "seq_len": seq_len}
        )["last_hidden_state"]

        input_ids = np.array([[MOONSHINE_TOKEN_START]], dtype=np.int32)
        seq_len = np.array([1], dtype=np.int32)
        initial_decoder_result = self._invoke_tflite(
            self.decoder_initial,
            {
                "input_ids": input_ids,
                "seq_len": seq_len,
                "last_hidden_state": last_hidden_state,
            },
        )
        logits = initial_decoder_result["logits"]
        cache = initial_decoder_result
        del cache["logits"]
        result_tokens = []
        for _ in range(max_len):
            previous_token = int(np.argmax(logits[0, 0, :]).astype(np.int32).flatten()[0])
            result_tokens.append(previous_token)
            if previous_token == MOONSHINE_TOKEN_END:
                break
            input_ids[0, 0] = previous_token
            seq_len[0] = seq_len[0] + 1
            decoder_input = {
                "input_ids": input_ids,
                "seq_len": seq_len,
                "last_hidden_state": last_hidden_state,
            }
            decoder_input.update(cache)
            decoder_result = self._invoke_tflite(self.decoder, decoder_input)
            logits = decoder_result["logits"]
            cache = decoder_result
            del cache["logits"]

        return result_tokens
