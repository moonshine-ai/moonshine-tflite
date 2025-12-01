import os
import tokenizers
from model import MoonshineTFLiteModel

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def load_audio(audio):
    if isinstance(audio, str):
        import librosa

        audio, _ = librosa.load(audio, sr=16_000)
        return audio[None, ...]
    else:
        return audio


def assert_audio_size(audio):
    assert len(audio.shape) == 2, "audio should be of shape [batch, samples]"
    num_seconds = audio.size / 16000
    assert 0.1 < num_seconds < 64, (
        "Moonshine models support audio segments that are between 0.1s and 64s in a single transcribe call. For transcribing longer segments, pre-segment your audio and provide shorter segments."
    )
    return num_seconds


def transcribe(
    audio, model_dir=None, model_name="moonshine/base", model_precision="float", num_threads=None
):
    model = MoonshineTFLiteModel(
        model_dir=model_dir, model_name=model_name, model_precision=model_precision, num_threads=num_threads
    )
    audio = load_audio(audio)
    assert_audio_size(audio)

    tokens = model.generate(audio)
    return load_tokenizer().decode(tokens)


def load_tokenizer():
    tokenizer_file = os.path.join(ASSETS_DIR, "tokenizer.json")
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    return tokenizer


def benchmark(
    audio, model_dir=None, model_name="moonshine/base", model_precision="float", num_threads=None
):
    import time

    model = MoonshineTFLiteModel(
        model_dir=model_dir, model_name=model_name, model_precision=model_precision, num_threads=num_threads
    )
    audio = load_audio(audio)
    num_seconds = assert_audio_size(audio)

    print("Warming up...")
    for _ in range(4):
        _ = model.generate(audio)

    print("Benchmarking...")
    N = 8
    start_time = time.time_ns()
    for _ in range(N):
        _ = model.generate(audio)
    end_time = time.time_ns()

    elapsed_time = (end_time - start_time) / N
    elapsed_time /= 1e6

    print(f"Time to transcribe {num_seconds:.2f}s of speech is {elapsed_time:.2f}ms")


if __name__ == "__main__":
    import tokenizers
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", type=str, default=os.path.join(ASSETS_DIR, "beckett.wav")
    )
    parser.add_argument("--model-name", type=str, default="base")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument(
        "--model-precision", type=str, default="float", choices=["float", "quantized"]
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--num-threads", type=int, default=None)
    args = parser.parse_args()

    audio = load_audio(args.audio)

    text = transcribe(
        audio,
        model_name=args.model_name,
        model_dir=args.model_dir,
        model_precision=args.model_precision,
        num_threads=args.num_threads,
    )
    print(text)

    if args.benchmark:
        benchmark(
            audio,
            model_name=args.model_name,
            model_dir=args.model_dir,
            model_precision=args.model_precision,
            num_threads=args.num_threads,
        )
