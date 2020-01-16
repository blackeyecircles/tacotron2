
import argparse
import os
import random
import time
import torch
# from apex import amp
from tacotron2.loader import parse_tacotron2_args
from tacotron2.loader import get_tacotron2_model
from tacotron2.text import text_to_sequence
from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger.autologging import log_hardware

from common.word2pinyin import word2pinyin
from wavernn.utils.dsp import *
from wavernn.fatchord_version import WaveRNN
# from utils.paths import Paths


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input-file', type=str, default="text.txt", help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', type=str, default="outputs", help='output folder to save audio (file per phrase)')
    parser.add_argument('--checkpoint_tacotron', type=str, default="pretrained_model/tacotron2_checkpoint.pyt", help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--checkpoint_wavernn', type=str, default="pretrained_model/wavernn_checkpoint.pyt", help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('-id', '--speaker-id', default=0, type=int, help='Speaker identity')
    parser.add_argument('-sn', '--speaker-num', default=1, type=int, help='Speaker number')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int, help='Sampling rate')
    parser.add_argument('--amp-run', action='store_true', help='inference with AMP')
    parser.add_argument('--log-file', type=str, default='nvlog.json', help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true', help='Include warmup')

    return parser


def load_and_setup_tacotron(parser, args):
    checkpoint_path = args.checkpoint_tacotron
    parser = parse_tacotron2_args(parser, add_help=False)
    args, _ = parser.parse_known_args()
    model = get_tacotron2_model(args, args.speaker_num, is_training=False)
    model.restore_checkpoint(checkpoint_path)
    model.eval()

    return model


def load_and_setup_wavernn(restore_path):

    model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    pad_val=hp.voc_pad_val,
                    mode=hp.voc_mode).cuda()

    model.restore(restore_path)

    return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(sequences):
    # Right zero-pad all one-hot text sequences to max input length
    text_lengths, ids_sorted_decreasing = torch.sort(
        torch.IntTensor([len(x) for x in sequences]),
        dim=0, descending=True)
    max_text_len = text_lengths[0]

    texts = []
    for i in range(len(ids_sorted_decreasing)):
        text = sequences[ids_sorted_decreasing[i]]
        texts.append(np.pad(text, [0, max_text_len - len(text)], mode='constant'))

    texts = torch.from_numpy(np.stack(texts))
    return texts, text_lengths, ids_sorted_decreasing


def prepare_input_sequence(texts, speaker_id):
    sequences = [text_to_sequence(text, speaker_id, ['basic_cleaners'])[:] for text in texts]
    texts, text_lengths, ids_sorted_decreasing = pad_sequences(sequences)

    if torch.cuda.is_available():
        texts = texts.cuda().long()
        text_lengths = text_lengths.cuda().int()
    else:
        texts = texts.long()
        text_lengths = text_lengths.int()

    return texts, text_lengths, ids_sorted_decreasing


class MeasureTime():
    def __init__(self, measurements, key):
        self.measurements = measurements
        self.key = key

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter() - self.t0


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(1234)

parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Inference')
parser = parse_args(parser)
args, _ = parser.parse_known_args()

LOGGER.set_model_name("TTS")
LOGGER.set_backends([
    dllg.StdOutBackend(log_file=None, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1),
    dllg.JsonBackend(log_file=args.log_file, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1)
])
LOGGER.register_metric("tacotron2_frames_per_sec", metric_scope=dllg.TRAIN_ITER_SCOPE)
LOGGER.register_metric("tacotron2_latency", metric_scope=dllg.TRAIN_ITER_SCOPE)
LOGGER.register_metric("total latency", metric_scope=dllg.TRAIN_ITER_SCOPE)

tacotron_model = load_and_setup_tacotron(parser, args)

wavernn_model = load_and_setup_wavernn(args.checkpoint_wavernn)
batched = hp.voc_gen_batched
target = hp.voc_target
overlap = hp.voc_overlap


log_hardware()

os.makedirs(args.output, exist_ok=True)


def synthetic_audio(text):
    sentences = word2pinyin(text)
    LOGGER.iteration_start()

    measurements = {}

    sequences, text_lengths, ids_sorted_decreasing = prepare_input_sequence(sentences, args.speaker_id)

    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time"):
        _, mels, _, _, mel_lengths = tacotron_model.infer(sequences, text_lengths)


    tacotron2_infer_perf = mels.size(0)*mels.size(2)/measurements['tacotron2_time']

    LOGGER.log(key="tacotron2_frames_per_sec", value=tacotron2_infer_perf)
    LOGGER.log(key="tacotron2_latency", value=measurements['tacotron2_time'])

    # recover to the original order and concatenate
    ids_sorted_decreasing = ids_sorted_decreasing.numpy().tolist()
    mels = [mel[:, :length] for mel, length in zip(mels, mel_lengths)]
    mels = [mels[ids_sorted_decreasing.index(i)] for i in range(len(ids_sorted_decreasing))]

    with torch.no_grad(), MeasureTime(measurements, "wavernn_time"):
        pcm = wavernn_model.generate(torch.tensor(np.concatenate(mels, axis=-1)).unsqueeze(0) + hp.mel_bias, 'outputs/eval_wave_long.wav', batched, target, overlap, hp.mu_law)

    LOGGER.log(key="wavernn_latency", value=measurements['wavernn_time'])
    LOGGER.log(key="latency", value=(measurements['tacotron2_time'] + measurements['wavernn_time']))
    LOGGER.iteration_stop()
    LOGGER.finish()
    return pcm


if __name__ == '__main__':
    text = "百日咳(pertussis，whoopingcough)是由百日咳杆菌所致的急性呼吸道传染病。其特征为阵发性痉挛性咳嗽，咳嗽末伴有特殊的鸡鸣样吸气吼声。病程较长，可达数周甚至3个月左右，故有百日咳之称。"
    synthetic_audio(text)