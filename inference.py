# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
import os
import random
import time
import torch
# from apex import amp
from common import audio
from tacotron2.loader import parse_tacotron2_args
from tacotron2.loader import get_tacotron2_model
from tacotron2.text import text_to_sequence
from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger.autologging import log_hardware, log_args


from wavernn.utils.dsp import *
from wavernn.fatchord_version import WaveRNN
# from utils.paths import Paths

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input-file', type=str, default="text.txt", help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', type=str, default="outputs", help='output folder to save audio (file per phrase)')
    parser.add_argument('--checkpoint_tacotron', type=str, default="logs/checkpoint_latest.pt", help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--checkpoint_wavernn', type=str, default="logs/wavernn_latest_weights.pyt", help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('-id', '--speaker-id', default=0, type=int, help='Speaker identity')
    parser.add_argument('-sn', '--speaker-num', default=1, type=int, help='Speaker number')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int, help='Sampling rate')
    parser.add_argument('--amp-run', action='store_true', help='inference with AMP')
    parser.add_argument('--log-file', type=str, default='nvlog.json', help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true', help='Include warmup')

    return parser


def load_checkpoint(checkpoint_path, model_name):
    assert os.path.isfile(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def load_and_setup_tacotron(parser, args):
    checkpoint_path = args.checkpoint_tacotron
    parser = parse_tacotron2_args(parser, add_help=False)
    args, _ = parser.parse_known_args()
    model = get_tacotron2_model(args, args.speaker_num, is_training=False)
    model.restore_checkpoint(checkpoint_path)
    model.eval()

    if args.amp_run:
        model, _ = amp.initialize(model, [], opt_level="O3")

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

    # paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    # restore_path = args.weights if args.weights else paths.voc_latest_weights

    model.restore(restore_path)

    # simple_table([('Generation Mode', 'Batched' if batched else 'Unbatched'),
    #               ('Target Samples', target if batched else 'N/A'),
    #               ('Overlap Samples', overlap if batched else 'N/A')])

    # k = model.get_step() // 1000
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    setup_seed(1234)

    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    LOGGER.set_model_name("Tacotron2_PyT")
    LOGGER.set_backends([
        dllg.StdOutBackend(log_file=None, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1),
        dllg.JsonBackend(log_file=args.log_file, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1)
    ])
    LOGGER.register_metric("tacotron2_frames_per_sec", metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("tacotron2_latency", metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("latency", metric_scope=dllg.TRAIN_ITER_SCOPE)

    tacotron = load_and_setup_tacotron(parser, args)

    wavernn = load_and_setup_wavernn(args.checkpoint_wavernn)
    batched = hp.voc_gen_batched
    samples = hp.voc_gen_at_checkpoint
    target = hp.voc_target
    overlap = hp.voc_overlap
    gta = False


    log_hardware()
    log_args(args)

    if args.include_warmup:
        sequences = torch.randint(low=0, high=148, size=(1,50),
                                  dtype=torch.long).cuda()
        text_lengths = torch.IntTensor([sequences.size(1)]).cuda().long()
        for i in range(3):
            with torch.no_grad():
                _, mels, _, _, mel_lengths = tacotron.infer(sequences, text_lengths)

    try:
        f = open(args.input_file)
        sentences = list(map(lambda s : s.strip(), f.readlines()))
    except UnicodeDecodeError:
        f = open(args.input_file, encoding='gbk')
        sentences = list(map(lambda s : s.strip(), f.readlines()))

    os.makedirs(args.output, exist_ok=True)

    LOGGER.iteration_start()

    measurements = {}

    sequences, text_lengths, ids_sorted_decreasing = prepare_input_sequence(sentences, args.speaker_id)

    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time"):
        _, mels, _, _, mel_lengths = tacotron.infer(sequences, text_lengths)

    # wavernn.generate(mels + hp.mel_bias, 'outputs/eval_wave_.wav', batched, target, overlap, hp.mu_law)

    tacotron2_infer_perf = mels.size(0)*mels.size(2)/measurements['tacotron2_time']

    LOGGER.log(key="tacotron2_frames_per_sec", value=tacotron2_infer_perf)
    LOGGER.log(key="tacotron2_latency", value=measurements['tacotron2_time'])

    # recover to the original order and concatenate
    ids_sorted_decreasing = ids_sorted_decreasing.numpy().tolist()
    mels = [mel[:, :length] for mel, length in zip(mels, mel_lengths)]
    mels = [mels[ids_sorted_decreasing.index(i)] for i in range(len(ids_sorted_decreasing))]
    # wav = audio.inv_mel_spectrogram(np.concatenate(mels, axis=-1))
    # audio.save_wav(wav, os.path.join(args.output, 'eval_gl.wav'))
    # for i in range(len(mels)):
    #     np.save(os.path.join(args.output, f'eval_mel_{i}.npy'), mels[i], allow_pickle=False)
    # np.save(os.path.join(args.output, f'eval_mel0.npy'), mels[0], allow_pickle=False)
    np.save(os.path.join(args.output, f'eval_mel1.npy'), np.concatenate(mels, axis=-1), allow_pickle=False)

    with torch.no_grad(), MeasureTime(measurements, "wavernn_time"):
        pcm = wavernn.generate(torch.tensor(np.concatenate(mels, axis=-1)).unsqueeze(0) + hp.mel_bias, 'outputs/eval_wave_long.wav', batched, target, overlap, hp.mu_law)

    LOGGER.log(key="wavernn_latency", value=measurements['wavernn_time'])
    LOGGER.log(key="latency", value=(measurements['tacotron2_time'] + measurements['wavernn_time']))
    LOGGER.iteration_stop()
    LOGGER.finish()
    # for i in range(len(mels)):
    #     mel = torch.tensor(mels[i]).unsqueeze(0)
    #     mel += hp.mel_bias
    #     save_str = f'outputs/eval_wave_{i}.wav'
    #
    #     wavernn.generate(mel, save_str, batched, target, overlap, hp.mu_law)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
