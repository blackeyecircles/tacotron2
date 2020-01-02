import numpy as np
import torch


# path0 = 'outputs/encoder_outputs0.pt'
# path1 = 'outputs/encoder_outputs1.pt'
#
# encoder_outputs0 = torch.load(path0).cpu().numpy()
# encoder_outputs1 = torch.load(path1).cpu().numpy()
# print((encoder_outputs0 == encoder_outputs1).all())
#
# path0 = 'outputs/mel_outputs_before0.pt'
# path1 = 'outputs/mel_outputs_before1.pt'
#
# mel_outputs_before0 = torch.load(path0).cpu().numpy()
# mel_outputs_before1 = torch.load(path1).cpu().numpy()
# print((mel_outputs_before0 == mel_outputs_before1).all())
#
# path0 = 'outputs/mel_outputs_after0.pt'
# path1 = 'outputs/mel_outputs_after1.pt'
#
# mel_outputs_after0 = torch.load(path0).cpu().numpy()
# mel_outputs_after1 = torch.load(path1).cpu().numpy()
# print((mel_outputs_after0 == mel_outputs_after1).all())
#
# path0 = 'outputs/output_mel0.npy'
# path1 = 'outputs/output_mel1.npy'
#
# output_mel0 = torch.load(path0).numpy()
# output_mel1 = torch.load(path1).numpy()
# print((output_mel0 == output_mel1).all())

# path0 = 'outputs/eval_mel0.npy'
# path1 = 'outputs/eval_mel1.npy'
#
# mel0 = np.load(path0)
# mel1 = np.load(path1)
# print((mel0 == mel1).all())

path0 = 'outputs/batched_mels0.pt'
path1 = 'outputs/batched_mels1.pt'

batched_mels0 = torch.load(path0).cpu().numpy()
batched_mels1 = torch.load(path1).cpu().numpy()
print((batched_mels0 == batched_mels1).all())

path0 = 'outputs/batched_aux0.pt'
path1 = 'outputs/batched_aux1.pt'

batched_aux0 = torch.load(path0).cpu().numpy()
batched_aux1 = torch.load(path1).cpu().numpy()
print((batched_aux0 == batched_aux1).all())

path0 = 'outputs/after_rnn1_0.pt'
path1 = 'outputs/after_rnn1_1.pt'

after_rnn1_0 = torch.load(path0).cpu().numpy()
after_rnn1_1 = torch.load(path1).cpu().numpy()
print((after_rnn1_0 == after_rnn1_1).all())

path0 = 'outputs/logits0.pt'
path1 = 'outputs/logits1.pt'

logits0 = torch.load(path0).cpu().numpy()
logits1 = torch.load(path1).cpu().numpy()
print((logits0 == logits1).all())

path0 = 'outputs/posterior0.pt'
path1 = 'outputs/posterior1.pt'

posterior0 = torch.load(path0).cpu().numpy()
posterior1 = torch.load(path1).cpu().numpy()
print(posterior0.argmax(-1))
print(posterior1.argmax(-1))
print(posterior0[3][403])
print(posterior1[3][403])
print('posterior:', (posterior0 == posterior1).all())

path0 = 'outputs/sample0.pt'
path1 = 'outputs/sample1.pt'

sample0 = torch.load(path0).cpu().numpy()
sample1 = torch.load(path1).cpu().numpy()
print(sample0)
print(sample1)
print('sample:', (sample0 == sample1))

path0 = 'outputs/x_after_sample0.pt'
path1 = 'outputs/x_after_sample1.pt'

x_after_sample0 = torch.load(path0).cpu().numpy()
x_after_sample1 = torch.load(path1).cpu().numpy()
print('x_after_sample:', (x_after_sample0 == x_after_sample1).all())


path0 = 'outputs/output_before_mulaw0.pt'
path1 = 'outputs/output_before_mulaw1.pt'

output_before_mulaw0 = torch.load(path0)
output_before_mulaw1 = torch.load(path1)
tmp = output_before_mulaw0.argmax(0) - output_before_mulaw1.argmax(0)
print([i for i, v in enumerate(tmp) if v != 0])
print(output_before_mulaw0[:, 2228])
print(output_before_mulaw1[:, 2228])
print(output_before_mulaw0.argmax(0)[2228])
print(output_before_mulaw1.argmax(0)[2228])
print((output_before_mulaw0 == output_before_mulaw1).all())

# path0 = 'outputs/output_batched0.pt'
# path1 = 'outputs/output_batched1.pt'
#
# output_batched0 = torch.load(path0)
# output_batched1 = torch.load(path1)
# print((output_batched0 == output_batched1).all())
#
path0 = 'outputs/eval_pcm0.npy'
path1 = 'outputs/eval_pcm1.npy'

pcm0 = np.load(path0)
pcm1 = np.load(path1)
print((pcm0 == pcm1).all())
# #
# a = np.array([1,2,3])
# b = np.array([1,2,3])

# print((a==b).all())