import numpy as np
import torch


path0 = 'outputs/encoder_outputs0.pt'
path1 = 'outputs/encoder_outputs1.pt'

encoder_outputs0 = torch.load(path0).cpu().numpy()
encoder_outputs1 = torch.load(path1).cpu().numpy()
print((encoder_outputs0 == encoder_outputs1).all())

path0 = 'outputs/mel_outputs_before0.pt'
path1 = 'outputs/mel_outputs_before1.pt'

mel_outputs_before0 = torch.load(path0).cpu().numpy()
mel_outputs_before1 = torch.load(path1).cpu().numpy()
print((mel_outputs_before0 == mel_outputs_before1).all())

path0 = 'outputs/mel_outputs_after0.pt'
path1 = 'outputs/mel_outputs_after1.pt'

mel_outputs_after0 = torch.load(path0).cpu().numpy()
mel_outputs_after1 = torch.load(path1).cpu().numpy()
print((mel_outputs_after0 == mel_outputs_after1).all())

path0 = 'outputs/output_mel0.npy'
path1 = 'outputs/output_mel1.npy'

output_mel0 = torch.load(path0).numpy()
output_mel1 = torch.load(path1).numpy()
print((output_mel0 == output_mel1).all())

path0 = 'outputs/eval_mel0.npy'
path1 = 'outputs/eval_mel1.npy'

mel0 = np.load(path0)
mel1 = np.load(path1)
print((mel0 == mel1).all())
#
# a = np.array([1,2,3])
# b = np.array([1,2,3])
# print((a==b).all())