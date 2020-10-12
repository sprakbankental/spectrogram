import numpy as np
import torch

import logging

# wge2wgi (Waveglow example to Waveglow infer)
# From: 
# from the Waveglow example format listed at
# https://github.com/NVIDIA/waveglow
# The files are at
# https://drive.google.com/file/d/1g_VXK2lpP9J25dQFhQwx7doWl_p20fXA/view?usp=sharing
# To:
# The format accepted by waveglow.infer(mel)
# The former is two-dimensional and can be displayed directly using e.g. plt.imshow
# The latter is three-dimensional, and is accepted by Waveglow as input
# The conversion is intended to be fully lossless, but we haven't verified this properly.
def wge2wgi(mel_example):
  logging.info("From:", mel_example.shape, type(mel_example))
  mel_example = torch.autograd.Variable(mel_example.cuda())
  mel_example = torch.unsqueeze(mel_example, 0)
  #mel_example = mel_example.half() if is_fp16 else mel1
  print("To:", mel_example.shape, type(mel_example))
  return mel_example
# wgi2wge (Waveglow infer to Waveglow example )
def wgi2wge(waveglow):
  logging.info("From:", waveglow.shape, type(waveglow))
  waveglow = torch.squeeze(waveglow, 0)
  waveglow = waveglow.cpu()
  print("To:", waveglow.shape, type(waveglow))
  return waveglow