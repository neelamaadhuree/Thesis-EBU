import numpy as np
import cv2

def RGB2YUV(x_rgb):
    x_yuv = np.zeros(x_rgb.shape, dtype=float)
    for i in range(x_rgb.shape[0]):
        img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        x_yuv[i] = img
    return x_yuv

def YUV2RGB(x_yuv):
    x_rgb = np.zeros(x_yuv.shape, dtype=float)
    for i in range(x_yuv.shape[0]):
        img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        x_rgb[i] = img
    return x_rgb

def DCT(x_train, window_size):
    x_dct = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3]), dtype=float)
    for i in range(x_train.shape[0]):
        for ch in range(x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_dct = cv2.dct(x_train[i, ch, w:w+window_size, h:h+window_size].astype(float))
                    x_dct[i, ch, w:w+window_size, h:h+window_size] = sub_dct
    return x_dct

def IDCT(x_train, window_size):
    x_idct = np.zeros(x_train.shape, dtype=float)
    for i in range(x_train.shape[0]):
        for ch in range(x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_idct = cv2.idct(x_train[i, ch, w:w+window_size, h:h+window_size].astype(float))
                    x_idct[i, ch, w:w+window_size, h:h+window_size] = sub_idct
    return x_idct

def poison_frequency_direct(x_train, args):
    freq_domain_channel_list = [0, 1, 2]
    pos_list = [(31, 31), (15, 15)]
    if x_train.shape[0] == 0:
        return x_train
    x_train = x_train.astype(float)
    x_train *= 255.0
    if args.freq_domain_yuv:
        x_train = RGB2YUV(x_train)

    x_train = DCT(x_train, args.freq_domain_window_size)

    for i in range(x_train.shape[0]):
        for ch in freq_domain_channel_list:
            for w in range(0, x_train.shape[2], args.freq_domain_window_size):
                for h in range(0, x_train.shape[3], args.freq_domain_window_size):
                    for pos in pos_list:
                        x_train[i, ch, w + pos[0], h + pos[1]] += args.freq_domain_magnitude

    x_train = IDCT(x_train, args.freq_domain_window_size)

    if args.freq_domain_yuv:
        x_train = YUV2RGB(x_train)
    x_train /= 255.0
    x_train = np.clip(x_train, 0, 1)
    return x_train
