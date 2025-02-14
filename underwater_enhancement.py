import cv2
import numpy as np

def histogram_spread(channel):
    hist, _ = np.histogram(channel, bins=256, range=(0, 1))
    return np.std(hist)

def LACC(input_img: np.ndarray, is_vid=False, is_run=False):
    """
    Locally Adaptive Color Correction.

    Parameters:
    - img (numpy.ndarray): a 3-channel color image (BGR). values in the range [0, 1]
    - is_vid (bool): check is it video
    - is_run (bool): check is it first run. (for video)

    Returns:
    - LACC_img (numpy.ndarray): color corrected img. values in the range [0, 1]
    - is_run
    """

    ## zip [(img_mean, img)], it (b, g, r)
    small, medium, large = sorted(list(zip(cv2.mean(input_img), cv2.split(input_img), ['b', 'g', 'r'])))
    ## sorted by mean (small to large)
    small, medium, large = list(small), list(medium), list(large)

    ## exchange wrong channel
    if is_vid and not is_run:
        if histogram_spread(medium[1]) < histogram_spread(large[1]) and (large[0] - medium[0]) < 0.07 and small[2] == 'r':
            large, medium = medium, large
            print('exchange!')
        is_run = True

    elif not is_vid:
        if histogram_spread(medium[1]) < histogram_spread(large[1]) and (large[0] - medium[0]) < 0.07 and small[2] == 'r':
            large, medium = medium, large

    ## Max attenuation
    max_attenuation = 1 - (small[1]**1.2)
    max_attenuation = np.expand_dims(max_attenuation, axis=2)

    ## Detail image
    blurred_image = cv2.GaussianBlur(input_img, (7, 7), 0)
    detail_image = input_img - blurred_image

    ## corrected large channel
    large[1] = (large[1] - cv2.minMaxLoc(large[1])[0]) * (1/(cv2.minMaxLoc(large[1])[1] - cv2.minMaxLoc(large[1])[0]))
    large[0] = cv2.mean(large[1])[0]

    ## Iter corrected
    loss = float('inf')
    while loss > 1e-2:
        medium[1] = medium[1] + (large[0] - cv2.mean(medium[1])[0]) * large[1]
        small[1] = small[1] + (large[0] - cv2.mean(small[1])[0]) * large[1]
        loss = abs(large[0] - cv2.mean(medium[1])[0]) + abs(large[0] - cv2.mean(small[1])[0])

    ## b, g, r combine
    for _, ch, color in [large, medium, small]:
        if color == 'b':
            b_ch = ch
        elif color == 'g':
            g_ch = ch
        else:
            r_ch = ch
    img_corrected = cv2.merge([b_ch, g_ch, r_ch])

    ## LACC Result
    LACC_img = detail_image + (max_attenuation * img_corrected) + ((1 - max_attenuation) * input_img)
    LACC_img = np.clip(LACC_img, 0.0, 1.0)

    return LACC_img, is_run

def process_block(block, lc_variance, block_mean, block_variance, beta):
    # Check for block_variance being zero to avoid divide by zero error
    if block_variance != 0:
        alpha = lc_variance / block_variance
    else:
        # Set alpha to 0 or another appropriate value when block_variance is zero
        alpha = 0

    # Adjust the block based on the comparison between alpha and beta
    if alpha < beta:
        block = block_mean + (alpha * (block - block_mean))
    else:
        block = block_mean + (beta * (block - block_mean))

    return block

def LACE(input_img: np.ndarray, beta: float):
    """
    Locally Adaptive Contrast Enhancement.

    Parameters:
    - img (numpy.ndarray): a 3-channel color image (BGR). values in the range [0, 255]

    Returns:
    - LACE_img (numpy.ndarray): contrast enhancement img. values in the range [0, 255]
    """
    ## Process input image
    input_img = input_img.astype(np.uint8)

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(input_img)

    ## Set parament
    block_size = 25
    beta = beta # Enhancement value
    radius = 10
    eps = 0.01

    ## Assuming l_channel
    lc_variance = np.var(l_channel)
    integral_sum, integral_sqsum = cv2.integral2(l_channel)
    height, width = l_channel.shape

    l_channel_processed = np.zeros_like(l_channel, dtype=np.float64)
    weight_sum = np.zeros_like(l_channel, dtype=np.float64)


    ## Process each block
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            ## Define block boundaries
            start_i = i
            end_i = min(i + block_size, height)
            start_j = j
            end_j = min(j + block_size, width)

            ## Extract block
            block = l_channel[start_i:end_i, start_j:end_j]

            ## Cal block var, mean
            block_sum = integral_sum[end_i, end_j] - integral_sum[start_i, end_j] - integral_sum[end_i, start_j] + integral_sum[start_i, start_j]
            block_mean = block_sum / ((end_i - start_i) * (end_j - start_j))
            block_sum_sq = integral_sqsum[end_i, end_j] - integral_sqsum[start_i, end_j] - integral_sqsum[end_i, start_j] + integral_sqsum[start_i, start_j]
            block_variance = block_sum_sq / ((end_i - start_i) * (end_j - start_j)) - np.square(block_mean)

            ## Process block
            block_processed = process_block(block, lc_variance, block_mean, block_variance, beta)

            ## Put block back into image
            l_channel_processed[start_i:end_i, start_j:end_j] += block_processed
            weight_sum[start_i:end_i, start_j:end_j] += 1.0

    l_channel_processed /= weight_sum
    l_channel_processed = np.clip(l_channel_processed, 0, 255).astype('uint8')

    ## guided (need install opencv-contrib-python)
    l_channel_processed = cv2.ximgproc.guidedFilter(l_channel, l_channel_processed, radius, eps)

    ## ab channel balance
    a_mean = np.mean(a_channel)
    b_mean = np.mean(b_channel)
    if a_mean > b_mean:
        b_channel = (b_channel + b_channel * ((a_mean - b_mean) / (a_mean + b_mean))).astype(np.uint8)
    else:
        a_channel = (a_channel + a_channel * ((b_mean - a_mean)/(a_mean + b_mean))).astype(np.uint8)

    ## Combine channel
    Result = cv2.merge([l_channel_processed, a_channel, b_channel])
    Result = cv2.cvtColor(Result, cv2.COLOR_LAB2BGR)
    return Result

def fusion(input_img):
    '''
    input[0, 1]
    output[0, 1]
    '''
    sigma = 20
    ksize = 4 * sigma + 1

    ## blur
    Igauss_blurred = cv2.GaussianBlur(input_img, (ksize, ksize), 20)

    ## Norm
    gain = 0.3
    Norm = (input_img - gain * Igauss_blurred)

    ## Histogram Equalization
    for n in range(3):
        Norm[:, :, n] = cv2.equalizeHist((Norm[:, :, n] * 255).astype(np.uint8)) / 255.0

    ## Sharp
    I_sharp = (input_img + Norm) / 2

    ## Gamma correct
    gamma = 1.8
    I_gamma = np.power(input_img, gamma)

    ## BGR to Lab
    I_sharp = I_sharp.astype(np.float32)
    I_gamma = I_sharp.astype(np.float32)
    I_sharp_lab = cv2.cvtColor(I_sharp, cv2.COLOR_BGR2Lab)
    I_gamma_lab = cv2.cvtColor(I_gamma, cv2.COLOR_BGR2Lab)


    ## Laplacian weight (W_l)

    # For I_sharp
    R1 = I_sharp_lab[:, :, 0] / 255.0

    # For I_gamma
    R2 = I_gamma_lab[:, :, 0] / 255.0

    ## Saliency weight (W_S)
    W_S1 = saliency_detection(I_sharp)
    W_S1 = W_S1 / np.max(W_S1)

    W_S2 = saliency_detection(I_gamma)
    W_S2 = W_S2 / np.max(W_S2)

    W_C1 = np.abs(cv2.Laplacian(R1, cv2.CV_32F))
    W_C2 = np.abs(cv2.Laplacian(R2, cv2.CV_32F))

    ## Saturation weight (W_Sat)
    W_SAT1 = np.sqrt(1/3 * ((I_sharp[:,:,0] - R1)**2 +
                            (I_sharp[:,:,1] - R1)**2 +
                            (I_sharp[:,:,2] - R1)**2))

    W_SAT2 = np.sqrt(1/3 * ((I_gamma[:,:,0] - R2)**2 +
                            (I_gamma[:,:,1] - R2)**2 +
                            (I_gamma[:,:,2] - R2)**2))

    ## Normalized weight
    W1 = (W_C1 + W_S1 + W_SAT1 + 0.1) / (W_C1 + W_S1 + W_SAT1 + W_C2 + W_S2 + W_SAT2 + 0.2)
    W2 = (W_C2 + W_S2 + W_SAT2 + 0.1) / (W_C1 + W_S1 + W_SAT1 + W_C2 + W_S2 + W_SAT2 + 0.2)

    ## gaussian pyramid
    level = 8
    Weight_1 = gaussian_pyramid(W1, level)
    Weight_2 = gaussian_pyramid(W2, level)

    ## laplacian pyramid
    B1 = laplacian_pyramid(I_sharp[:, :, 0], level)
    G1 = laplacian_pyramid(I_sharp[:, :, 1], level)
    R1 = laplacian_pyramid(I_sharp[:, :, 2], level)

    ## gamma img
    B2 = laplacian_pyramid(I_gamma[:, :, 0], level)
    G2 = laplacian_pyramid(I_gamma[:, :, 1], level)
    R2 = laplacian_pyramid(I_gamma[:, :, 2], level)

    ## fusion
    Rr = []
    Rg = []
    Rb = []

    for k in range(level):
        Rr.append(Weight_1[k] * R1[k] + Weight_2[k] * R2[k])
        Rg.append(Weight_1[k] * G1[k] + Weight_2[k] * G2[k])
        Rb.append(Weight_1[k] * B1[k] + Weight_2[k] * B2[k])

    B = np.clip(pyramid_reconstruct(Rb), 0, 1)
    G = np.clip(pyramid_reconstruct(Rg), 0, 1)
    R = np.clip(pyramid_reconstruct(Rr), 0, 1)

    return cv2.merge([B, G, R])

def saliency_detection(img):
    # Gaussian blur
    gfrgb = cv2.GaussianBlur(img, (3,3), 0)

    # Convert image from BGR to Lab color space
    lab = cv2.cvtColor(gfrgb, cv2.COLOR_BGR2Lab).astype(np.float64)

    # Compute Lab average values
    l, a, b = cv2.split(lab)
    lm, am, bm = np.mean(l), np.mean(a), np.mean(b)

    # Compute the saliency map
    sm = (l-lm)**2 + (a-am)**2 + (b-bm)**2

    return sm

def gaussian_pyramid(img, level):
    h = np.array([1, 4, 6, 4, 1]) / 16
    filt = np.outer(h, h)
    out = []

    filtered_img = cv2.filter2D(img, -1, filt, borderType=cv2.BORDER_REPLICATE)
    out.append(filtered_img)

    temp_img = filtered_img
    for i in range(1, level):
        temp_img = temp_img[::2, ::2]  # Downsample by a factor of 2
        filtered_img = cv2.filter2D(temp_img, -1, filt, borderType=cv2.BORDER_REPLICATE)
        out.append(filtered_img)

    return out

def laplacian_pyramid(img, level):
    out = [img]
    temp_img = img

    for i in range(1, level):
        temp_img = temp_img[::2, ::2]  # Downsample by a factor of 2
        out.append(temp_img)

    for i in range(level - 1):
        m, n = out[i].shape
        upscaled = cv2.resize(out[i+1], (n, m), interpolation=cv2.INTER_LINEAR)
        out[i] = out[i] - upscaled

    return out

def pyramid_reconstruct(pyramid):
    level = len(pyramid)
    for i in range(level - 1, 0, -1):
        m, n = pyramid[i-1].shape[:2]
        upscaled = cv2.resize(pyramid[i], (n, m), interpolation=cv2.INTER_LINEAR)
        pyramid[i-1] += upscaled

    return pyramid[0]


