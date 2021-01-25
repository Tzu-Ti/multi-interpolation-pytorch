import numpy as np

def pixelDownShuffle(img_tensor, patch_size):
    """
    :param img_tensor: input origin size sequence tensor []
    :param patch_size: PixelShuffle parameter
    :return: down shuffle tensor [batch_size, seq_length, patch_size*patch_size*img_channel, img_size//patch_size, img_size//patch_size]
    """
    
    shape = np.shape(img_tensor)
    batch_size = shape[0]
    seq_length = shape[1]
    img_channel = shape[2]
    img_size = shape[3]
    
    img_tensor = np.transpose(img_tensor, [0,1,3,4,2]) # transpose to [batch_size, seq_length, img_size, img_size, img_channel]
    
    # pixel down shuffle
    a = np.reshape(img_tensor, [batch_size, seq_length,
                                img_size//patch_size, patch_size,
                                img_size//patch_size, patch_size, img_channel])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_size//patch_size,
                                  img_size//patch_size, patch_size*patch_size*img_channel])
    ###
    
    patch_tensor = np.transpose(patch_tensor, [0,1,4,2,3]) # transpose back to [batch_size, seq_length, patch_size*patch_size*img_channel, img_size//patch_size, img_size//patch_size]
    
    return patch_tensor

def pixelUpShuffle(patch_tensor, patch_size):
    """
    :param img_tensor: input downshuffled sequence tensor [batch_size, seq_length, patch_size*patch_size*img_channel, img_size//patch_size, img_size//patch_size]
    :param patch_size: PixelShuffle parameter
    :return: down shuffle tensor [batch_size, seq_length, img_channel, img_size, img_size]
    """
    
    shape = np.shape(patch_tensor)
    batch_size = shape[0]
    seq_length = shape[1]
    patch_channel = shape[2]
    patch_width = shape[3]
    
    patch_tensor = np.transpose(patch_tensor, [0,1,3,4,2]) # transpose to [batch_size, seq_length, img_size//patch_size, img_size//patch_size, patch_size*patch_size*img_channel]
    
    # pixel up shuffle
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_width, patch_width,
                                  patch_size, patch_size,
                                  patch_channel // (patch_size * patch_size)])
    b = np.transpose(a, [0,1,2,4,3,5,6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_width * patch_size,
                                patch_width * patch_size,
                                patch_channel // (patch_size * patch_size)])
    ###
    
    img_tensor = np.transpose(img_tensor, [0,1,4,2,3]) # transpose back to [batch_size, seq_length, img_channel, img_size, img_size]
    
    return img_tensor
