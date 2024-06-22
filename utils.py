import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

number_token_ids = {20489, 4118, 4122, 2079, 20522, 24622, 16431, 6192, 14388, 24636, 22594, 24643, 4165, 2122, 2128, 4177, 18517, 20579, 26724, 4198, 4201, 24700, 26750, 4225, 28803, 26755, 20615, 24719, 4240, 4241, 22671, 26774, 26782, 10402, 26793, 24748, 4271, 26811, 6348, 4305, 22744, 2266, 4314, 4327, 14574, 24816, 22772, 2294, 24837, 2313, 26898, 16660, 26901, 4389, 26918, 22823, 26920, 26937, 2368, 24898, 4433, 4440, 2394, 20829, 4448, 10593, 4450, 357, 26987, 8584, 2445, 20879, 27025, 24978, 20885, 20889, 4508, 20896, 2469, 22954, 16824, 22974, 4542, 4552, 8651, 4560, 2517, 2518, 27096, 27097, 4581, 2534, 4591, 2555, 20987, 27133, 4608, 2560, 4613, 519, 12814, 18959, 2577, 27156, 536, 2596, 25129, 25133, 27181, 2606, 27184, 12869, 14926, 16975, 591, 25174, 2658, 25188, 2668, 632, 4729, 2688, 21129, 27278, 23188, 15003, 23201, 27308, 10927, 19129, 25285, 21196, 2773, 27357, 25312, 4834, 755, 4853, 17147, 4867, 19208, 25356, 23330, 25379, 23335, 4906, 25398, 27452, 4928, 23360, 2884, 17225, 19277, 17246, 4959, 2915, 7015, 25449, 25463, 2938, 11138, 15239, 19337, 17304, 25502, 927, 23459, 940, 948, 21441, 5062, 21448, 15348, 19446, 27640, 3072, 27650, 3076, 19479, 25626, 27693, 27697, 27699, 5176, 17465, 27720, 17485, 9295, 15442, 27730, 21594, 3166, 27749, 23658, 17518, 27760, 19568, 13427, 21653, 23702, 15518, 23714, 19621, 11434, 1206, 3264, 25794, 23747, 23757, 13527, 3288, 25828, 3301, 9457, 11505, 5373, 27904, 3328, 3341, 1298, 27931, 23838, 15660, 5426, 3390, 25926, 25946, 3420, 19818, 3436, 3449, 25987, 25988, 25991, 23946, 21899, 23948, 28070, 5548, 5553, 1458, 17864, 26059, 3539, 21976, 21987, 19947, 15866, 22011, 26106, 22012, 11776, 19978, 19993, 24090, 22048, 26144, 3628, 20016, 3632, 28212, 3647, 3651, 13907, 24151, 20056, 24163, 24175, 26225, 3707, 3708, 3710, 18057, 22174, 9887, 3747, 11944, 1714, 3769, 20159, 26311, 28363, 22224, 26320, 24274, 20176, 1752, 1755, 5865, 24300, 9979, 26363, 16128, 3840, 1808, 1828, 28456, 16169, 24361, 20275, 3891, 26426, 22335, 3914, 3916, 22367, 3940, 3951, 6007, 3959, 24441, 12172, 10124, 4013, 4018, 26556, 18365, 22464, 26578, 4056, 4060, 26593, 22504, 10218, 12278, 22520}

special_token_ids = [55, 58, 233, 2824, 9374, 1300, 1682, 137, 13883, 15070, 16463, 16937]
# ! 55
# ? 58
# ... 233 
# .... 2824
# ..... 9374
# ). 137
# 1). 13883
# 2). 15070
# ...). 16463 
# ”). 16937

end_sentence_with_number_ids = [1300, 1682]
# 1. 1300
# 2. 1682

# when tokens[row_idx, col_idx] == '.' (5)
# .. 5 5
# u.s.a (if . occurs after . in the next two characters or three, then it's not eos) 
# 1.9 (if this is 1 . 9, then . is not eos)

# I am happy... I have a computer.

def check_eos(tokens, col_idx):
 
    mask = np.zeros(tokens.shape[0], dtype=np.bool_)
    for row_idx in range(tokens.shape[0]):
        
        # Soultion: space
        # Soultion: nothing

        # Solution: word
        # Solution: word.
        
        if col_idx > 0 and tokens[row_idx, col_idx] == 1 \
            and not tokens[row_idx, col_idx-1] in special_token_ids \
            and tokens[row_idx, col_idx-1] != 5 \
            and tokens[row_idx, col_idx-1] != 10:
            mask[row_idx] = True
        
        elif tokens[row_idx, col_idx] in end_sentence_with_number_ids and col_idx > 1 and tokens[row_idx, col_idx - 1] == 25072:
            mask[row_idx] = True
        # comma. Mr.
        elif tokens[row_idx, col_idx] == 5 and col_idx > 3 and tokens[row_idx, col_idx - 1] == 9 \
                and tokens[row_idx, col_idx - 2] == 51 and tokens[row_idx, col_idx - 3] == 287:
            mask[row_idx] = True
        elif tokens[row_idx, col_idx] in special_token_ids or tokens[row_idx, col_idx] == 5:
            # Solution: .
            if col_idx > 1 and tokens[row_idx, col_idx -1] == 10 and tokens[row_idx, col_idx -2] == 17942:
                continue

            #  Washington, D.C.,
            if col_idx > 2 and col_idx - 1 < tokens.shape[1] and tokens[row_idx, col_idx - 1] == 254 \
                and tokens[row_idx, col_idx - 2] == 5 and tokens[row_idx, col_idx - 3] == 309 and tokens[row_idx, col_idx + 1] == 6:
                continue

            #  St. Louis
            if col_idx > 0 and tokens[row_idx, col_idx - 1] == 472:
                continue

            #  Dr.
            if col_idx > 0 and tokens[row_idx, col_idx - 1] == 707:
                continue
            
            #  Mr.
            if col_idx > 0 and tokens[row_idx, col_idx - 1] == 1363:
                continue

            #  Mrs.
            if col_idx > 0 and tokens[row_idx, col_idx - 1] == 8667:
                continue


            if not ((col_idx < tokens.shape[1] - 1 and tokens[row_idx, col_idx + 1] == 5) \
                or (col_idx < tokens.shape[1] - 2 and tokens[row_idx, col_idx + 2] == 5) \
                or (col_idx < tokens.shape[1] - 3 and tokens[row_idx, col_idx + 3] == 5) \
                # or (col_idx < tokens.shape[1] - 4 and tokens[row_idx, col_idx + 4] == 5) \
                or (col_idx < tokens.shape[1] - 1 and tokens[row_idx, col_idx + 1].item() in number_token_ids)):

                mask[row_idx] = True

    return mask


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)