import torch
import os


def select_device(schedual, log):
    CUDA_VISIBLE_DEVICES = ''
    if 'CUDA_VISIBLE_DEVICES' in schedual:
        CUDA_VISIBLE_DEVICES = schedual['CUDA_VISIBLE_DEVICES']
    else:
        CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
    log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

    if CUDA_VISIBLE_DEVICES == '':
        raise ValueError(f'This machine has no visible cuda devices!')

    CUDA_SELECTED_DEVICES = ''
    if 'CUDA_SELECTED_DEVICES' in schedual:
        CUDA_SELECTED_DEVICES = schedual['CUDA_SELECTED_DEVICES']
    else:
        CUDA_SELECTED_DEVICES = CUDA_VISIBLE_DEVICES
    log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')

    CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
    CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))

    CUDA_VISIBLE_DEVICES_SET = set(CUDA_VISIBLE_DEVICES_LIST)
    CUDA_SELECTED_DEVICES_SET = set(CUDA_SELECTED_DEVICES_LIST)
    if not (CUDA_SELECTED_DEVICES_SET <= CUDA_VISIBLE_DEVICES_SET):
        raise ValueError(f'CUDA_VISIBLE_DEVICES should be a subset of CUDA_VISIBLE_DEVICES!')
    
    GPU_num = len(CUDA_SELECTED_DEVICES_SET)
    device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(CUDA_SELECTED_DEVICE) for CUDA_SELECTED_DEVICE in CUDA_SELECTED_DEVICES_LIST]
    
    return GPU_num, device_ids
