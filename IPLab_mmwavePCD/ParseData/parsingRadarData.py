"""
Author: USTC IP_LAB millimeter wave point cloud group
Members: Ruixu Geng (gengruixu@mail.ustc.edu.cn), Yadong Li, Jincheng Wu, Yating Gao
Copyright: USTC IP_LAB, 2023
Date: March 2023

This Python file provides utility functions to process and visualize millimeter wave radar point cloud data. It includes the following main functions: 'parse_radar_config', 'radar_process_frame', and 'read_radar_frame'.

The 'parse_radar_config' function parses the configuration file for a given sensor and parameter, initializes radar configuration objects, and radar objects with the parsed data.

The 'radar_process_frame' function processes the time-domain radar data into a complex frame format. It reshapes and transposes the raw data to match the expected structure for further processing.

The 'read_radar_frame' function reads radar frame data from a file, processes the raw data, and converts it into point cloud data using 'adc2pcd' function from the 'radarPcdProcessing' module.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict as edict
from radarPcdProcessing import adc2pcd, adc2pcd_peakdetection
import os
import pdb

def parse_radar_config(sensor="1843", parameter="coloradar"):
    """
    This function parses the configuration file for a given sensor and parameter,
    initializes radar configuration object and radar object with the parsed data.

    Args:
        sensor (str, optional): Sensor ID. Defaults to "1843".
        parameter (str, optional): Parameter type. Defaults to "coloradar".

    Returns:
        tuple: radar_config (dict) and radarObj (dict) containing parsed data and initialized parameters.
    """
    
    config_file_name = sensor + "_" + parameter + ".yml"

    current_file_dir = os.path.dirname(os.path.abspath(__file__)) # 'E:\\workspace\\ruixu\\SelfMadePackage\\IPLab_mmwavePCD\\ParseData'
    parent_dir = os.path.dirname(current_file_dir)
    config_file_path = os.path.join(parent_dir, "config", config_file_name)
    with open(config_file_path, 'r', encoding="utf-8") as fid:
        radar_config = edict(yaml.load(fid, Loader=yaml.FullLoader))
    # radar_config.numChirpsPerFrame
    radar_config.secondsPerFile = radar_config.framesPerFile / radar_config.frameRate

    radar_config.numAntenna = radar_config.numTxChan * radar_config.numRxChan
    radar_config.chirpRampTime = radar_config.SamplePerChripUp / radar_config.Fs
    # radar_config.Fs = radar_config.SamplePerChripUp / radar_config.chirpRampTime
    radar_config.chirpBandwidth = radar_config.Kr * radar_config.chirpRampTime
    radar_config.rangeBinSize = (3e8 * radar_config.chirpRampTime * radar_config.Fs) / (2 * radar_config.chirpBandwidth * radar_config.range_fftsize)
    radar_config.wavelength = 3e8 / (radar_config.StartFrequency + (radar_config.adc_start_time + ((radar_config.SamplePerChripUp / (radar_config.Fs)) / 2) ) * radar_config.Kr)
    # pdb.set_trace()
    # radar_config.velocityBinSize = 3e8 / (2 * radar_config.StartFrequency * radar_config.chirpRampTime * radar_config.doppler_fftsize)

    radar_config.frame_time = (radar_config.Ideltime + radar_config.chirpRampTime + radar_config.adc_start_time) * radar_config.numChirpsPerFrame
    radar_config.max_velocity = radar_config.wavelength / (4 * radar_config.frame_time / radar_config.numChirpsPerFrame)
    radar_config.velocityBinSize = radar_config.max_velocity / radar_config.doppler_fftsize
    
    radarObj = {}
    radarObj['frameRate'] = radar_config.frameRate
    radarObj['framesPerFile'] = radar_config.framesPerFile
    radarObj['secondsPerFile'] = radar_config.secondsPerFile

    radarObj['gAdcOneSampleSize'] = 4
    radarObj['numAdcSamples'] = radar_config.numAdcSamples
    radarObj['numRxChan'] = radar_config.numRxChan
    radarObj['numChirpsPerFrame'] = radar_config.numTxChan * radar_config.numChirpsPerFrame * radarObj['framesPerFile']
    radarObj['nLoopsIn1Frame'] = radar_config.numChirpsPerFrame
    radarObj['nChirpsIn1Loop'] = radar_config.numTxChan
    radarObj['range_fftsize'] = radar_config.range_fftsize
    radarObj['doppler_fftsize'] = radar_config.doppler_fftsize
    radarObj['angle_fftsize'] = radar_config.angle_fftsize
    radarObj['frameComplex'] = np.zeros(
        (radarObj['numChirpsPerFrame'], radarObj['numRxChan'], radarObj['numAdcSamples']), dtype=complex)   # 3*128, 4, 128
    radarObj['frameComplexFinal'] = np.zeros(
        (radarObj['framesPerFile'], radarObj['nLoopsIn1Frame'], radarObj['nChirpsIn1Loop'], radarObj['numRxChan'], radarObj['numAdcSamples']),
        dtype=complex)  # 128, 3, 4, 128
    
    radarObj['framesComplexFinal'] = np.zeros(
        (radarObj['framesPerFile'], radarObj['nLoopsIn1Frame'], radarObj['nChirpsIn1Loop'], radarObj['numRxChan'], radarObj['numAdcSamples']),
        dtype=complex)   # 10, 128, 3, 4, 128
    return radar_config, radarObj

def radar_process_frame(radarObj, timeDomainData, frame_idx=0):
    frameComplex = radarObj['frameComplex']
    frameComplexFinal = radarObj['frameComplexFinal']
    framesComplexFinal = radarObj['framesComplexFinal']
    frameRate = radarObj['frameRate']
    framesPerFile = radarObj['framesPerFile']
    secondsPerFile = radarObj['secondsPerFile']

    numAdcSamples = radarObj['numAdcSamples']
    numRxChan = radarObj['numRxChan']
    numChirpsPerFrame = radarObj['numChirpsPerFrame']
    nLoopsIn1Frame = radarObj['nLoopsIn1Frame']
    nChirpsIn1Loop = radarObj['nChirpsIn1Loop']


    rawData4 = np.reshape(timeDomainData, (4, int(timeDomainData.size / 4)), order='F') # (4, 983040)
    rawDataI = np.reshape(rawData4[0:2, :], (-1, 1), order='F')
    rawDataQ = np.reshape(rawData4[2:4, :], (-1, 1), order='F')
    frameCplx = rawDataI + 1j * rawDataQ # (1966080, 1)
    # pdb.set_trace()
    frameCplxTemp = np.reshape(frameCplx, (numAdcSamples * numRxChan, numChirpsPerFrame), order='F')    # 128*4, 3*128 | (512, 6000)
    frameCplxTemp = np.transpose(frameCplxTemp, (1, 0)) # (6000, 512)
    for jj in range(0, numChirpsPerFrame, 1):
        frameComplex[jj, :, :] = np.transpose(
            np.reshape(frameCplxTemp[jj, :], (numAdcSamples, numRxChan), order='F'), (1, 0))
    # =================================================================
    # for nFrame in range(0,framesPerFile,1):
    #     for nLoop in range(0, nLoopsIn1Frame, 1):
    #         for nChirp in range(0, nChirpsIn1Loop, 1):
    #             frameComplexFinal[nFrame, nLoop, nChirp, :, :] = frameComplex[nFrame * nLoopsIn1Frame * nChirpsIn1Loop + nLoop * nChirpsIn1Loop + nChirp, :, :]
    #             # frameComplexFinal[nLoop, nChirp, :, :] = frameComplex[nLoop * nChirpsIn1Loop + nChirp, :, :]
    # # frameComplexFinalTmp = np.transpose(frameComplexFinal, (3, 0, 2, 1))
    # frameComplexFinalTmp = np.transpose(frameComplexFinal, (0, 4, 1, 3, 2))
    # =================================================================
    # pdb.set_trace()
    nFrame = frame_idx  # (3840, 4, 128)
    for nLoop in range(0, nLoopsIn1Frame, 1):
        for nChirp in range(0, nChirpsIn1Loop, 1):
            frameComplexFinal[nFrame, nLoop, nChirp, :, :] = frameComplex[nFrame * nLoopsIn1Frame * nChirpsIn1Loop + nLoop * nChirpsIn1Loop + nChirp, :, :]
    frameComplexFinalTmp = np.transpose(frameComplexFinal, (0, 4, 1, 3, 2))
    # pdb.set_trace()
    return frameComplexFinalTmp[nFrame, :, :, :, :].squeeze()


def radar_process_frame_new(radarObj, timeDomainData, frame_idx=0):
    framesPerFile = radarObj['framesPerFile']
    lengthPerFrame = int(len(timeDomainData) / framesPerFile)
    targetFrameData = timeDomainData[frame_idx * lengthPerFrame: (frame_idx + 1) * lengthPerFrame]
    timeDomainData = targetFrameData

    numChirpsPerFrame = int(radarObj['numChirpsPerFrame'] / framesPerFile)
    frameComplex = np.zeros((numChirpsPerFrame, radarObj['numRxChan'], radarObj['numAdcSamples']), dtype=complex)
    frameComplexFinal = np.zeros((radarObj['nLoopsIn1Frame'], radarObj['nChirpsIn1Loop'], radarObj['numRxChan'], radarObj['numAdcSamples']),dtype=complex)

    numAdcSamples = radarObj['numAdcSamples']
    numRxChan = radarObj['numRxChan']
    
    nLoopsIn1Frame = radarObj['nLoopsIn1Frame']
    nChirpsIn1Loop = radarObj['nChirpsIn1Loop']
    
    rawData4 = np.reshape(timeDomainData, (4, int(timeDomainData.size / 4)), order='F') # (4, 98304)
    rawDataI = np.reshape(rawData4[0:2, :], (-1, 1), order='F')
    rawDataQ = np.reshape(rawData4[2:4, :], (-1, 1), order='F')
    frameCplx = rawDataI + 1j * rawDataQ # (196608, 1)
    frameCplxTemp = np.reshape(frameCplx, (numAdcSamples * numRxChan, numChirpsPerFrame), order='F')
    frameCplxTemp = np.transpose(frameCplxTemp, (1, 0)) # (384, 512)
    for jj in range(0, numChirpsPerFrame, 1):
        frameComplex[jj, :, :] = np.transpose(
            np.reshape(frameCplxTemp[jj, :], (numAdcSamples, numRxChan), order='F'), (1, 0))
    pdb.set_trace()
    for nLoop in range(0, nLoopsIn1Frame, 1):
        for nChirp in range(0, nChirpsIn1Loop, 1):
            frameComplexFinal[nLoop, nChirp, :, :] = frameComplex[nLoop * nChirpsIn1Loop + nChirp, :, :]
    frameComplexFinalTmp = np.transpose(frameComplexFinal, (3, 0, 2, 1))

    return frameComplexFinalTmp

def radar_process_frames(radarObj, timeDomainData, frame_idx=0):
    frameComplex = radarObj['frameComplex']
    frameComplexFinal = radarObj['frameComplexFinal']
    framesComplexFinal = radarObj['framesComplexFinal']
    frameRate = radarObj['frameRate']
    framesPerFile = radarObj['framesPerFile']
    secondsPerFile = radarObj['secondsPerFile']

    numAdcSamples = radarObj['numAdcSamples']
    numRxChan = radarObj['numRxChan']
    numChirpsPerFrame = radarObj['numChirpsPerFrame']
    nLoopsIn1Frame = radarObj['nLoopsIn1Frame']
    nChirpsIn1Loop = radarObj['nChirpsIn1Loop']


    rawData4 = np.reshape(timeDomainData, (4, int(timeDomainData.size / 4)), order='F')
    rawDataI = np.reshape(rawData4[0:2, :], (-1, 1), order='F')
    rawDataQ = np.reshape(rawData4[2:4, :], (-1, 1), order='F')
    frameCplx = rawDataI + 1j * rawDataQ
    # pdb.set_trace()
    frameCplxTemp = np.reshape(frameCplx, (numAdcSamples * numRxChan, numChirpsPerFrame), order='F')    # 128*4, 3*128 | (512, 6000)
    frameCplxTemp = np.transpose(frameCplxTemp, (1, 0)) # (6000, 512)
    for jj in range(0, numChirpsPerFrame, 1):
        frameComplex[jj, :, :] = np.transpose(
            np.reshape(frameCplxTemp[jj, :], (numAdcSamples, numRxChan), order='F'), (1, 0))
    # =================================================================
    for nFrame in range(0,framesPerFile,1):
        for nLoop in range(0, nLoopsIn1Frame, 1):
            for nChirp in range(0, nChirpsIn1Loop, 1):
                frameComplexFinal[nFrame, nLoop, nChirp, :, :] = frameComplex[nFrame * nLoopsIn1Frame * nChirpsIn1Loop + nLoop * nChirpsIn1Loop + nChirp, :, :]
                # frameComplexFinal[nLoop, nChirp, :, :] = frameComplex[nLoop * nChirpsIn1Loop + nChirp, :, :]
    # frameComplexFinalTmp = np.transpose(frameComplexFinal, (3, 0, 2, 1))
    frameComplexFinalTmp = np.transpose(frameComplexFinal, (0, 4, 1, 3, 2))
    # =================================================================
    # nFrame = frame_idx
    # for nLoop in range(0, nLoopsIn1Frame, 1):
    #     for nChirp in range(0, nChirpsIn1Loop, 1):
    #         frameComplexFinal[nFrame, nLoop, nChirp, :, :] = frameComplex[nFrame * nLoopsIn1Frame * nChirpsIn1Loop + nLoop * nChirpsIn1Loop + nChirp, :, :]
    # frameComplexFinalTmp = np.transpose(frameComplexFinal, (0, 4, 1, 3, 2))
    # pdb.set_trace()
    return frameComplexFinalTmp

def read_radar_frame(radarfilename_old, radarObj,radar_config):
    
    radarfilename, frame_idx = process_radar_filename(radarfilename_old)
    # int16: Old Version
    # complex128: New version
    radar_adc_data = np.fromfile(radarfilename,dtype = "complex128")
    # radar_adc_data = radar_process_frame(radarObj, radar_frame_data, frame_idx) # (128, 128, 4, 3)
    # radar_adc_data_new = radar_process_frame_new(radarObj, radar_frame_data, frame_idx)
    # radar_adc_data_s = radar_process_frames(radarObj, radar_frame_data, frame_idx) # (10, 128, 128, 4, 3)
    # radar_adc_data == radar_adc_data_s[frame_idx, :, :, :, :]  # True
    return radar_adc_data, frame_idx

def read_radar_frames(radarfilename_old, radarObj,radar_config):
    # pdb.set_trace()
    radarfilename, frame_idx = process_radar_filename(radarfilename_old)
    # int16: Old Version
    # complex128: New version
    radar_frame_data = np.fromfile(radarfilename,dtype = "complex128")
    radar_adc_data = radar_process_frames(radarObj, radar_frame_data, frame_idx)
    return radar_adc_data, frame_idx

def process_radar_filename(filename):
    if not filename.endswith(".bin"):
        raise ValueError("Input should be a .bin file")

    path, file_name_ori = os.path.split(filename)

    underscore_count = file_name_ori.count("_")
    frame_index = 0
    if underscore_count == 2:
        file_name = "_".join(file_name_ori.split("_")[:-1]) + ".bin"
        # pdb.set_trace()
        frame_index = int(file_name_ori.split("_")[-1][:-4])

        new_filename = os.path.join(path, file_name)

        return new_filename, frame_index
    
    else:
        return filename, frame_index

def read_process_save_radar_frame(radar_filename):
    # pdb.set_trace()
    try:
        dataset_label = os.path.split(os.path.split(os.path.split(radar_filename)[0])[0])[1].split("_")[-1]
    except:
        pdb.set_trace()
    radar_config, radarObj = parse_radar_config(sensor="1843", parameter=dataset_label)
    
    # pdb.set_trace()
    radar_adc_data, frame_idx = read_radar_frame(radar_filename, radarObj,radar_config)    # [frame_index, sample_index, chirp_index, rx_index, tx_index]
    rangeFFTOut, DopplerFFTOut, point_cloud = adc2pcd(radar_adc_data,radar_config)
    point_cloud = point_cloud[:, [0,1,2,5]].astype(np.float32)
    # rangeFFTOut, DopplerFFTOut, point_cloud = adc2pcd_peakdetection(radar_adc_data,radar_config)
    # point_cloud = point_cloud[:, [0,1,2,3]].astype(np.float32)


    save_file_name = radar_filename.replace('1843', '1843_pcd', 1)
    point_cloud.tofile(save_file_name)
    print(save_file_name + " saved!")

    # Use the following code to load the saved data:
    # cloud_vals = np.fromfile(save_file_name,dtype = 'float32')
    # cloud = cloud_vals.reshape((-1,4))
