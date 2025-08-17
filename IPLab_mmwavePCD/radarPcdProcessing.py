import open3d as o3d
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import pdb
try:
    import torch
except:
    print("Warning: PyTorch is not installed")
try:
  import findpeaks
except:
  print("Warning: findpeaks not found, unable to run point cloud generation based on peak detection.")

import numpy as np
import open3d as o3d
import scipy.linalg as LA


def reshape_fortran(x, shape):
    """
    This function reshapes a given input tensor in Fortran (column-major) order.
    
    :param x: Input tensor to be reshaped
    :param shape: A tuple containing the desired shape
    :return: Reshaped tensor
    """
    if len(x.shape) > 0:
        # Reverse the order of dimensions in the input tensor
        x = x.permute(*reversed(range(len(x.shape))))

    # Reshape the tensor and reverse the order of dimensions in the output tensor
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def rangeFFT(input, opt):
    """
    Performs range FFT on the input data using the specified options.

    :param input: Input data array (numpy array)
    :param opt: Options including range_fftsize
    :return: Array with applied range FFT (numpy array)
    """
    fftsize = opt.range_fftsize
    rangeWindowCoeffVec = np.hanning(input.shape[0] + 2)[1:-1]
    numLines = input.shape[1]
    numAnt = input.shape[2]
    out = np.zeros((fftsize, numLines, numAnt), dtype=complex)

    for i_an in range(numAnt):
        inputMat = input[:, :, i_an]
        # pdb.set_trace()
        # visData.plotHeatmap(np.abs(inputMat))
        inputMat = np.subtract(inputMat, np.mean(inputMat, axis=0))
        inputMat = inputMat * rangeWindowCoeffVec[:, None]
        fftOutput = np.fft.fft(inputMat, fftsize, axis=0)
        # pdb.set_trace()
        out[:, :, i_an] = fftOutput
        
    return out


def DopplerFFT(input, opt):
    """
    Performs Doppler FFT on the input data using the specified options.

    :param input: Input data array (numpy array)
    :param opt: Options including doppler_fftsize
    :return: Array with applied Doppler FFT (numpy array)
    """
    fftsize = opt.doppler_fftsize
    dopplerWindowCoeffVec = np.hanning(input.shape[1] + 2)[1:-1]
    numLines = input.shape[0]
    numAnt = input.shape[2]
    out = np.zeros((numLines, fftsize, numAnt), dtype=complex)

    for i_an in range(numAnt):
        inputMat = np.squeeze(input[:, :, i_an])
        inputMat = inputMat * dopplerWindowCoeffVec[None, :]
        fftOutput = np.fft.fft(inputMat, fftsize, axis=1)
        fftOutput = np.fft.fftshift(fftOutput, axes=1)
        out[:, :, i_an] = fftOutput

    return out

def DopplerFFT_torch_onetime(input, opt):
    """
    This function performs Doppler FFT on the input data using the Hamming window.
    
    :param input: Input tensor with dimensions (batch_size, num_lines, num_samples, num_antennas, 3)
    :param opt: An object containing the Doppler FFT size attribute (doppler_fftsize)
    :return: Output tensor with dimensions (batch_size, num_lines, num_samples)
    """
    fftsize = opt.doppler_fftsize

    # Create Hamming window coefficients and move to GPU
    dopplerWindowCoeffVec = torch.hamming_window(input.shape[1] + 2, periodic=True, alpha=0.5, beta=0.5).cuda()
    dopplerWindowCoeffVec = dopplerWindowCoeffVec[1:-1]

    bs = input.shape[0]          # batch size
    numLines = input.shape[1]    # number of lines
    numAnt = input.shape[3]      # number of antennas

    # Apply the Hamming window coefficients to the input data
    inputMat_new = input * dopplerWindowCoeffVec[None, None, :, None, None]

    # Perform FFT on the input data and shift the zero-frequency component to the center of the spectrum
    fftOutput = torch.fft.fft(inputMat_new, fftsize, dim=2)
    fftOutput = torch.fft.fftshift(fftOutput, dim=2)

    return fftOutput

def CFAR(input,opt,debug=False):
  sig_integrate = np.sum((np.abs(input))**2,axis=2) + 1
  angleFFTSize = opt.angle_fftsize
  angleBinSkipLeft = 4
  angleBinSkipRight = 4
  if debug:
     pdb.set_trace()
  N_obj_Rag, Ind_obj_Rag, noise_obj, CFAR_SNR = CFAR_CASO_Range(sig_integrate, opt)


  # visData.plotHeatmap(np.abs(sig_integrate))
  # pdb.set_trace()
  detection_results = {}
  if (N_obj_Rag>0):
    
    N_obj, Ind_obj, noise_obj_an = CFAR_CASO_Doppler_overlap(sig_integrate,Ind_obj_Rag,input,opt)
    detection_results = {}

    # Use aggregate noise estimation from the first pass and apply
    # it to objects confirmed in the second pass
    noise_obj_agg = []
    for i_obj in range(1, N_obj+1):
      try:
        indx1R = Ind_obj[i_obj-1,0] # array([[10, 61],[21, 59]])
        indx1D = Ind_obj[i_obj-1,1]
      except:
        Ind_obj = Ind_obj.reshape([1,-1]) # Ind_obj: array([50, 73])
        indx1R = Ind_obj[i_obj-1,0]
        indx1D = Ind_obj[i_obj-1,1]
      
      ind2R = np.argwhere(Ind_obj_Rag[:,0] == indx1R) # index in python

      ind2D = np.argwhere(Ind_obj_Rag[ind2R,1] == indx1D) # index in python
      noiseInd = ind2R[ind2D[0][0],ind2D[0][1]].squeeze()
      if noiseInd.size !=0:
        noise_obj_agg.append(noise_obj[noiseInd])
      else:
        pdb.set_trace()

    for i_obj in range(1,N_obj+1):
        xind = (Ind_obj[i_obj-1,0]-1) +1
        detection_results[i_obj] = {'rangeInd':Ind_obj[i_obj-1, 0] - 1}

        detection_results[i_obj]['range'] = (detection_results[i_obj]['rangeInd'] + 1) * opt.rangeBinSize  # range estimation
        dopplerInd  = Ind_obj[i_obj-1, 1] - 1 # Doppler index
        detection_results[i_obj]['dopplerInd_org'] = dopplerInd
        detection_results[i_obj]['dopplerInd'] = dopplerInd
        
        # velocity estimation
        detection_results[i_obj]['doppler'] = (dopplerInd + 1 - opt.doppler_fftsize/2)*opt.velocityBinSize
        detection_results[i_obj]['doppler_corr'] = detection_results[i_obj]['doppler']

        detection_results[i_obj]['noise_var'] = noise_obj_agg[i_obj-1]       # noise variance

        
        detection_results[i_obj]['bin_val']  = np.reshape(input[xind, Ind_obj[i_obj-1,1],:],(opt.numAntenna,1),order='F')  # 2d FFT value for the 4 antennas

        # 
        # detection_results(i_obj).estSNR  = 10*log10(sum(abs(detection_results (i_obj).bin_val).^2)/sum(detection_results (i_obj).noise_var));  %2d FFT value for the 4 antennas
        detection_results[i_obj]['estSNR']  = (np.sum(np.abs(detection_results[i_obj]['bin_val'] ** 2))/np.sum(detection_results[i_obj]['noise_var'])) 
        
        sig_bin = np.array([[]]).transpose()
        # only apply max velocity extention if it is enabled and distance is larger than minDisApplyVmaxExtend
        if ((opt.applyVmaxExtend == 1) and (detection_results[i_obj]['range'] > opt.minDisApplyVmaxExtend) and (len(opt.overlapAntenna_ID))):
          raise Exception("NotCompleted Error")
        else:
          # Doppler phase correction due to TDM MIMO without apply Vmax extention algorithm
          deltaPhi = 2*(math.pi)*(dopplerInd + 1 - opt.doppler_fftsize/2)/( opt.numTxChan*opt.doppler_fftsize)
          sig_bin_org = detection_results[i_obj]['bin_val']
          for i_TX in range(1,opt.numTxChan+1):
            RX_ID = np.linspace((i_TX-1)*opt.numRxChan+1, i_TX*opt.numRxChan, i_TX*opt.numRxChan-(i_TX-1)*opt.numRxChan)

            sig_bin = np.concatenate((sig_bin,sig_bin_org[RX_ID.astype(int) -1]* np.exp(complex(0,(1-i_TX)*deltaPhi))))

          detection_results[i_obj]['bin_val'] = sig_bin
          detection_results[i_obj]['doppler_corr_overlap'] = detection_results[i_obj]['doppler_corr']
          detection_results[i_obj]['doppler_corr_FFT'] = detection_results[i_obj]['doppler_corr']

  return detection_results




def CFAR_CASO_Range(sig, opt):

  '''
  This function performs CFAR_CASO detection along range direction

  input
    obj: object instance of CFAR_CASO
    sig: a 2D real valued matrix, range x Doppler 

  return
    N_obj: number of objects detected
    Ind_obj: 2D bin index of the detected object
    noise_obj: noise value at the detected point after integration
  '''
  
  cellNum = opt.refWinSize
  gapNum = opt.guardWinSize
  cellNum = cellNum[0]
  gapNum = gapNum[0]
  K0 = opt.K0[0]

  M_samp = sig.shape[0]
  N_pul = sig.shape[1]

  # for each point under test, gapNum samples on the two sides are excluded from averaging. Left cellNum/2 and right cellNum/2 samples are used for averaging

  gaptot = gapNum + cellNum
  N_obj = 0
  Ind_obj = []
  noise_obj = []
  CFAR_SNR = []
  
  discardCellLeft = opt.discardCellLeft
  discardCellRight = opt.discardCellRight

  # for the first gaptot samples only use the right sample

  for k in range(N_pul):
    sigv=np.transpose(sig[:,k])
    vec = sigv[discardCellLeft:(M_samp-discardCellRight)]
    vecLeft = vec[0:(gaptot)]
    vecRight = vec[-(gaptot):]
    # 
    vec = np.concatenate((vecLeft, vec, vecRight))
    for j in range(1,1+M_samp-discardCellLeft-discardCellRight):
      
      cellInd= np.concatenate((np.linspace(j-gaptot,j-gapNum-1,j-gapNum-1-(j-gaptot)+1) ,np.linspace(j+gapNum+1,j+gaptot,j+gaptot-(j+gapNum+1)+1)))
      cellInd = cellInd + gaptot
      cellInda = np.linspace(j-gaptot,j-gapNum-1,j-gapNum-1-(j-gaptot)+1)
      cellInda = cellInda + gaptot
      cellIndb = np.linspace(j+gapNum+1,j+gaptot,j+gaptot-(j+gapNum+1)+1)
      cellIndb = cellIndb + gaptot
      
      cellave1a = 0
      for index_cellInda in cellInda:
        
        cellave1a = cellave1a + vec[int(index_cellInda)-1]
      cellave1a = cellave1a/cellNum

      cellave1b = 0
      for index_cellIndb in cellIndb:
        cellave1b = cellave1b + vec[int(index_cellIndb)-1]
      cellave1b = cellave1b/cellNum

      cellave1 = min(cellave1a,cellave1b)

      if opt.maxEnable==1:
        pass
      else:
        if vec[j+gaptot-1]>K0*cellave1:
          N_obj = N_obj+1
          
          Ind_obj.append([j+discardCellLeft-1, k])
          noise_obj.append(cellave1) #save the noise level
          CFAR_SNR.append(vec[j+gaptot-1]/cellave1)

  # get the noise variance for each antenna
  Ind_obj = np.array(Ind_obj)

  for i_obj in range(N_obj):
      # pdb.set_trace()
      ind_range = Ind_obj[i_obj,0]
      ind_Dop = Ind_obj[i_obj,1]
      if ind_range<= gaptot:
        # on the left boundary, use the right side samples twice
        cellInd = np.concatenate((np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum) , np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum)))
      elif ind_range>=M_samp-gaptot+1:
        # on the right boundary, use the left side samples twice
        cellInd = np.concatenate((np.linspace(ind_range-gaptot, ind_range-gapNum-1,gaptot-gapNum) , np.linspace(ind_range-gaptot, ind_range-gapNum-1,gaptot-gapNum)))
      else:
        cellInd = np.concatenate((np.linspace(ind_range-gaptot, ind_range-gapNum-1,gaptot-gapNum) ,  np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum)))
  
  return N_obj, Ind_obj, noise_obj, CFAR_SNR

def CFAR_CASO_Doppler_overlap(sig_integ,Ind_obj_Rag,sigCpml,opt):

  '''
  % This function performs 1D CFAR_CASO detection along the Doppler direction, and declare detection only if the index overlap with range detection results. 

  %input
  %   obj: object instance of CFAR_CASO
  %   Ind_obj_Rag: index of range bins that has been determined by the first
  %   step detection along the range direction
  %   sigCpml: a 3D complex matrix, range x Doppler x antenna array
  %   sig_integ: a 2D real valued matrix, range x Doppler 

  %output
  %   N_obj: number of objects detected
  %   Ind_obj: 2D bin index of the detected object
  %   noise_obj_an: antenna specific noise estimation before integration
  '''
  maxEnable = opt.maxEnable
  cellNum0 = opt.refWinSize
  gapNum0 = opt.guardWinSize
  cellNum = cellNum0[1]
  gapNum = gapNum0[1]
  K0 = opt.K0[1]

  rangeNumBins = sig_integ.shape[0]

  # extract the detected points after range detection
  detected_Rag_Cell = np.unique(Ind_obj_Rag[:,0])
  
  sig = sig_integ[detected_Rag_Cell-1,:]

  M_samp = sig.shape[0]
  N_pul= sig.shape[1]


  # for each point under test, gapNum samples on the two sides are excluded
  # from averaging. Left cellNum/2 and right cellNum/2 samples are used for averaging
  gaptot = gapNum + cellNum

  N_obj = 0
  Ind_obj = np.array([])
  noise_obj_an = []
  vec = np.zeros([N_pul+gaptot*2]) 

  for k in range(1,M_samp+1):
      # get the range index at current range index
      detected_Rag_Cell_i = detected_Rag_Cell[k-1]
      ind1 = np.argwhere(Ind_obj_Rag[:,0] == detected_Rag_Cell_i)
      indR = Ind_obj_Rag[ind1, 1]
      # extend the left the vector by copying the left most the right most
      # gaptot samples are not detected.
      sigv = (sig[k-1,:])
      # pdb.set_trace()
      vec[0:gaptot] = sigv[-gaptot:]
      vec[gaptot: N_pul+gaptot] = sigv
      vec[N_pul+gaptot:] = sigv[:gaptot]
      # start to process
      ind_loc_all = []
      ind_loc_Dop = []
      ind_obj_0 = 0
      noiseEst = np.zeros([N_pul])
      for j in range(1+gaptot,N_pul+gaptot+1):
        cellInd= np.concatenate((np.linspace(j-gaptot, j-gapNum-1,gaptot-gapNum) ,np.linspace(j+gapNum+1,j+gaptot,gaptot-gapNum)))
        noiseEst[j-gaptot-1] = np.sum(vec[ (cellInd-1).astype(int)])

      for j in range(1+gaptot,N_pul+gaptot+1):
          j0 = j - gaptot
          cellInd = np.concatenate((np.linspace(j-gaptot, j-gapNum-1,gaptot-gapNum) ,np.linspace(j+gapNum+1,j+gaptot,gaptot-gapNum)))
          cellInda = np.linspace(j-gaptot,j-gapNum-1,j-gapNum-1-(j-gaptot)+1)
          cellIndb = np.linspace(j+gapNum+1,j+gaptot,j+gaptot-(j+gapNum+1)+1)
          
          cellave1a = 0
          for index_cellInda in cellInda:
            
            cellave1a = cellave1a + vec[int(index_cellInda)-1]
          cellave1a = cellave1a/cellNum

          cellave1b = 0
          for index_cellIndb in cellIndb:
            cellave1b = cellave1b + vec[int(index_cellIndb)-1]
          cellave1b = cellave1b/cellNum

          cellave1 = min(cellave1a,cellave1b)     
          
          maxInCell = np.max(vec[(cellInd-1).astype(int)])

          if maxEnable==1:
              # detect only if it is the maximum within window
              condition = ((vec[j-1]>K0*cellave1)) and ((vec[j-1]>maxInCell))
          else:
              condition = vec[j-1]>K0*cellave1
          
          if condition==1:
              # check if this detection overlap with the Doppler detection
              # indR+1 --> differeny index between matlab and python
              # pdb.set_trace()
              if np.isin((indR+1).squeeze(), j0).any():
                  # find overlap, declare a detection
                  ind_win = detected_Rag_Cell_i
                  # range index
                  ind_loc_all = ind_loc_all + [ind_win]
                  # Doppler index
                  ind_loc_Dop = ind_loc_Dop + [j0-1]

      ind_loc_all = np.array(ind_loc_all)
      ind_loc_Dop = np.array(ind_loc_Dop)
      

      if len(ind_loc_all)>0:
        ind_obj_0 = np.stack((ind_loc_all,ind_loc_Dop),axis=1)
        if Ind_obj.shape[0] == 0:
          
          Ind_obj = ind_obj_0
          
        else:    
          # following process is to avoid replicated detection points
          ind_obj_0_sum = ind_loc_all + 10000*ind_loc_Dop
          Ind_obj_sum = Ind_obj[:,0] + 10000*Ind_obj[:,1]
          for ii  in range(1,ind_loc_all.shape[0]+1):
              if not np.isin(Ind_obj_sum, ind_obj_0_sum[ii-1]).any():
                  Ind_obj = np.concatenate((Ind_obj, ind_obj_0[ii-1,np.newaxis,:]),axis = 0)
  
  N_obj = Ind_obj.shape[0]

  # reset the ref window size to range direction
  cellNum = cellNum0[0]
  gapNum = gapNum0[0]
  gaptot = gapNum + cellNum
  # get the noise variance for each antenna
  N_obj_valid = 0
  Ind_obj_valid = []
  
  for i_obj in range(1,N_obj+1):    
    ind_range = Ind_obj[i_obj-1,0]
    ind_Dop = Ind_obj[i_obj-1,1]
    # skip detected points with signal power less than obj.powerThre

    
    if (min(np.abs(sigCpml[ind_range, ind_Dop,:]) ** 2) < opt.powerThre):
        continue

    if (ind_range+1) <= gaptot:
        # on the left boundary, use the right side samples twice
        cellInd = np.concatenate((np.linspace(ind_range+gapNum+1,ind_range+gaptot, gaptot-gapNum), np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum)))
    elif (ind_range+1) >= rangeNumBins-gaptot+1:
        # on the right boundary, use the left side samples twice
        cellInd = np.concatenate((np.linspace(ind_range-gaptot,ind_range-gapNum-1, gaptot-gapNum), np.linspace(ind_range-gaptot,ind_range-gapNum-1,gaptot-gapNum)))
    else:
        cellInd = np.concatenate((np.linspace(ind_range-gaptot,ind_range-gapNum-1, gaptot-gapNum), np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum)))
    
    
    N_obj_valid = N_obj_valid +1
    # pdb.set_trace()
    noise_obj_an.append( np.reshape((np.mean(abs(sigCpml[cellInd.astype(int), ind_Dop,np.newaxis, :].copy()) ** 2,axis=0)), (opt.numAntenna, 1, 1), order="F"))
    
    Ind_obj_valid.append( Ind_obj[i_obj-1,:])    
      
  N_obj = N_obj_valid
  Ind_obj = np.array(Ind_obj_valid).squeeze()
  noise_obj_an = np.array(noise_obj_an).squeeze()
  
  return N_obj, Ind_obj, noise_obj_an

def DOA(detected_obj,opt, output_est_results = True):
  numObj = len(detected_obj)
  out = copy.deepcopy(detected_obj)
  numAoAObjCnt = 0
  angle_sepc_2D_fft_dict = {}
  DOA_index = []
  # extended detection_obj to include the angles information
  for i_obj in range(1,numObj+1):
    current_obj = detected_obj[i_obj]
    estSNR = 10*math.log10(np.sum(np.abs(current_obj['bin_val'] ** 2)/np.sum(current_obj['noise_var'])) + 1e-10)
    X = current_obj['bin_val']

    R = np.dot(X,X.T.conjugate())

    if opt.method == 1:
      # 2D beamforming angle estimation, azimuth is estimated based on 1D FFT output   
      # if i_obj == 9:
      #   pdb.set_trace()
      # pdb.set_trace()       
      if not output_est_results: # if not output_est_results, then only return the DOA angles
         angle_sepc_2D_fft = DOA_beamformingFFT_2D(X, opt, output_est_results=False)
         current_obj_range = current_obj["range"]
         angle_sepc_2D_fft_dict[current_obj_range] = angle_sepc_2D_fft
         if i_obj == numObj:
            return angle_sepc_2D_fft_dict
         else:
            continue
        #  pdb.set_trace()
         
      
      DOA_angles, angle_sepc_2D_fft = DOA_beamformingFFT_2D(X, opt)
      if (numAoAObjCnt == 0):
        out = {}

      
      for i_obj in range(1,DOA_angles.shape[0]+1):
          
          if DOA_angles.size == 0:

            break
          numAoAObjCnt = numAoAObjCnt+1
          
          out[numAoAObjCnt] = {'rangeInd': current_obj['rangeInd']}
          out[numAoAObjCnt]['dopplerInd'] = current_obj['dopplerInd']
          out[numAoAObjCnt]['range'] = current_obj['range']
          out[numAoAObjCnt]['doppler_corr'] = current_obj['doppler_corr']
          out[numAoAObjCnt]['dopplerInd_org'] = current_obj['dopplerInd_org']

          out[numAoAObjCnt]['noise_var'] = current_obj['noise_var']
          out[numAoAObjCnt]['bin_val'] = current_obj['bin_val']
          out[numAoAObjCnt]['estSNR'] = current_obj['estSNR']
          out[numAoAObjCnt]['doppler_corr_overlap'] = current_obj['doppler_corr_overlap']
          out[numAoAObjCnt]['doppler_corr_FFT'] = current_obj['doppler_corr_FFT']
          out[numAoAObjCnt]["angle_azi"] = DOA_angles[i_obj-1,0] # 
          out[numAoAObjCnt]["angle_ele"] = DOA_angles[i_obj-1,1] # 
          angle_SNR = 10 * np.log10(np.abs(angle_sepc_2D_fft[int(DOA_angles[i_obj-1,2])-1,int(DOA_angles[i_obj-1,3])-1])**2 / np.sum(np.abs(angle_sepc_2D_fft))**2 + 1e-10)
          min_snr_db = -200
          max_snr_db = 0  
          normalized_snr = (angle_SNR - min_snr_db) / (max_snr_db - min_snr_db)
          normalized_snr = np.clip(normalized_snr, 0, 1)
          out[numAoAObjCnt]["SNR_angle"] =  normalized_snr # 2D FFT value for the 4 antennas
          out[numAoAObjCnt]['angles'] = DOA_angles[i_obj-1,:]
          out[numAoAObjCnt]['spectrum'] = angle_sepc_2D_fft

    elif opt.method == 2:
      raise Exception("NotCompleted Error")
    else:
      raise Exception("Wrong Parameter method Error")

  return out



def DOA_elevation_matrix(sig, opt,index=1):
  '''
  % DOA_beamformingFFT_2D function to get elevation values
  %is done in 1D FFT domain, the elevation peak selection is done after 2D FFT

  % input:
  %   obj: object instance
  %   sig: complex signal vector, with each value corresponding to each
  %   antenna. The length of this vector equals to numTX x numRX enabled.
  %   There can be overlapped antennas. this signal needs to be re-arranged
  %   based on D value to form the virtual antenna array

  % output:
  %   angle_sepc_2D_fft: angle 2D fft spectrum
  '''

  # field of view to do beamforming
  angles_DOA_az = opt.angles_DOA_az
  angles_DOA_ele = opt.angles_DOA_ele

  # distance unit in terms of wavelength
  d = opt.antDis
  # 2D matrix providing antenna coordinates
  D = opt.D
  angleFFTSize = opt.angle_fftsize


  # FFT based implementation
  # first form a 2D matrix based on the antenna coordinates
  D = np.array(D) + 1
  apertureLen_azim = max(D[:,0])
  apertureLen_elev = max(D[:,1])
  sig_2D = np.zeros((apertureLen_azim+1, apertureLen_elev+1),dtype = "complex_")
  for i_line in range(1,apertureLen_elev+1):
    ind = np.argwhere(D[:,1] == i_line)
    D_sel = D[ind,0]
    sig_sel = sig[ind]
    value_index, indU = np.unique(D_sel, return_index=True) # -1 to fit matlab's index
    
    sig_2D[D_sel[indU].squeeze() - 1,i_line-1] = sig_sel[indU].squeeze()
  
  # run FFT on azimuth and elevation
  angle_sepc_1D_fft = np.fft.fftshift(np.fft.fft(sig_2D,angleFFTSize,axis=0),axes = 0)
  angle_sepc_2D_fft = np.fft.fftshift(np.fft.fft(angle_sepc_1D_fft,angleFFTSize,axis = 1),axes = 1); 
  
  return angle_sepc_2D_fft


def DOA_beamformingFFT_2D(sig, opt, index="a", output_est_results = True):
  '''
  % DOA_beamformingFFT_2D function perform 2D angle estimation based on FFT beamforming, the azimuth peak selection
  %is done in 1D FFT domain, the elevation peak selection is done after 2D FFT

  % input:
  %   obj: object instance
  %   sig: complex signal vector, with each value corresponding to each
  %   antenna. The length of this vector equals to numTX x numRX enabled.
  %   There can be overlapped antennas. this signal needs to be re-arranged
  %   based on D value to form the virtual antenna array

  % output:
  %   angleObj_est: angle estimation results
  %   angle_sepc_2D_fft: angle 2D fft spectrum
  '''

  # field of view to do beamforming
  angles_DOA_az = opt.angles_DOA_az
  angles_DOA_ele = opt.angles_DOA_ele

  # distance unit in terms of wavelength
  d = opt.antDis
  # 2D matrix providing antenna coordinates
  D = opt.D
  angleFFTSize = opt.angle_fftsize


  # FFT based implementation
  # first form a 2D matrix based on the antenna coordinates
  D = np.array(D) + 1
  apertureLen_azim = max(D[:,0])
  apertureLen_elev = max(D[:,1])
  sig_2D = np.zeros((apertureLen_azim+1, apertureLen_elev+1),dtype = "complex_")
  for i_line in range(1,apertureLen_elev+1):
    ind = np.argwhere(D[:,1] == i_line)
    D_sel = D[ind,0]
    sig_sel = sig[ind]
    value_index, indU = np.unique(D_sel, return_index=True) # -1 to fit matlab's index
    
    sig_2D[D_sel[indU].squeeze() - 1,i_line-1] = sig_sel[indU].squeeze()
  
  # run FFT on azimuth and elevation
  angle_sepc_1D_fft = np.fft.fftshift(np.fft.fft(sig_2D,angleFFTSize,axis=0),axes = 0)
  angle_sepc_2D_fft = np.fft.fftshift(np.fft.fft(angle_sepc_1D_fft,angleFFTSize,axis = 1),axes = 1); 
  if not output_est_results:
    return angle_sepc_2D_fft
  
  pi = math.pi

  wx_vec = np.linspace(-pi, pi, angleFFTSize+1)
  wz_vec = np.linspace(-pi, pi, angleFFTSize+1)
  wx_vec = wx_vec[:-1]
  wz_vec = wz_vec[:-1]
  # use one row with complete azimuth antenna of 1D FFT output for azimuth estimation
  spec_azim = np.abs(angle_sepc_1D_fft[:,0])
  opt.sidelobeLevel_dB = opt.sidelobeLevel_dB_azim

  # pdb.set_trace()
  # spec_azim: (128, )
  if not (index == "a"):
    
    norm = lambda x: (x - x.min())/(x.max() - x.min())
    spec_azim_norm = norm(spec_azim)
    spec_azim_plot = []
    fft_axis = []
    # 
    for index1, value in enumerate(spec_azim_norm):
      theta = math.asin(index1/(angleFFTSize/2) - 1)* 180 / math.pi
      fft_axis.append(theta)
      spec_azim_plot.append(value)
    # pdb.set_trace()
    
    # plt.plot(music_axis,spec_azim_plot, label='FFT')
    plt.plot(fft_axis,spec_azim_plot, label='FFT')
    plt.legend()
    plt.show()
    plt.savefig("AngleFFT_MUSIC_" + str(index) + ".jpg")
    print("AngleFFT_MUSIC_" + str(index) + ".jpg")
    plt.close()

  peakVal_azim, peakLoc_azim = DOA_BF_PeakDet_loc(spec_azim, opt)

  if apertureLen_elev == 1:
    # azimuth array only, no elevation antennas
    obj_cnt = 1
    angleObj_est= []
    for i_obj in range(1, 1+ peakLoc_azim.size):
      ind = peakLoc_azim[i_obj - 1]
      
      azim_est = (math.asin(wx_vec[(ind-1).astype(int)]/(2*pi*d)))/(2*math.pi)*360
      if ((azim_est >= angles_DOA_az[0]) and (azim_est <= angles_DOA_az[1])):
        angleObj_est.append([azim_est,0,ind,0])
        obj_cnt = obj_cnt+1
          
      else:
        continue

    if len(angleObj_est) == 0:
      angleObj_est = np.array([[]])
    else:
      angleObj_est = np.array(angleObj_est)
      
  else:
    # azimuth and elevation angle estimation
  
    #  figure(1);plot(spec_azim); hold on; grid on
    #  plot(peakLoc_azim, spec_azim(peakLoc_azim),'ro');hold on
    
    # for each detected azimuth, estimate its elevation
    #  figure(2)
    obj_cnt = 1
    angleObj_est = []
    opt.sidelobeLevel_dB = opt.sidelobeLevel_dB_elev
    for i_obj in range(1, 1+ peakLoc_azim.size):
      ind = peakLoc_azim[i_obj - 1]
      spec_elev = np.abs(angle_sepc_2D_fft[(ind-1).astype(int),:])
      peakVal_elev, peakLoc_elev = DOA_BF_PeakDet_loc(spec_elev, opt)
      # calcualte the angle values
      for j_elev in range(1, 1+ peakVal_elev.size):

        azim_est = (math.asin(wx_vec[(ind-1).astype(int)]/(2*pi*d)))/(2*math.pi)*360
        # pdb.set_trace()
        # print(azim_est)
        elev_est = (math.asin(wz_vec[peakLoc_elev[j_elev-1].astype(int)-1]/(2*pi*d)))/(2*math.pi)*360
        if ((azim_est >= angles_DOA_az[0]) and (azim_est <= angles_DOA_az[1]) and (elev_est >= angles_DOA_ele[0]) and (elev_est <= angles_DOA_ele[1])):
          angleObj_est.append([azim_est,elev_est,ind[0],peakLoc_elev[j_elev-1][0]])

          obj_cnt = obj_cnt+1
            
        else:
          continue

    if len(angleObj_est) == 0:
      angleObj_est = np.array([[]])
    else:
      angleObj_est = np.array(angleObj_est)
  # pdb.set_trace()
  
  return angleObj_est, angle_sepc_2D_fft



def DOA_BF_PeakDet_loc(inData, opt):
  gamma = opt.gamma
  sidelobeLevel_dB = opt.sidelobeLevel_dB
  inData = inData.squeeze()

  minVal = float("inf")
  maxVal = 0
  maxLoc = 0
  maxData = np.array([[]])

  locateMax = 0  #  at beginning, not ready for peak detection
  km = 1         # constant value used in variance calculation

  numMax = 0
  extendLoc = 0
  initStage = 1
  absMaxValue = 0

  i = 0
  N = inData.size
  while (i < (N + extendLoc - 1)):
    i = i+1
    i_loc = ((i-1) % N) +1
    currentVal = inData[i_loc-1]
    # pdb.set_trace()
    # record the maximum value
    # if i_loc == 256:
    #   pdb.set_trace()
    # print(currentVal, '+', i_loc)
    # pdb.set_trace()
    try:
      if currentVal > absMaxValue:
        
        absMaxValue = currentVal
    except:
      pdb.set_trace()

    # record the current max value and location
    if currentVal > maxVal:
        maxVal = currentVal
        maxLoc = i_loc
        maxLoc_r = i
    
    # record for the current min value and location
    if currentVal < minVal:
        minVal = currentVal

    
    if locateMax:
      if currentVal < (maxVal/gamma):
        numMax = numMax + 1
        bwidth = i - maxLoc_r
        # Assign maximum value only if the value has fallen below the max by
        # gamma, thereby declaring that the max value was a peak

        if maxData.size == 0:
          maxData = np.concatenate((maxData,np.array([[maxLoc, maxVal, bwidth, maxLoc_r]])), axis=1)
        else:
          # pdb.set_trace()
          maxData = np.concatenate((maxData[:numMax-1,:],np.array([[maxLoc, maxVal, bwidth, maxLoc_r]])), axis=0)
        
        minVal = currentVal
        locateMax = 0

    else:
      if currentVal > minVal*gamma:
        # Assign minimum value if the value has risen above the min by
        # gamma, thereby declaring that the min value was a valley
        locateMax = 1
        maxVal = currentVal
        if (initStage == 1):
          extendLoc = i
          initStage = 0
  
  # make sure the max value needs to be cetain dB higher than the side lobes to declare any detection
  estVar = np.zeros((numMax, 1))
  peakVal = np.zeros((numMax, 1))
  peakLoc = np.zeros((numMax, 1))
  delta = []

  # [v ind] = max(maxData(:,2));
  # peakMean = mean(maxData([1:(ind-1) ind+1:end],2));
  # SNR_DOA = v/peakMean;
  # if v>peakMean*maxPeakThre
  #  if the max is different by more than sidelobeLevel_dB dB from the peak, then removed it as a sidelobe
  maxData_ = np.zeros((numMax, maxData.shape[1]))
  numMax_ = 0
  totPower = 0
  for i in range(1,numMax+1):
      if maxData[i-1, 1] >= (absMaxValue * (pow(10, -sidelobeLevel_dB/10))):
        numMax_ = numMax_ + 1
        maxData_[numMax_ - 1, :] = maxData[i-1, :]
        totPower = totPower + maxData[i-1, 1]

  maxData = maxData_
  numMax = numMax_

  estVar = np.zeros((numMax, 1))
  peakVal = np.zeros((numMax, 1))
  peakLoc = np.zeros((numMax, 1))

  delta = []
  for ind in range(1,numMax+1): 
    peakVal[ind-1] = maxData[ind-1,1]
    peakLoc[ind-1] = ((maxData[ind-1,0]-1) %  N) + 1
  return peakVal, peakLoc


def pcdOutput(sig_integrate,detection_results,angleEst, DopplerFFTOut, opt):

  angles_all_points = np.zeros([len(angleEst),6])
  xyz = np.zeros([len(angleEst),11])
  
  if len(angleEst) > 0:
    for iobj in range(1,len(angleEst)+1):
      angles_all_points[iobj-1][0:2] = angleEst[iobj]['angles'][0:2] 
      angles_all_points[iobj-1][2] = angleEst[iobj]['estSNR']
      angles_all_points[iobj-1][3] = angleEst[iobj]['rangeInd']
      angles_all_points[iobj-1][4] = angleEst[iobj]['doppler_corr']
      angles_all_points[iobj-1][5] = angleEst[iobj]['range']

      xyz[iobj-1][0] = angles_all_points[iobj-1][5]*math.sin((angles_all_points[iobj-1][0])*(1)*(2*math.pi)/360)*math.cos(angles_all_points[iobj-1][1]*(2*math.pi)/360)

      xyz[iobj-1][1] = angles_all_points[iobj-1][5]*math.cos((angles_all_points[iobj-1][0])*(1)*(2*math.pi)/360)*math.cos(angles_all_points[iobj-1][1]*(2*math.pi)/360)

      
      # switch upside and down, the elevation angle is flipped (NO?)
      # xyz[iobj-1][2] = angles_all_points[iobj-1][5]*math.sin((angles_all_points[iobj-1][1])*(-1)*(2*math.pi)/360)
      xyz[iobj-1][2] = angles_all_points[iobj-1][5]*math.sin((angles_all_points[iobj-1][1])*(1)*(2*math.pi)/360)
      
      xyz[iobj-1][3] = angleEst[iobj]['doppler_corr']
      xyz[iobj-1][8] = angleEst[iobj]['SNR_angle']
      xyz[iobj-1][4] = angleEst[iobj]['range']
      xyz[iobj-1][5] = angleEst[iobj]['estSNR']
      xyz[iobj-1][6] = angleEst[iobj]['angle_azi']
      xyz[iobj-1][7] = angleEst[iobj]['angle_ele']
      xyz[iobj-1][9] = angleEst[iobj]['rangeInd']
      xyz[iobj-1][10] = angleEst[iobj]['dopplerInd']

  return xyz

def adc2pcd(radar_adc_data,radar_config):
    # pdb.set_trace()
    if (radar_config.numTxChan) and (radar_config.numChirpsPerFrame) == 1:
      radar_adc_data = np.squeeze(radar_adc_data)
      rangeFFTOut = np.fft.fft(radar_adc_data, radar_config.range_fftsize, axis=0)
      rangeFFTOut[0:int(radar_config.range_fftsize*0.05),:] = 0
      rangeFFTOut[-int(radar_config.range_fftsize*0.1):,:] = 0
      aoa_tof_out = np.fft.fft(rangeFFTOut, radar_config.angle_fftsize, axis=1)

      fp_range = findpeaks.findpeaks(method='topology', whitelist=['peak'], lookahead=200, interpolate=None, limit=0.1, imsize=None, scale=True, togray=True, denoise='fastnl', window=3, cu=0.25, params_caerus={'minperc': 3, 'nlargest': 10, 'threshold': 0.25, 'window': 50}, figsize=(15, 8), verbose=3)
      fp_azi = findpeaks.findpeaks(method='topology', whitelist=['peak'], lookahead=10, interpolate=None, limit=0.05, imsize=None, scale=True, togray=True, denoise='fastnl', window=10, cu=0.25, figsize=(15, 8), verbose=3)

      sig_range = np.abs(aoa_tof_out.sum(1)) / np.abs(aoa_tof_out.sum(1)).max()
      # pdb.set_trace()
      results_range = fp_range.fit(sig_range)
      # visData.plotVector(FinalElevFFTOut.sum(1).sum(1).sum(1))
      range_pos_len =  len(results_range["persistence"].y)

      xyz = []
      for range_index_i in range(range_pos_len):
        range_index = results_range["persistence"].y[range_index_i]
        Azi_Data = aoa_tof_out[range_index,:] # (128, 128, 128)
        sig_azi = np.abs(Azi_Data)/np.max(np.abs(Azi_Data))
        results_azi = fp_azi.fit(sig_azi)

        azi_pos_len =  len(results_azi["persistence"].y)
        for azi_index_i in range(azi_pos_len):
          azi_index = results_azi["persistence"].y[azi_index_i]
          if azi_index < 10 or azi_index > 110:
            continue
          range_i = range_index * radar_config.rangeBinSize
          
          sin_theta_i = azi_index/(radar_config.angle_fftsize/2) - 1
          cos_theta_i = math.sqrt(1 - (sin_theta_i * sin_theta_i))
          
          theta_i = math.asin(sin_theta_i)* 180 / math.pi

          ele_index = radar_config.angle_fftsize//2
          sin_phi_i = ele_index/(radar_config.angle_fftsize/2) - 1
          cos_phi_i = math.sqrt(1 - (sin_phi_i * sin_phi_i))
          
          phi_i = math.asin(sin_phi_i)* 180 / math.pi

          x_cor = range_i * cos_phi_i * sin_theta_i

          y_cor = range_i * cos_phi_i * cos_theta_i

          z_cor = range_i * sin_phi_i
          
          intensity = np.abs(aoa_tof_out[range_index,azi_index])

          xyz.append([x_cor, y_cor, z_cor,intensity])

      xyz = np.array(xyz)
      if xyz.size == 0:
          print("warning: xyz is empty!")
          xyz = np.zeros([1,9])
          
      return None, None, xyz, xyz

    else:
      rangeFFTOut = np.zeros((radar_config.range_fftsize,radar_config.numChirpsPerFrame,radar_config.numRxChan,radar_config.numTxChan),dtype=complex)
      DopplerFFTOut = np.zeros((radar_config.range_fftsize,radar_config.doppler_fftsize,radar_config.numRxChan,radar_config.numTxChan),dtype=complex)
      for i_tx in range(radar_adc_data.shape[3]):
          rangeFFTOut[:,:,:,i_tx] = rangeFFT(radar_adc_data[:,:,:,i_tx], radar_config)
          rangeFFTOut[0:int(radar_config.range_fftsize*0.05),:,:,:] = 0
          rangeFFTOut[-int(radar_config.range_fftsize*0.1):,:,:,:] = 0
          DopplerFFTOut[:,:,:,i_tx] = DopplerFFT(rangeFFTOut[:,:,:,i_tx], radar_config)
      
      DopplerFFTOut = np.reshape(DopplerFFTOut, (DopplerFFTOut.shape[0], DopplerFFTOut.shape[1],-1), order="F")
      sig_integrate = np.dot(10,np.log10(np.sum((np.abs(DopplerFFTOut))**2,axis=2) + 1))
      detection_results = CFAR(DopplerFFTOut,radar_config,debug=False)

      if len(detection_results) > 0:
          angleEst = DOA(detection_results, radar_config)
          angleEst_nomultipath = filter_angleEst_nomultipath(angleEst, radar_config)
          xyz_ticode = pcdOutput(sig_integrate,detection_results,angleEst,DopplerFFTOut,radar_config)
          xyz_ticode_nomultipath = pcdOutput(sig_integrate,detection_results,angleEst_nomultipath,DopplerFFTOut,radar_config)
          xyz_ticode = xyz_ticode[:,:9]
          xyz_ticode_nomultipath = xyz_ticode_nomultipath[:,:9]

      else:
          print("Warning: No object detected!")
          xyz_ticode = np.zeros([1,9])
          xyz_ticode_nomultipath = np.zeros([1,9])
      
      if xyz_ticode.size == 0:
          xyz_ticode = np.zeros([1,9])
      if xyz_ticode_nomultipath.size == 0:
          xyz_ticode_nomultipath = np.zeros([1,9])
      return rangeFFTOut, DopplerFFTOut, xyz_ticode, xyz_ticode_nomultipath


def adc2pcd_RPD_generation(radar_adc_data,radar_config):
    rangeFFTOut = np.zeros((radar_config.range_fftsize,radar_config.numChirpsPerFrame,radar_config.numRxChan,radar_config.numTxChan),dtype=complex)
    DopplerFFTOut = np.zeros((radar_config.range_fftsize,radar_config.doppler_fftsize,radar_config.numRxChan,radar_config.numTxChan),dtype=complex)
    # pdb.set_trace()
    for i_tx in range(radar_adc_data.shape[3]):
        rangeFFTOut[:,:,:,i_tx] = rangeFFT(radar_adc_data[:,:,:,i_tx], radar_config)
        # rangeFFTOut[0:20,:,:,:] = 0
        # rangeFFTOut[-30:,:,:,:] = 0
        rangeFFTOut[0:int(radar_config.range_fftsize*0.05),:,:,:] = 0
        rangeFFTOut[-int(radar_config.range_fftsize*0.1):,:,:,:] = 0
        DopplerFFTOut[:,:,:,i_tx] = DopplerFFT(rangeFFTOut[:,:,:,i_tx], radar_config)
    
    # visData.plotVector(rangeFFTOut[:,1,1,1])
    DopplerFFTOut = np.reshape(DopplerFFTOut, (DopplerFFTOut.shape[0], DopplerFFTOut.shape[1],-1), order="F")
    sig_integrate = np.dot(10,np.log10(np.sum((np.abs(DopplerFFTOut))**2,axis=2) + 1))
    detection_results = CFAR(DopplerFFTOut,radar_config,debug=False)

    if len(detection_results) > 0:
        angleEst = DOA(detection_results, radar_config)
        angleEst_nomultipath = filter_angleEst_nomultipath(angleEst, radar_config)
        xyz_ticode = pcdOutput(sig_integrate,detection_results,angleEst,DopplerFFTOut,radar_config)
        # xyz_ticode = xyz_ticode[:,:9]
        # pdb.set_trace()
        # visData.plotPcd(xyz_ticode_nomultipath)

    else:
        print("Warning: No object detected!")
        xyz_ticode = np.zeros([1,11])
    
    if xyz_ticode.size == 0:
        xyz_ticode = np.zeros([1,11])
    # print("Finished pcdOutput!")
    return rangeFFTOut, DopplerFFTOut, xyz_ticode



def filter_angleEst_nomultipath(angleEst, radar_config):
    angleEst_nomultipath = {}
    index = 1

    for key1, point1 in angleEst.items():
        keep_point = True
        for key2, point2 in angleEst.items():
            if key1 != key2:
                az_diff = abs(point1['angles'][2] - point2['angles'][2])
                el_diff = abs(point1['angles'][3] - point2['angles'][3])
                if az_diff <= 4 and el_diff <= 4:
                    if point1['rangeInd'] > point2['rangeInd']:
                        keep_point = False
                        break

        if keep_point:
            angleEst_nomultipath[index] = point1
            index += 1

    return angleEst_nomultipath


def adc2pcd_peakdetection(radar_adc_data,radar_config):
    radar_adc_data_new = np.zeros([radar_adc_data.shape[0], radar_adc_data.shape[1], 8, 2], dtype=np.complex64)
    radar_adc_data_new[:,:,:4,0] = radar_adc_data[:,:,:,0]
    radar_adc_data_new[:,:,4:,0] = radar_adc_data[:,:,:,2]
    radar_adc_data_new[:,:,2:6,1] = radar_adc_data[:,:,:,1]
    rangeWindowCoeffVec = np.hanning(radar_adc_data_new.shape[0] + 2)[1:-1]
    radar_adc_data_new = np.subtract(radar_adc_data_new, np.mean(radar_adc_data_new, axis=0))
    radar_adc_data_new = radar_adc_data_new * rangeWindowCoeffVec[:, None, None, None]
    rangeFFTOut = np.zeros((radar_config.range_fftsize,radar_config.numChirpsPerFrame,8,2),dtype=complex)
    rangeFFTOut = np.fft.fft(radar_adc_data_new, radar_config.range_fftsize, axis=0)
    rangeFFTOut[0:int(radar_config.range_fftsize*0.05),:,:,:] = 0
    rangeFFTOut[-int(radar_config.range_fftsize*0.1):,:,:,:] = 0

    dopplerWindowCoeffVec = np.hanning(radar_adc_data_new.shape[1] + 2)[1:-1]
    DopplerFFTOut = np.zeros((radar_config.range_fftsize,radar_config.doppler_fftsize,8,2),dtype=complex)
    DopplerFFTOut = rangeFFTOut * dopplerWindowCoeffVec[None, :, None, None]
    DopplerFFTOut = np.fft.fftshift(np.fft.fft(DopplerFFTOut, radar_config.doppler_fftsize, axis=1), axes=1)

    AziFFTOut = np.zeros((radar_config.range_fftsize,radar_config.doppler_fftsize,radar_config.angle_fftsize,2),dtype=complex)
    AziFFTOut = np.fft.fftshift(np.fft.fft(DopplerFFTOut, radar_config.angle_fftsize, axis=2), axes=2)

    FinalElevFFTOut = np.zeros((radar_config.range_fftsize,radar_config.doppler_fftsize,radar_config.angle_fftsize,radar_config.angle_fftsize),dtype=complex)
    FinalElevFFTOut = np.fft.fftshift(np.fft.fft(AziFFTOut, radar_config.angle_fftsize, axis=3), axes=3)


    fp_range = findpeaks.findpeaks(method='topology', whitelist=['peak'], lookahead=200, interpolate=None, limit=0.1, imsize=None, scale=True, togray=True, denoise='fastnl', window=3, cu=0.25, params_caerus={'minperc': 3, 'nlargest': 10, 'threshold': 0.25, 'window': 50}, figsize=(15, 8), verbose=3)

    fp_doppler = findpeaks.findpeaks(method='topology', whitelist=['peak'], lookahead=10, interpolate=None, limit=0.1, imsize=None, scale=True, togray=True, denoise='fastnl', window=10, cu=0.25, figsize=(15, 8), verbose=3)

    fp_azi = findpeaks.findpeaks(method='topology', whitelist=['peak'], lookahead=10, interpolate=None, limit=0.05, imsize=None, scale=True, togray=True, denoise='fastnl', window=10, cu=0.25, figsize=(15, 8), verbose=3)

    fp_ele = findpeaks.findpeaks(method='topology', whitelist=['peak'], lookahead=10, interpolate=None, limit=0.05, imsize=None, scale=True, togray=True, denoise='fastnl', window=10, cu=0.25, figsize=(15, 8), verbose=3)

    sig_range = np.abs(FinalElevFFTOut.sum(1).sum(1).sum(1)) / np.abs(FinalElevFFTOut.sum(1).sum(1).sum(1)).max()
    results_range = fp_range.fit(sig_range)
    # visData.plotVector(FinalElevFFTOut.sum(1).sum(1).sum(1))
    range_pos_len =  len(results_range["persistence"].y)

    xyz = []


    for range_index_i in range(range_pos_len):
      range_index = results_range["persistence"].y[range_index_i]
      Doppler_Azi_Ele_Data = FinalElevFFTOut[range_index,:,:,:] # (128, 128, 128)
      sig_doppler = np.abs(Doppler_Azi_Ele_Data.sum(1).sum(1))/np.max(np.abs(Doppler_Azi_Ele_Data.sum(1).sum(1)))
      results_doppler = fp_doppler.fit(sig_doppler)

      # visData.plotVector(sig)
      # fp_doppler.plot()
      doppler_pos_len =  len(results_doppler["persistence"].y)
      for doppler_index_i in range(doppler_pos_len):
        doppler_index = results_doppler["persistence"].y[doppler_index_i]
        Azi_Ele_data = Doppler_Azi_Ele_Data[doppler_index, :, :]
        sig_azi = np.abs(Azi_Ele_data.sum(1))/np.max(np.abs(Azi_Ele_data.sum(1)))
        # visData.plotVector(sig_azi)
        results_azi = fp_azi.fit(sig_azi)
        
        azi_pos_len =  len(results_azi["persistence"].y)

        for azi_index_i in range(azi_pos_len):
          azi_index = results_azi["persistence"].y[azi_index_i]
          if azi_index < 10 or azi_index > 110:
             continue
          Ele_data = Azi_Ele_data[azi_index, :]
          sig_ele = np.abs(Ele_data)/np.max(np.abs(Ele_data))
          # visData.plotVector(sig_ele)
          results_ele = fp_ele.fit(sig_ele)

          ele_pos_len =  len(results_ele["persistence"].y)

          for ele_index_i in range(ele_pos_len):
            
            ele_index = results_ele["persistence"].y[ele_index_i]
            if ele_index < 10 or ele_index > 110:
               continue
            # fp_ele.plot()
            range_i = range_index * radar_config.rangeBinSize
            
            sin_theta_i = azi_index/(radar_config.angle_fftsize/2) - 1
            cos_theta_i = math.sqrt(1 - (sin_theta_i * sin_theta_i))
            
            theta_i = math.asin(sin_theta_i)* 180 / math.pi

            sin_phi_i = ele_index/(radar_config.angle_fftsize/2) - 1
            cos_phi_i = math.sqrt(1 - (sin_phi_i * sin_phi_i))
            
            phi_i = math.asin(sin_phi_i)* 180 / math.pi

            x_cor = range_i * cos_phi_i * sin_theta_i

            y_cor = range_i * cos_phi_i * cos_theta_i

            z_cor = range_i * sin_phi_i
            
            intensity = np.abs(FinalElevFFTOut[range_index,doppler_index,azi_index,ele_index])

            xyz.append([x_cor, y_cor, z_cor,intensity])

    # pdb.set_trace()
    xyz = np.array(xyz)

    if xyz.size == 0:
        xyz = np.zeros([1,9])
    # print("Finished pcdOutput!")
    return rangeFFTOut, DopplerFFTOut, xyz

def adc2rangeFFT(radar_adc_data,radar_config):
    rangeFFTOut = np.zeros((radar_config.range_fftsize,radar_config.numChirpsPerFrame,radar_config.numRxChan,radar_config.numTxChan),dtype=complex)

    for i_tx in range(radar_adc_data.shape[3]):
        rangeFFTOut[:,:,:,i_tx] = rangeFFT(radar_adc_data[:,:,:,i_tx], radar_config)
        # rangeFFTOut[0:20,:,:,:] = 0
        # rangeFFTOut[-30:,:,:,:] = 0
        rangeFFTOut[0:int(radar_config.range_fftsize*0.05),:,:,:] = 0
        rangeFFTOut[-int(radar_config.range_fftsize*0.1):,:,:,:] = 0

    return rangeFFTOut


def adc2rangeFFT_multiframe(radar_adc_data,radar_config):
    pdb.set_trace()
    rangeWindowCoeffVec = np.hanning(radar_adc_data.shape[1] + 2)[1:-1]
    radar_adc_data_win = radar_adc_data * rangeWindowCoeffVec 
    rangeFFTOut = np.fft.fft(radar_adc_data_win, n=radar_config.range_fftsize, axis=0)
    rangeFFTOut[0:int(radar_config.range_fftsize * 0.05), :, :, :] = 0
    rangeFFTOut[-int(radar_config.range_fftsize * 0.1):, :, :, :] = 0


    rangeFFTOut = np.zeros((radar_config.range_fftsize,radar_config.numChirpsPerFrame,radar_config.numRxChan,radar_config.numTxChan),dtype=complex)

    for i_tx in range(radar_adc_data.shape[3]):
        rangeFFTOut[:,:,:,i_tx] = rangeFFT(radar_adc_data[:,:,:,i_tx], radar_config)
        # rangeFFTOut[0:20,:,:,:] = 0
        # rangeFFTOut[-30:,:,:,:] = 0
        rangeFFTOut[0:int(radar_config.range_fftsize*0.05),:,:,:] = 0
        rangeFFTOut[-int(radar_config.range_fftsize*0.1):,:,:,:] = 0

    return rangeFFTOut

