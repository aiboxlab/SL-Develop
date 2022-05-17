## Importing necessary libraries
import numpy as np
import tensorflow as tf
import albumentations as A

## Coordinates normalization face size
NORM_WIDTH = 300
NORM_HEIGHT = 300

## Defines the pre-processing transformation
AE_TRANSFORM = A.Compose(
    [
     A.LongestMaxSize(
         NORM_WIDTH,
         p=1)
    ],
    keypoint_params = A.KeypointParams(format='xy', remove_invisible=False),
)

#--------------------------------------------------------------------------------
def coord2Flattened(coords, kind='2d'):
  '''
  Converts coordinates to flattened format.
  e.g.: [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]] ---> [x1,x2,x3,y1,y2,y3,z1,z2,z3]
  '''
  if len(coords.shape) == 2:
    coords = coords.reshape((1, coords.shape[0],  coords.shape[1]))

  data_npy = []
  if kind == '2d':
      for sample in coords:
          aux1 = [x for x,y in sample]
          aux2 = [y for x,y in sample]
          data_npy.append(aux1+aux2)
  elif kind == '3d':
      for sample in coords:
          aux1 = [x for x,y,z in sample]
          aux2 = [y for x,y,z in sample]
          aux3 = [z for x,y,z in sample]
          data_npy.append(aux1+aux2+aux3)
      
  return np.asarray(data_npy)

#--------------------------------------------------------------------------------
def pix2NormArr(arr, img_width, img_height):
  '''
  Converts an array of coordinates (x,y) in pixels to normalized coordinates,
  based on image dimensions.
  '''
  arr_aux = arr.copy()
  arr_aux = arr_aux.astype(float)
  shape = arr_aux.shape

  #We have just an array (i.e. one image). We turn it into an array of arrays (i.e. several images)
  if len(shape) == 2:
     arr_aux = arr_aux.reshape((1,arr_aux.shape[0], arr_aux.shape[1]))

  for i in range(arr_aux.shape[0]):
    arr_aux[i][:,0] = arr_aux[i][:,0]/img_width
    arr_aux[i][:,1] = arr_aux[i][:,1]/img_height

  return arr_aux

#--------------------------------------------------------------------------------
def norm2PixArr(arr, img_width, img_height, limit=True):
  '''
  Converts an normalized array of coordinates to pixel values, based on image dimensions,
  respecting left/up (0) and right/down (img_width/img_height) limits, if limit=True. 
  '''
  aux_arr = arr.copy()

  #If we have just one array, reshape it o be an array of arrays (i.e. an array with one image)
  if len(aux_arr.shape) == 2:
    aux_arr = aux_arr.reshape((1,aux_arr.shape[0], aux_arr.shape[1]))

  #Defines the up/down/left/right image limits  
  width_limit = (img_width-1)*np.ones(aux_arr[0].shape[0]).reshape(-1,1)
  height_limit = (img_height-1)*np.ones(aux_arr[0].shape[0]).reshape(-1,1)
  zero_limit = np.zeros(aux_arr[0].shape[0]).reshape(-1,1)

  if limit:
    #For each sample
    for i in range(aux_arr.shape[0]):
      #Creates an array in which each pair is [coordinate, limit]
      rx = np.hstack((np.floor(aux_arr[i][:,0]*img_width).reshape(-1,1), width_limit))
      ry = np.hstack((np.floor(aux_arr[i][:,1]*img_height).reshape(-1,1), height_limit))
      
      #Gets the minimum between coordinate and limit. It prevents us of
      #getting a coordinate outside the image right/down
      rx = np.min(rx, axis=1)
      ry = np.min(ry, axis=1)

      #Creates an array in which each pair is [coordinate, limit]
      rx = np.hstack((rx.reshape(-1,1), zero_limit))
      ry = np.hstack((ry.reshape(-1,1), zero_limit))

      #Gets the maximum between coordinate and limit. It prevents us of
      #getting a coordinate outside the image left/up
      aux_arr[i][:,0] = np.max(rx, axis=1)
      aux_arr[i][:,1] = np.max(ry, axis=1)
  else:
    #For each sample
    for i in range(aux_arr.shape[0]):
      aux_arr[i][:,0] = np.floor(aux_arr[i][:,0]*img_width)
      aux_arr[i][:,1] = np.floor(aux_arr[i][:,1]*img_height)

  return aux_arr

#--------------------------------------------------------------------------------
def translateToOrigin(orig_coords):
  '''
  Gets an array of coordinates (i.e. one image or more) and translates the
  axis of each sample to the begining of the coordinates (as whether we are
  moving the coordintes to the center of the coordinates plane).
  '''
  coords = orig_coords.copy()

  if len(coords.shape) < 3:
    coords = coords.reshape((1,)+ coords.shape)
  
  new_coords = []
  for sample in coords:
    x_min, y_min = np.min(sample[:,0]), np.min(sample[:,1])
    # print(x_min, y_min)
    sample[:,0] = sample[:,0] - x_min
    sample[:,1] = sample[:,1] - y_min

    new_coords.append(sample)
  
  return np.asarray(new_coords)

#--------------------------------------------------------------------------------
def getSizeFromLandmarks(landmarks):
  '''
  Gets the width and height of the rectangle that is around the landmarks,
  according to the minimum and maximum of each coordinate axis.
  '''
  # x coordintes
  img_width = int(np.max(landmarks[:,0]) - np.min(landmarks[:,0]))
  # y coordinates
  img_height = int(np.max(landmarks[:,1]) - np.min(landmarks[:,1]))
  
  return img_width, img_height

#--------------------------------------------------------------------------------
def preprocess2WayAutoencoder(keypoints_mpp,keypoints_dlib='', size=300, train=True):
  '''
  Implements the whole pre-processing pipeline, which consists in:
     1. Bring all coordinates to the origin of the axis (the minimum x and y coordinates should be on 0).
     2. Resize the 'image' size, so all the points are in the (size, size) desired rectangle.
     3. Normalizes the coordinates according to the desired fixed image size.
  '''
  if train:
    ##Gets the keypoints
    mpp_kpts = keypoints_mpp.copy()
    dlib_kpts = keypoints_dlib.copy()

    ##Translate them to origin
    mpp_translated = translateToOrigin(mpp_kpts)[0]
    dlib_translated = translateToOrigin(dlib_kpts)[0]

    ##Resize them to the desired rectangle
    w,h = getSizeFromLandmarks(mpp_translated)
    mpp_resized = np.asarray(AE_TRANSFORM(image=np.zeros((w,h)), keypoints=mpp_translated)['keypoints'])

    w,h = getSizeFromLandmarks(dlib_translated)
    dlib_resized = np.asarray(AE_TRANSFORM(image=np.zeros((w,h)), keypoints=dlib_translated)['keypoints'])

    ##Normalizes them according to the desired image size
    mpp_norm = pix2NormArr(mpp_resized, size, size)[0]
    dlib_norm = pix2NormArr(dlib_resized, size, size)[0]

    return mpp_norm, dlib_norm
  
  else:
    ##Gets the keypoints
    mpp_kpts = keypoints_mpp.copy()
    ##Translate them to origin
    mpp_translated = translateToOrigin(mpp_kpts)[0]
    ##Resize them to the desired rectangle
    w,h = getSizeFromLandmarks(mpp_translated)
    mpp_resized = np.asarray(AE_TRANSFORM(image=np.zeros((w,h)), keypoints=mpp_translated)['keypoints'])
    ##Normalizes them according to the desired image size
    mpp_norm = pix2NormArr(mpp_resized, size, size)[0]

    return mpp_norm

##-------------------------------------------------------------------------------
def getCoordFromDict(dictionary):
    '''
    Gets a dictionary of coordinates, where keys are indexes and values are (x,y) values
    and turns it into an array of coordinates [x,y].
    '''
    coords = [[dictionary[key][0],dictionary[key][1]] for key in dictionary]
    return np.asarray(coords)

##------------------------------------------------------------------------------------------
def flattened2Coord(feats, kind='2d'):
  '''
  Converts flattened coordinates to coordinate format.
  e.g.: [x1,x2,x3,y1,y2,y3,z1,z2,z3] ---> [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
  '''
  all_coords = []
  if len(feats.shape) == 1:
    feats = feats.reshape((1,feats.shape[0]))
  
  if kind == '2d':
    pad = feats[0].shape[0]//2
    for feat in feats:
      coords = [[feat[i], feat[i+pad]] for i in range(pad)]
      all_coords.append(coords)
  elif kind == '3d':
    pad = feats[0].shape[0]//3
    for feat in feats:
      coords = [[feat[i], feat[i+pad], feat[i+2*pad]] for i in range(pad)]
      all_coords.append(coords)
  
  return np.asarray(all_coords)


if __name__=='__main__':
    dic_example = {0: (58, 99), 1: (57, 85), 2: (57, 90), 3: (53, 74), 4: (57, 81), 5: (57, 76), 6: (56, 67), 7: (32, 69), 8: (56, 58), 9: (56, 53), 10: (55, 40), 11: (58, 101), 12: (58, 103), 13: (58, 103), 14: (58, 104), 15: (58, 106), 16: (58, 108), 17: (58, 110), 18: (58, 114), 19: (57, 87), 20: (54, 88), 21: (21, 60), 22: (42, 71), 23: (38, 72), 24: (35, 72), 25: (31, 71), 26: (45, 70), 27: (37, 62), 28: (41, 62), 29: (34, 63), 30: (31, 64), 31: (29, 73), 32: (45, 121), 33: (31, 68), 34: (20, 74), 35: (25, 71), 36: (40, 86), 37: (53, 99), 38: (54, 103), 39: (49, 101), 40: (47, 103), 41: (51, 103), 42: (49, 104), 43: (42, 109), 44: (54, 85), 45: (53, 81), 46: (26, 60), 47: (46, 76), 48: (46, 86), 49: (46, 84), 50: (30, 86), 51: (53, 77), 52: (34, 55), 53: (29, 57), 54: (24, 52), 55: (49, 57), 56: (44, 63), 57: (40, 106), 58: (24, 108), 59: (49, 88), 60: (52, 89), 61: (44, 106), 62: (45, 106), 63: (27, 55), 64: (46, 88), 65: (40, 55), 66: (39, 52), 67: (37, 42), 68: (26, 54), 69: (38, 47), 70: (25, 60), 71: (23, 60), 72: (54, 101), 73: (50, 102), 74: (48, 104), 75: (50, 89), 76: (45, 106), 77: (46, 106), 78: (46, 105), 79: (51, 86), 80: (50, 104), 81: (52, 104), 82: (55, 104), 83: (54, 114), 84: (54, 110), 85: (54, 108), 86: (54, 106), 87: (55, 104), 88: (49, 105), 89: (49, 105), 90: (48, 106), 91: (47, 108), 92: (43, 99), 93: (21, 91), 94: (57, 88), 95: (47, 105), 96: (47, 106), 97: (53, 90), 98: (48, 90), 99: (52, 90), 100: (43, 79), 101: (38, 82), 102: (45, 86), 103: (29, 47), 104: (30, 49), 105: (32, 52), 106: (45, 111), 107: (47, 52), 108: (46, 47), 109: (45, 40), 110: (32, 72), 111: (25, 76), 112: (46, 69), 113: (28, 66), 114: (49, 73), 115: (48, 84), 116: (22, 79), 117: (27, 78), 118: (32, 79), 119: (38, 78), 120: (43, 76), 121: (46, 74), 122: (53, 68), 123: (23, 87), 124: (26, 66), 125: (55, 87), 126: (47, 79), 127: (20, 76), 128: (49, 72), 129: (45, 87), 130: (30, 69), 131: (48, 82), 132: (22, 100), 133: (46, 68), 134: (51, 79), 135: (31, 115), 136: (31, 119), 137: (20, 89), 138: (27, 110), 139: (22, 67), 140: (44, 125), 141: (56, 88), 142: (44, 83), 143: (23, 73), 144: (35, 70), 145: (38, 70), 146: (45, 107), 147: (24, 94), 148: (51, 131), 149: (40, 127), 150: (36, 124), 151: (56, 47), 152: (59, 131), 153: (41, 70), 154: (44, 69), 155: (45, 68), 156: (24, 66), 157: (43, 66), 158: (40, 65), 159: (38, 65), 160: (35, 66), 161: (33, 67), 162: (20, 67), 163: (33, 70), 164: (57, 93), 165: (46, 96), 166: (49, 87), 167: (53, 94), 168: (56, 63), 169: (35, 119), 170: (39, 122), 171: (51, 128), 172: (28, 115), 173: (45, 67), 174: (51, 73), 175: (59, 128), 176: (45, 129), 177: (22, 97), 178: (52, 105), 179: (51, 105), 180: (51, 107), 181: (50, 109), 182: (49, 113), 183: (47, 105), 184: (46, 105), 185: (45, 104), 186: (41, 102), 187: (28, 95), 188: (51, 70), 189: (49, 65), 190: (47, 66), 191: (48, 105), 192: (28, 104), 193: (52, 64), 194: (47, 117), 195: (56, 73), 196: (53, 71), 197: (56, 70), 198: (49, 79), 199: (58, 124), 200: (58, 119), 201: (52, 119), 202: (39, 111), 203: (43, 90), 204: (43, 115), 205: (35, 90), 206: (40, 94), 207: (32, 97), 208: (51, 123), 209: (47, 82), 210: (36, 114), 211: (40, 118), 212: (36, 107), 213: (25, 100), 214: (32, 108), 215: (23, 104), 216: (37, 100), 217: (49, 76), 218: (49, 85), 219: (47, 87), 220: (51, 82), 221: (46, 61), 222: (40, 59), 223: (36, 59), 224: (32, 60), 225: (29, 62), 226: (28, 70), 227: (20, 82), 228: (30, 75), 229: (34, 75), 230: (38, 75), 231: (42, 74), 232: (46, 72), 233: (48, 71), 234: (20, 84), 235: (48, 88), 236: (51, 76), 237: (52, 85), 238: (54, 87), 239: (52, 85), 240: (49, 89), 241: (54, 87), 242: (55, 88), 243: (48, 68), 244: (50, 68), 245: (51, 68), 246: (32, 68), 247: (30, 66), 248: (59, 74), 249: (81, 68), 250: (60, 87), 251: (91, 58), 252: (71, 71), 253: (75, 71), 254: (78, 71), 255: (82, 70), 256: (69, 70), 257: (75, 62), 258: (72, 62), 259: (78, 62), 260: (81, 64), 261: (85, 72), 262: (72, 121), 263: (82, 67), 264: (94, 72), 265: (88, 70), 266: (74, 84), 267: (62, 98), 268: (62, 102), 269: (66, 100), 270: (69, 101), 271: (65, 103), 272: (67, 103), 273: (74, 108), 274: (60, 85), 275: (60, 81), 276: (85, 60), 277: (67, 76), 278: (68, 85), 279: (68, 83), 280: (84, 85), 281: (60, 77), 282: (78, 55), 283: (82, 57), 284: (88, 51), 285: (63, 57), 286: (69, 63), 287: (76, 104), 288: (93, 107), 289: (66, 87), 290: (63, 88), 291: (73, 105), 292: (71, 104), 293: (84, 55), 294: (68, 87), 295: (72, 55), 296: (72, 52), 297: (74, 41), 298: (86, 53), 299: (73, 46), 300: (87, 59), 301: (89, 59), 302: (62, 101), 303: (65, 101), 304: (68, 102), 305: (65, 88), 306: (72, 104), 307: (70, 105), 308: (71, 104), 309: (64, 85), 310: (67, 103), 311: (64, 103), 312: (61, 103), 313: (63, 114), 314: (63, 110), 315: (62, 107), 316: (62, 105), 317: (62, 104), 318: (67, 104), 319: (68, 104), 320: (69, 105), 321: (69, 107), 322: (72, 97), 323: (95, 90), 324: (69, 104), 325: (70, 104), 326: (62, 90), 327: (67, 90), 328: (62, 89), 329: (71, 78), 330: (76, 80), 331: (69, 85), 332: (82, 45), 333: (81, 48), 334: (79, 52), 335: (71, 110), 336: (64, 53), 337: (65, 46), 338: (66, 40), 339: (81, 71), 340: (88, 75), 341: (67, 69), 342: (84, 65), 343: (65, 73), 344: (66, 83), 345: (92, 78), 346: (86, 77), 347: (82, 78), 348: (76, 77), 349: (71, 75), 350: (67, 73), 351: (60, 68), 352: (92, 85), 353: (87, 65), 354: (59, 87), 355: (67, 79), 356: (95, 74), 357: (64, 71), 358: (69, 86), 359: (83, 68), 360: (66, 81), 361: (95, 98), 362: (67, 67), 363: (63, 79), 364: (86, 113), 365: (86, 118), 366: (95, 87), 367: (90, 109), 368: (92, 65), 369: (73, 125), 370: (59, 88), 371: (70, 82), 372: (91, 71), 373: (78, 69), 374: (75, 69), 375: (71, 106), 376: (91, 92), 377: (66, 131), 378: (77, 126), 379: (82, 123), 380: (72, 69), 381: (69, 68), 382: (68, 68), 383: (89, 64), 384: (69, 66), 385: (72, 65), 386: (75, 65), 387: (78, 65), 388: (80, 66), 389: (93, 65), 390: (79, 69), 391: (69, 95), 392: (66, 86), 393: (62, 93), 394: (82, 118), 395: (78, 121), 396: (66, 127), 397: (90, 114), 398: (67, 67), 399: (62, 73), 400: (72, 129), 401: (94, 95), 402: (65, 104), 403: (65, 105), 404: (66, 106), 405: (66, 108), 406: (68, 112), 407: (69, 104), 408: (71, 104), 409: (72, 103), 410: (75, 101), 411: (87, 93), 412: (62, 70), 413: (64, 64), 414: (66, 65), 415: (69, 104), 416: (88, 102), 417: (61, 64), 418: (70, 116), 419: (60, 71), 420: (65, 79), 421: (64, 118), 422: (78, 110), 423: (72, 89), 424: (74, 114), 425: (80, 89), 426: (75, 93), 427: (83, 95), 428: (66, 123), 429: (67, 81), 430: (81, 113), 431: (76, 117), 432: (80, 105), 433: (91, 98), 434: (84, 106), 435: (93, 102), 436: (78, 98), 437: (64, 76), 438: (65, 84), 439: (67, 86), 440: (63, 82), 441: (66, 61), 442: (72, 59), 443: (76, 59), 444: (80, 60), 445: (83, 62), 446: (85, 69), 447: (95, 80), 448: (83, 73), 449: (80, 74), 450: (75, 74), 451: (71, 73), 452: (68, 72), 453: (65, 70), 454: (95, 82), 455: (67, 87), 456: (62, 76), 457: (62, 84), 458: (60, 86), 459: (62, 85), 460: (66, 89), 461: (60, 87), 462: (60, 88), 463: (65, 67), 464: (63, 68), 465: (62, 68), 466: (81, 67), 467: (83, 65)}
    coords = getCoordFromDict(dic_example)
    print(coords)
