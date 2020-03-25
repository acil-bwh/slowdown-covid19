import matplotlib.pyplot as plt
import numpy as np

from skimage import exposure, filters
from matplotlib import image

from os import listdir
from os.path import isfile, join

def equalize(img, clip_limit=0.01, med_filt=5,output_type='uint16', flag_draw=False,):
    if img.dtype is np.dtype(np.float32):
        img_norm = img / img.max()                                    # Format adaptation
    else:
        img_norm = img.astype('float32') / np.iinfo(img.dtype).max                  # Format adaptation
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=clip_limit)        # CLAHE
    img_clahe_median = filters.median(img_clahe,np.ones((5,5))).astype('float32')   # Median Filter

    lower, upper = np.percentile(img_clahe_median.flatten(), [2, 98])
    img_clip = np.clip(img_clahe_median,lower, upper)
    img_out = (img_clip - lower)/(upper - lower)

    if output_type is not None:
       max_val=np.iinfo(output_type).max
       img_out=(max_val*img_out).astype(output_type)
    else:
      max_val=1.0

    if flag_draw is True:
        # plt.imshow(np.concatenate([img, img_clahe, img_clahe_median / img_clahe_median.max(), img_out],axis=1),cmap='gray')
        plt.imshow(np.concatenate([max_val*img_norm, img_out],axis=1),cmap='gray')
        plt.axis('off')
        plt.tight_layout()

        plt.figure(50)
        plt.clf()
        plt.hist(img_norm.flatten(), 256, alpha=0.2, density=True, label='original')
        plt.hist(img_clahe.flatten(), 256, alpha=0.2, density=True, label='clahe')
        plt.hist(img_out.flatten(),256, alpha=0.2, density=True, label='clahe_clip')
        plt.legend()
        plt.axis(xmin=0., xmax=1)
        plt.show()

    return img_out


#basedir='/Users/rjosest/projects/COVID19/COVID_project/'
basedir='Users/gvegas/data/earlyCOVID-19/'
dir_name = basedir+'CheXpert/mild/'
cxr_files_chexpert = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]

dir_name = basedir+'PADCHEST/moderate-severe/'
cxr_files_padchest = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]

dir_name = basedir+'NIHDeepLesion/moderate-severe/'
cxr_files_nihdeeplesion = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]


img = image.imread(cxr_files_chexpert[3])
img2 = equalize(img, flag_draw=True)

img = image.imread(cxr_files_padchest[3])
img2 = equalize(img, flag_draw=True)

img = image.imread(cxr_files_nihdeeplesion[0])
img2 = equalize(img, flag_draw=True)


print(img2.min())
print(img2.max())
