

import skimage.data
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
import cv2

# Number of batches and batch size for this example
nb_batches = 10
batch_size = 32


#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))
size = (200, 200)
result = 0
path_save = '~/Documentos/projetos/env_cliente/libras/FacialActionLibras/data/outputs/teste-mari/'
result = cv2.VideoWriter(
    #path_save, cv2.VideoWriter_fourcc('*"MJPG"'), 30, size, isColor=False #for codec origin MPEG-4 Video
    path_save, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, size, isColor=False #for codec origin JPEG
)

# Example augmentation sequence to run in the background
augseq = iaa.Sequential([
    iaa.JpegCompression(compression=(10, 10))
])

# For simplicity, we use the same image here many times
astronaut = skimage.data.astronaut()
astronaut = ia.imresize_single_image(astronaut, (64, 64))

# Make batches out of the example image (here: 10 batches, each 32 times
# the example image)
batches = []
for _ in range(nb_batches):
    batches.append(UnnormalizedBatch(images=[astronaut] * batch_size))

# Show the augmented images.
# Note that augment_batches() returns a generator.
for images_aug in augseq.augment_batches(batches, background=True):
    ia.imshow(ia.draw_grid(images_aug.images_aug, cols=8))
    #print(type(images_aug))
    #media.write_video(path_save, images_aug.numpy(), fps=30)
