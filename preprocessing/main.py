import preprocessing as pp
import data as dat
import tsne as t
import time

path_train = '/home/barbara/Documents/Trabalho/train-jpg/'
new_path_train = '/home/barbara/Documents/Trabalho/train-jpg_resized/'
path_label = '/home/barbara/Documents/Trabalho/train_v2.csv'
path_tsne = 'tsne10'

#Resizing images 32x32
#li.resize_images(path_train, new_path_train, 32)

#Loading images
start_time = time.time()
data = dat.load_images(new_path_train)
print 'Loaded in ' + str(time.time()-start_time) + 's'

# Preprocessing
data = data.astype('float32')
data /= 255.  
data = pp.st_scale(data)
data = pp.normalize_l2(data)
data, i = pp.PCA_reduction(data, 0, 10)

#Loading labels
label = dat.load_labels(path_label)

#Generate t-SNE
start_time = time.time()
t.generate_tsne(path_tsne, data, label)
print 'Generated in ' + str(time.time()-start_time) + 's'

#Generate a histogram of labels (?)
