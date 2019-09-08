from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from head.utils import ImageCoder, ProgressBar,\
     make_multi_crop_batch, make_multi_image_batch, face_detection_model
from head.detect import FACE_PAD
import os
import json
import csv
import cv2

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

# tf.app.flags.DEFINE_string('model_dir', '',
#                            'Model directory (where training data lives)')

# tf.app.flags.DEFINE_string('class_type', 'age',
#                            'Classification type (age|gender)')


# tf.app.flags.DEFINE_string('device_id', '/cpu:0',
#                            'What processing unit to execute inference on')

# tf.app.flags.DEFINE_string('filename', '',
#                            'File (Image) or File list (Text/No header TSV) to process')

# tf.app.flags.DEFINE_string('target', '',
#                            'CSV file containing the filename processed along with best guess and score')

# tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
#                           'Checkpoint basename')

# tf.app.flags.DEFINE_string('model_type', 'default',
#                            'Type of convnet')

# tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

# tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

# tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

# tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')
FLAGS = {}
FLAGS.model_dir= './model/' #'Model directory (where training data lives)')
FLAGS.class_type= 'age',# Classification type (age|gender)')
FLAGS.device_id= '/cpu:0',# 'What processing unit to execute inference on')
# FLAGS.filename= ''# 'File (Image) or File list (Text/No header TSV) to process')
FLAGS.target= './output/'#  'CSV file containing the filename processed along with best guess and score')
FLAGS.checkpoint= 'checkpoint'# Checkpoint basename')
FLAGS.model_type= 'default'# 'Type of convnet')
FLAGS.requested_step= '' # 'Within the model directory, a requested step to restore e.g., 9000')
FLAGS.single_look=  False # 'single look at the image or multiple crops')
FLAGS.face_detection_model= ''# 'Do frontal face detection with model specified')
FLAGS.face_detection_type='cascade'
FLAGS.pip = 0.175
FLAGS.pip_off = 0.01

          



def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])

def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None


def classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer):
    try:
        num_batches = math.ceil(len(image_files) / MAX_BATCH_SZ)
        pg = ProgressBar(num_batches)
        for j in range(num_batches):
            start_offset = j * MAX_BATCH_SZ
            end_offset = min((j + 1) * MAX_BATCH_SZ, len(image_files))
            
            batch_image_files = image_files[start_offset:end_offset]
            print(start_offset, end_offset, len(batch_image_files))
            image_batch = make_multi_image_batch(batch_image_files, coder)
            batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
            batch_sz = batch_results.shape[0]
            for i in range(batch_sz):
                output_i = batch_results[i]
                best_i = np.argmax(output_i)
                best_choice = (label_list[best_i], output_i[best_i])
                print('Guess @ 1 %s, prob = %.2f' % best_choice)
                if writer is not None:
                    f = batch_image_files[i]
                    writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
            pg.update()
        pg.done()
    except Exception as e:
        print(e)
        print('Failed to run all images')

def classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file, writer):
    try:

        print('Running file %s' % image_file)
        image_batch = make_multi_crop_batch(image_file, coder)

        batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]
    
        for i in range(1, batch_sz):
            output = output + batch_results[i]
        
        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)
    
        nlabels = len(label_list)
        if nlabels > 2:
            output[best] = 0
            second_best = np.argmax(output)
            print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

        if writer is not None:
            writer.writerow((image_file, best_choice[0], '%.2f' % best_choice[1]))
    except Exception as e:
        print(e)
        print('Failed to run image %s ' % image_file)

def list_images(srcfile):
    with open(srcfile, 'r') as csvfile:
        delim = ',' if srcfile.endswith('.csv') else '\t'
        reader = csv.reader(csvfile, delimiter=delim)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            print('skipping header')
            _ = next(reader)
        
        return [row[0] for row in reader]

class DETECT : 
    model_dir: str
    class_type: str 
    device_id: str
    filename: str
    target: str 
    checkpoint: str
    model_type: str 
    requested_step: str
    single_look: bool
    face_detection_model: str 
    face_detection_type: str
    pip :int
    pip_off : int
    def __init__(self,img, **kwargs):
        self.__dict__.update(FLAGS) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.dim =( img.shape[0],img.shape[1])
        self.pip_dim = int(self.pip*self.dim[0])
        self.pip_off = int(self.pip_off*self.dim[0])
        self.pip_rows = self.dim[0] // self.pip_dim


    def annotate(self,img, faces, rectangles):
        for i in range(len(faces)):
            n_col = i  //self.pip_rows
            n_row =  i % self.pip_rows
            x = self.dim[0] - (n_col+1)*(self.pip_dim +self.pip_off)  
            y = self.pip_off+ n_row * (self.pip_dim +self.pip_off) 
            pip = cv2.resize(faces[i],(self.pip_dim,self.pip_dim))
            img[x:x+self.pip_dim,y:y+self.pip_dim,:]=pip
            x, y, w, h = rectangles[i]
            self.draw_rect(img,x, y, w, h)
        return img
            
    def draw_rect(self, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        cv2.rectangle(img, (lower_cut[1], lower_cut[0]), (upper_cut[1], upper_cut[0]), (255, 0, 0), 2)


    def main(self, img):  # pylint: disable=unused-argument

        files = []
        
        if self.face_detection_model:
            print('Using face detector (%s) %s' % (self.face_detection_type, self.face_detection_model))
            face_detect = face_detection_model(self.face_detection_type, self.face_detection_model)
            face_files, rectangles = face_detect.run(img)
            print(face_files)
            self.annotate(img,face_files,rectangles)
            files += face_files

        # config = tf.ConfigProto(allow_soft_placement=True)
        # with tf.Session(config=config) as sess:

        #     label_list = AGE_LIST if self.class_type == 'age' else GENDER_LIST
        #     nlabels = len(label_list)

        #     print('Executing on %s' % self.device_id)
        #     model_fn = select_model(self.model_type)

        #     with tf.device(self.device_id):
                
        #         images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
        #         logits = model_fn(nlabels, images, 1, False)
        #         init = tf.global_variables_initializer()
                
        #         requested_step = self.requested_step if self.requested_step else None
            
        #         checkpoint_path = '%s' % (self.model_dir)

        #         model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, self.checkpoint)
                
        #         saver = tf.train.Saver()
        #         saver.restore(sess, model_checkpoint_path)
                            
        #         softmax_output = tf.nn.softmax(logits)

        #         coder = ImageCoder()

        #         # Support a batch mode if no face detection model
        #         if len(files) == 0:
        #             if (os.path.isdir(self.filename)):
        #                 for relpath in os.listdir(self.filename):
        #                     abspath = os.path.join(self.filename, relpath)
                            
        #                     if os.path.isfile(abspath) and any([abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
        #                         print(abspath)
        #                         files.append(abspath)
        #             else:
        #                 files.append(self.filename)
        #                 # If it happens to be a list file, read the list and clobber the files
        #                 if any([self.filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
        #                     files = list_images(self.filename)
                    
        #         writer = None
        #         output = None
        #         if self.target:
        #             print('Creating output file %s' % self.target)
        #             output = open(self.target, 'w')
        #             writer = csv.writer(output)
        #             writer.writerow(('file', 'label', 'score'))
        #         image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
        #         print(image_files)
        #         if self.single_look:
        #             classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer)

        #         else:
        #             for image_file in image_files:
        #                 classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file, writer)

        #         if output is not None:
        #             output.close()
        
if __name__ == '__main__':
    tf.app.run()
