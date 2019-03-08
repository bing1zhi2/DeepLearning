import tensorflow as tf
import numpy as np
import mnist_input_data
'''
[dl35] F:\code\mycode\tensorflow_learn\src\tensorflow_pra\serving\model>saved_model_cli show --dir 1 --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['predict_images']:
The given SavedModel SignatureDef contains the following input(s):
inputs['images'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 784)
    name: x:0
The given SavedModel SignatureDef contains the following output(s):
outputs['scores'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 10)
    name: y:0
Method name is: tensorflow/serving/predict

signature_def['serving_default']:
The given SavedModel SignatureDef contains the following input(s):
inputs['inputs'] tensor_info:
    dtype: DT_STRING
    shape: unknown_rank
    name: tf_example:0
The given SavedModel SignatureDef contains the following output(s):
outputs['classes'] tensor_info:
    dtype: DT_STRING
    shape: (-1, 10)
    name: index_to_string_Lookup:0
outputs['scores'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 10)
    name: TopKV2:0
Method name is: tensorflow/serving/classify
'''

export_dir = 'model/1/'

mnist = mnist_input_data.read_data_sets('/tmp', one_hot=True)
image0= mnist.test.images[0]
batch0 = mnist.test.next_batch(1)
print(image0.shape)

with tf.Session(graph=tf.Graph()) as sess:
  metagraph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

  signature_def = metagraph_def.signature_def[
      'predict_images']
  input_tensor = sess.graph.get_tensor_by_name(
      signature_def.inputs['images'].name)
  output_tensor = sess.graph.get_tensor_by_name(
      signature_def.outputs['scores'].name)


  def forward_images(images):
      images = images.astype(np.float32) / 255.0
      images = output_tensor.eval(feed_dict={input_tensor: images})
      return images


  # image0 =np.reshape(image0,(-1,784))
  outimage = forward_images(batch0[0])
  print(outimage)

