import tensorflow as tf
import matplotlib.pyplot as plt

with tf.Session() as sess:
  x_vals = tf.linspace(-1., 1., 500)
  target = tf.constant(0.)

  l2_y_vals = tf.square(target - x_vals)
  l2_y_out = sess.run(l2_y_vals)
  x_vals_out = sess.run(x_vals)

  # print("l2_y_out :", l2_y_out)
  # print("x_vals_out :", x_vals_out)

  l1_y_vals = tf.abs(target - x_vals)
  l1_y_out = sess.run(l1_y_vals)

  delta1 = tf.constant(0.25)
  phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals) / delta1)) - 1.)

  phuber1_y_out = sess.run(phuber1_y_vals)
  delta2 = tf.constant(5.)

  phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals) / delta2)) - 1.)
  phuber2_y_out = sess.run(phuber2_y_vals)

  #print("l1_y_out :", l1_y_out)

  x_vals = tf.linspace(-3., 5., 500)
  target = tf.constant(1.)
  targets = tf.fill([500,], 1.)

  hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
  hinge_y_out = sess.run(hinge_y_vals)

  #print("hinge_y_out :", hinge_y_out)

  xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
  xentropy_y_out = sess.run(xentropy_y_vals)

  #print("xentropy_y_out :", xentropy_y_out)

  x_val_input = tf.expand_dims(x_vals, 1)
  target_input = tf.expand_dims(targets, 1)
  x_in, tar_in = sess.run([x_val_input, target_input])
  # print("x_in :", x_in)
  # print("tar_in :", tar_in)
  xentropy_sigmoid_y_vals = tf.nn.softmax_cross_entropy_with_logits(logits=x_val_input, labels = target_input)
  xentropy_sigmoid_y_vals = sess.run(xentropy_sigmoid_y_vals)

  # print("xentropy_sigmoid_y_vals :", xentropy_sigmoid_y_vals)

  # 소프트맥스 교차 엔트로피
  """
  정규화되지 않은 출력 값을 대상으로 한다 이 함수는 여럿이 아닌 하나의 분류 대상에 대한 비용을 측정할 때 사용한다
  이 때문에 이 함수는 softmax 함수를 이용해 결과 값으 ㄹ확률 분포로 변환하고 실제 확률 분포와 비교하는 방식으로 비용을 계산한다
  """
  unscaled_logits = tf.constant([[1., -3., 10.]])
  target_dist = tf.constant([[0.1, 0.02, 0.88]])
  softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=target_dist)
  print(sess.run(softmax_xentropy))

  # 희소 소프트맥스 교차 엔트로피 비용 함수
  # Sparse Softmax Cross Entropy
  """
  앞의 소프트맥스 교차 엔트로피 함수가 확률 분포를 대상으로 하는 것과 달리 실제 속한 분류가 어디인지를 표시한 지표를 대상으로 한다
  모든 원소 값이 0이고 한 원소만 1인 대상 값 벡터를 사용하는 대신 다음과 같이 어떤 분류가 실제 값인지를 나타내는 지표만 전달한다
  """

  unscaled_logits = tf.constant([[1., -3., 10.]])
  sparse_target_dist = tf.constant([2])
  sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=sparse_target_dist)
  sparse_xentropy_out = sess.run(sparse_xentropy)
  print("sparse_xentropy :", sparse_xentropy_out)

  # 다음은 matplotlib을 이용해 회귀 비용 함수를 그리는 코드다

  # x_array = sess.run(x_vals)
  # plt.plot(x_array, l2_y_out, 'b-', label='L2 loss')
  # plt.plot(x_array, l1_y_out, 'r--', label='L1 loss')
  # plt.plot(x_array, phuber1_y_out, 'k-', label='P-Huber Loss (0.25)')
  # plt.plot(x_array, phuber2_y_out, 'g:', label='P-Huber Loss (5.0)')
  # plt.ylim(-0.2, 0.4)
  # plt.legend(loc='lower right', prop={'size':11})
  # plt.show()
  #
  #
  nsteps = 2
  nprocs = 8
  nscripts = 4
  print(sess.run(tf.concat([tf.zeros([nscripts * nsteps, 1]),tf.ones([(nprocs - nscripts) * nsteps, 1])],axis=0)))



