# def tower_loss(scope):
#     images, labels = mnist_inputs(min_after_dequeue=MIN_AFTER_DEQUEUE)
#     logits = mnist.inference(images)
#     _ = mnist.loss(logits, labels)
#     loss = tf.get_collection("loss", scope=scope)[0]
#     return loss
#
#
# def averaged_gradients(tower_gradients):
#     gradients_mean = []
#     for gradient_variable in zip(*tower_gradients):
#         gradients = []
#         for gradient, _ in gradient_variable:
#             expanded_gradient = tf.expand_dims(gradient, 0)
#             gradients.append(expanded_gradient)
#         concat = tf.concat(0, gradients)
#         averaged_gradient = tf.reduce_mean(concat, reduction_indices=0)
#         grad_var_tuple = (averaged_gradient, gradient_variable[0][1])
#         gradients_mean.append(grad_var_tuple)
#     return gradients_mean