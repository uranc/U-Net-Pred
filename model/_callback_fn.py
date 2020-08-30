import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import os


class TensorBoardCustom(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoardCustom, self).__init__(**kwargs)
        self.mean = K.constant([103.939, 116.779, 123.68],
                               dtype=tf.float32,
                               shape=[1, 1, 1, 3],
                               name='img_mean')  # BGR

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if not self.validation_data and self.histogram_freq:
            raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                # We need a second forward-pass here because we're passing
                # the `embeddings_data` explicitly. This design allows to pass
                # arbitrary data as `embeddings_data` and results from the fact
                # that we need to know the size of the `tf.Variable`s which
                # hold the embeddings in `set_model`. At this point, however,
                # the `validation_data` is not yet set.

                # More details in this discussion:
                # https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

                embeddings_data = self.embeddings_data
                n_samples = embeddings_data[0].shape[0]

                i = 0
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)

                    if type(self.model.input) == list:
                        feed_dict = {_input: embeddings_data[idx][batch]
                                     for idx, _input in enumerate(self.model.input)}
                    else:
                        feed_dict = {
                            self.model.input: embeddings_data[0][batch]}

                    feed_dict.update({self.batch_id: i, self.step: step})

                    if self.model.uses_learning_phase:
                        feed_dict[K.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir,
                                                 'keras_embedding.ckpt'),
                                    epoch)

                    i += self.batch_size

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        if (index % 50) == 0:
            self._write_logs(logs, index)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = self.model.mode + '_' + name
            self.writer.add_summary(summary, index)
            # print(y_pred)
        # def tf_summary_image(tensor):
        #     import io
        #     from PIL import Image

        #     tensor = tensor.astype(np.uint8)
        #     ba, hi, wi, ch = tensor.shape
        #     # image = Image.fromarray(tensor[img_id, :, :, ::-1])
        #     image = Image.fromarray(tensor[img_id, :, :, :])
        #     output = io.BytesIO()
        #     image.save(output, format='PNG')
        #     image_string = output.getvalue()
        #     output.close()
        #     return tf.Summary.Image(height=hi,
        #                             width=wi,
        #                             colorspace=ch,
        #                             encoded_image_string=image_string)
        # # cem hacks
        # # im_out = tf.clip_by_value(self.model.output+self.mean, 0, 255)
        # im_out = tf.clip_by_value(self.model.output, 0, 255)
        # # lab_out = self.model.targets  # + self.mean
        # with K.get_session().as_default() as ses:
        #     img_out = ses.run([im_out])
        #     # img_out, lab_out = ses.run([im_out, lab_out])
        # # lab_out = lab_out[:, :, :, ::-1]
        # im_summaries = []
        # for img_id in range(img_out.shape[0]):
        #     img_sum = tf_summary_image(img_out)
        #     # lab_sum = tf_summary_image(lab_out)
        #     # Create a Summary value
        #     im_summaries.append(tf.Summary.Value(
        #         tag=self.model.mode+'_images_'+str(img_id), image=img_sum))
        #     # im_summaries.append(tf.Summary.Value(
        #     #     tag=self.model.mode+'_labels_'+str(img_id), image=lab_sum))

        # # Create and write Summary
        # summary = tf.Summary(value=im_summaries)
        # self.writer.add_summary(summary, index)

        # hackend
        self.writer.flush()
