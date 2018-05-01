
if i % embedding_log == 0:  # print every embedding_log mini-batches
    print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 100))
    running_loss = 0.0
    writer.add_embedding(output1.data, metadata=label1.data, label_img=input1.data, global_step=2 * n_iter)
    writer.add_embedding(output2.data, metadata=label2.data, label_img=input2.data, global_step=2 * n_iter + 1)