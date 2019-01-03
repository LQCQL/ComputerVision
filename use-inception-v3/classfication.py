from NodeLookup import *
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    with tf.gfile.GFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        # 遍历目录
        for root, dirs, files in os.walk('images/'):
            for file in files:
                # 载入图片
                image_data = tf.gfile.GFile(os.path.join(root, file), 'rb').read()
                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data}) # 图片格式为jpeg格式
                predictions = np.squeeze(predictions) # 将结果转为1维数据
                print(predictions.shape)

                # 打印图片路径及名称
                image_path = os.path.join(root, file)
                print(image_path)
                # 显示图片
                img = Image.open(image_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show()

                # 排序
                top_k = predictions.argsort()[-5:][::-1]
                node_lookup = NodeLookup()
                for node_id in top_k:
                    # 获取分类名称
                    human_string = node_lookup.id_to_string(node_id)
                    # 获取该分类的置信度
                    score = predictions[node_id]
                    print('%s (score = %.5f)' % (human_string, score))
                print()

if __name__ == '__main__':
    main()