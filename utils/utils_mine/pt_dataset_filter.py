# A stupid way to generate the image is to filter the image manually, then generate a NeRF-W style csv
import csv
from argparse import ArgumentParser
import os
import sys



def orgnize_data_frame(train_id_list, test_id_list, scene_short_name):
    for itm in test_id_list:
        print(itm)
        train_id_list.remove(itm)
    print('Length of the train list:', len(train_id_list))
    print('Length of the test list:', len(test_id_list))
    data_frame_list = [["filename", "id", "split", "dataset"]]
    for itm in test_id_list:
        data_frame = [itm, 0, "test", scene_short_name]
        data_frame_list.append(data_frame)
    for itm in train_id_list:
        data_frame = [itm, 0, "train", scene_short_name]
        data_frame_list.append(data_frame)
    return data_frame_list


def write_csv(data_frame_list, save_csv_path):
    with open(save_csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file,delimiter='\t')
        writer.writerows(data_frame_list)
if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_path', type=str, default="/media/wangyz/DATA/UBUNTU_data/dataset/PT/temple_nara_japan")

    args = parser.parse_args(sys.argv[1:])
    img_folder_path = os.path.join(args.source_path,'dense','images')
    test_id_txt_path = os.path.join(args.source_path,"test_id.txt")
    img_list = os.listdir(img_folder_path)

    # 读取文件内容并按行存储到列表中
    with open(test_id_txt_path, 'r', encoding='utf-8') as file:
        test_id_list = file.read().splitlines()
    scene_short_name = args.source_path.split('/')[-1].split('_')[0]
    save_csv_path = os.path.join(args.source_path,scene_short_name+'.tsv')

    data_frame_list = orgnize_data_frame(img_list, test_id_list, scene_short_name)
    print(data_frame_list)
    write_csv(data_frame_list, save_csv_path)

