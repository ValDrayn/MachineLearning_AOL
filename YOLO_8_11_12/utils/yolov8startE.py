import os
import shutil
import yaml
from functions import parse_opt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def update_yaml(yaml_path, dataset_path):
    with open(yaml_path, 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
        yaml_file['path'] = dataset_path
        yaml_file['train'] = os.path.join(dataset_path, 'train/images')
        yaml_file['val'] = os.path.join(dataset_path, 'val/images')
        yaml_file['test'] = os.path.join(dataset_path, 'test/images')

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_file, f)

def copy_data_to_yolov8(repo_path, yolov8_path):
    src_data_path = os.path.join(repo_path, 'data')
    dst_data_path = os.path.join(yolov8_path, 'data')

    # Hapus folder data lama jika ada
    if os.path.exists(dst_data_path):
        shutil.rmtree(dst_data_path)

    # Salin folder data
    shutil.copytree(src_data_path, dst_data_path)
    print(f"Data copied to {dst_data_path} ✅")

    # Salin dataset.yaml
    shutil.copyfile(os.path.join(dst_data_path, 'dataset.yaml'), os.path.join(yolov8_path, 'dataset.yaml'))
    os.remove(os.path.join(dst_data_path, 'dataset.yaml'))

    # Update YAML agar path sesuai
    update_yaml(os.path.join(yolov8_path, 'dataset.yaml'), dst_data_path)

def main(opt):
    repo_path = opt.reporoot
    yolov8_path = os.path.join(repo_path, 'yolov8')

    print(f"Repo path: {repo_path}")
    print(f"Yolo path: {yolov8_path}")

    copy_data_to_yolov8(repo_path, yolov8_path)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
