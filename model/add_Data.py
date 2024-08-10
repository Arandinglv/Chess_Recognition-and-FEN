import os
import shutil


class Merge:
    def __init__(self, src_path, dst_path):
        self.src_path = src_path
        self.dst_path = dst_path

    def merge_dataset(self):
        for class_folder in os.listdir(self.src_path):
            src_class_folder = os.path.join(self.src_path, class_folder)
            dst_class_folder = os.path.join(self.dst_path, class_folder)

            for image_file in os.listdir(src_class_folder):
                src_image_path = os.path.join(src_class_folder, image_file)
                dst_image_path = os.path.join(dst_class_folder, image_file)

                shutil.copy2(src_image_path, dst_image_path)
                print(f"Copy file from {src_image_path} -> {dst_image_path}")

        print("Success! ")

merge = Merge("../dataset/Rename", "../dataset/valid_ImageDataset")
merge.merge_dataset()


